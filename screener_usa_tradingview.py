"""
screener_usa_tradingview.py

Stock screener USA:
- Query TradingView (tradingview-screener)
- Arricchimento via yfinance
- Top 5 con score
- Invio email via common_utility.mailer (secrets in GitHub Actions)

Richiesta chiave:
- NON fermarti se un campo TradingView non esiste.
  => rimuovi campo + eventuali filtri collegati e riprova.
- In fondo alla mail aggiungi la lista dei campi rimossi/mancanti.
"""

import time
import re
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from tradingview_screener import Query, Column
import yfinance as yf

# IMPORT UNICO: niente doppioni tipo "from mailer import ..."
from common_utility.mailer import send_email


# ============================================================
# UTILS
# ============================================================

def safe_get(val, default=0.0) -> float:
    """Convertitore robusto per numeri (None/NaN/stringhe vuote -> default)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val) if val != "" else default
    except Exception:
        return default


def extract_unknown_field(err: Exception) -> str | None:
    """
    Estrae il nome del campo sconosciuto dal messaggio TradingView.
    Esempio body: Unknown field "earnings_per_share_diluted_yoy"
    """
    s = str(err)
    m = re.search(r'Unknown field\s+\\"([^\\"]+)\\"', s)
    if not m:
        m = re.search(r'Unknown field\s+"([^"]+)"', s)
    return m.group(1) if m else None


# ============================================================
# 1) TRADINGVIEW SCREENER con retry su campi sconosciuti
# ============================================================

def run_tradingview_screener() -> tuple[pd.DataFrame, list[str]]:
    """
    Esegue screener TradingView.
    Se TradingView risponde "Unknown field", rimuove quel campo e riprova.

    Ritorna:
    - df risultati
    - lista campi rimossi/mancanti (da mettere in fondo alla mail)
    """

    print("\n🔍 Interrogo TradingView Screener...")
    print("   (USA, Large Cap, Momentum + Quality)")

    removed_fields: list[str] = []

    # Campi che proviamo a selezionare (se qualcuno non esiste, lo togliamo)
    select_fields = [
        "name", "close",
        "market_cap_basic",
        "price_earnings_ttm",
        "earnings_per_share_diluted_yoy_growth_ttm",
        "total_revenue_yoy_growth_ttm",
        "dividends_yield_current",
        "return_on_equity",
        "return_on_invested_capital",
        "operating_margin",
        "debt_to_equity",
        "Perf.6M",
        "Perf.3M",
        "RSI",
        "volume",
        "sector",
        "SMA200",
        "SMA50",
    ]

    # Filtri: li teniamo “agganciati” ai campi per poterli togliere se il campo non esiste
    def build_where(active_fields: set[str]):
        w = []

        # Size
        if "market_cap_basic" in active_fields:
            w.append(Column("market_cap_basic") > 10_000_000_000)

        # Growth
        if "earnings_per_share_diluted_yoy_growth_ttm" in active_fields:
            w.append(Column("earnings_per_share_diluted_yoy_growth_ttm") > 10)
        if "total_revenue_yoy_growth_ttm" in active_fields:
            w.append(Column("total_revenue_yoy_growth_ttm") > 10)

        # Momentum
        if "Perf.6M" in active_fields:
            w.append(Column("Perf.6M") > 5)
        if "Perf.3M" in active_fields:
            w.append(Column("Perf.3M") > 3)

        # Quality
        if "return_on_equity" in active_fields:
            w.append(Column("return_on_equity") > 15)
        if "return_on_invested_capital" in active_fields:
            w.append(Column("return_on_invested_capital") > 12)
        if "operating_margin" in active_fields:
            w.append(Column("operating_margin") > 10)

        # Leverage
        if "debt_to_equity" in active_fields:
            w.append(Column("debt_to_equity") < 1)

        # Trend (richiede SMA)
        if "close" in active_fields and "SMA200" in active_fields:
            w.append(Column("close") > Column("SMA200"))
        if "close" in active_fields and "SMA50" in active_fields:
            w.append(Column("close") > Column("SMA50"))

        # RSI
        if "RSI" in active_fields:
            w.append(Column("RSI") >= 45)
            w.append(Column("RSI") <= 75)

        # Liquidity
        if "volume" in active_fields:
            w.append(Column("volume") > 2_000_000)

        # Valuation
        if "price_earnings_ttm" in active_fields:
            w.append(Column("price_earnings_ttm") >= 10)
            w.append(Column("price_earnings_ttm") <= 45)

        return w

    # Retry loop
    active = set(select_fields)

    for attempt in range(1, 11):
        try:
            where_clauses = build_where(active)

            q = (
                Query()
                .set_markets("america")
                .select(*[f for f in select_fields if f in active])
                .where(*where_clauses)
                .order_by("Perf.6M", ascending=False)  # se Perf.6M manca, TradingView potrebbe comunque accettare, ma in caso la togliamo
                .limit(50)
            )

            _, df = q.get_scanner_data()
            print(f"✅ TradingView ha restituito {len(df)} azioni")
            return df, removed_fields

        except Exception as e:
            unknown = extract_unknown_field(e)

            # Se non è un errore "unknown field", stoppo i retry (altrimenti loop inutile)
            if not unknown:
                print(f"❌ Errore TradingView API (non gestito): {e}")
                return pd.DataFrame(), removed_fields

            # Se è "unknown field", lo rimuovo e riprovo
            if unknown in active:
                active.remove(unknown)
                removed_fields.append(unknown)
                print(f"⚠️ Campo non supportato da TradingView: {unknown} -> rimosso (retry {attempt}/10)")
                continue

            # Se per qualche motivo non è in active, comunque fermiamoci
            print(f"❌ Campo sconosciuto ma non presente in lista: {unknown}")
            return pd.DataFrame(), removed_fields

    print("❌ Troppi retry: impossibile completare lo screener.")
    return pd.DataFrame(), removed_fields


# ============================================================
# 2) YFINANCE DETAILS
# ============================================================

def get_yfinance_details(ticker: str) -> dict:
    """Arricchisce ticker con dati tecnici/target via yfinance (best effort)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info or {}

        if hist.empty or len(hist) < 50:
            return {}

        price = float(hist["Close"].iloc[-1])

        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
        ma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else ma50

        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        high_20d = float(hist["High"].tail(20).max())
        low_20d = float(hist["Low"].tail(20).min())

        support_1 = max(low_20d, ma200)
        support_2 = low_52w
        resistance_1 = high_20d
        resistance_2 = high_52w

        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss))
        rsi_trend = "↗️" if rsi_series.iloc[-1] > rsi_series.iloc[-5] else "↘️"

        returns = hist["Close"].pct_change().dropna()
        vol_annual = float(returns.std() * np.sqrt(252) * 100)

        analyst_target = safe_get(info.get("targetMeanPrice"), 0.0)
        if analyst_target > price:
            target = analyst_target
            target_source = "Analisti"
        else:
            target = price * (1 + (1.5 * vol_annual / 100))
            target_source = "Volatilità"

        target = min(target, price * 1.40)
        upside = ((target / price) - 1) * 100

        timeframe = "6-9 mesi" if upside > 20 else ("9-12 mesi" if upside > 10 else "12+ mesi")

        company_name = info.get("longName") or info.get("shortName") or ticker

        avg_vol = float(hist["Volume"].tail(20).mean())
        volume_trend = "↗️" if float(hist["Volume"].iloc[-1]) > avg_vol else "↘️"

        forward_pe = safe_get(info.get("forwardPE"), 0.0)
        peg = safe_get(info.get("pegRatio"), 0.0)
        div_yield = safe_get(info.get("dividendYield"), 0.0) * 100
        profit_margin = safe_get(info.get("profitMargins"), 0.0) * 100

        return {
            "company_name": company_name,
            "target": float(target),
            "target_source": target_source,
            "upside": float(upside),
            "timeframe": timeframe,
            "vol_annual": float(vol_annual),
            "ma50": float(ma50),
            "ma200": float(ma200),
            "high_52w": float(high_52w),
            "low_52w": float(low_52w),
            "dist_52w": float(((price / high_52w) - 1) * 100),
            "support_1": float(support_1),
            "support_2": float(support_2),
            "resistance_1": float(resistance_1),
            "resistance_2": float(resistance_2),
            "rsi_trend": rsi_trend,
            "volume_trend": volume_trend,
            "forward_pe": float(forward_pe),
            "peg": float(peg),
            "div_yield": float(div_yield),
            "profit_margin": float(profit_margin),
        }

    except Exception:
        return {}


# ============================================================
# 3) SCORING + TOP5
# ============================================================

def calculate_score(row: dict) -> int:
    """Score semplice (best effort). Se manca qualche campo, safe_get lo gestisce."""
    score = 0

    perf6m = safe_get(row.get("Perf.6M"), 0)
    perf3m = safe_get(row.get("Perf.3M"), 0)
    rsi = safe_get(row.get("RSI"), 50)

    if perf6m > 20: score += 20
    elif perf6m > 10: score += 15
    elif perf6m > 5: score += 10

    if perf3m > 10: score += 10
    elif perf3m > 5: score += 7
    elif perf3m > 3: score += 4

    if 55 <= rsi <= 68: score += 10
    elif 45 <= rsi <= 75: score += 6

    roe = safe_get(row.get("return_on_equity"), 0)
    roic = safe_get(row.get("return_on_invested_capital"), 0)
    op_margin = safe_get(row.get("operating_margin"), 0)
    eps_growth = safe_get(row.get("earnings_per_share_diluted_yoy_growth_ttm"), 0)

    if roe > 30: score += 10
    elif roe > 20: score += 7
    elif roe > 15: score += 4

    if roic > 25: score += 10
    elif roic > 15: score += 7
    elif roic > 12: score += 4

    if op_margin > 25: score += 10
    elif op_margin > 15: score += 7
    elif op_margin > 10: score += 4

    if eps_growth > 30: score += 5
    elif eps_growth > 15: score += 3

    pe = safe_get(row.get("price_earnings_ttm"), 50)
    debt_eq = safe_get(row.get("debt_to_equity"), 2)

    if 10 <= pe <= 20: score += 8
    elif 20 < pe <= 30: score += 5
    elif 30 < pe <= 45: score += 2

    if debt_eq < 0.3: score += 7
    elif debt_eq < 0.5: score += 5
    elif debt_eq < 1: score += 2

    volume = safe_get(row.get("volume"), 0)
    if volume > 10_000_000: score += 10
    elif volume > 5_000_000: score += 7
    elif volume > 2_000_000: score += 4

    return int(score)


def build_top5(df_tv: pd.DataFrame) -> list[dict]:
    """Top5 con max 2 titoli per settore + arricchimento yfinance."""
    if df_tv.empty:
        return []

    df = df_tv.copy()
    df["score"] = df.apply(calculate_score, axis=1)
    df = df.sort_values("score", ascending=False)

    results = []
    seen_sectors = {}

    print("\n📊 Analisi dettagliata top candidati...")

    for _, row in df.iterrows():
        if len(results) >= 5:
            break

        ticker = row.get("name", "")
        sector = row.get("sector", "Altro")

        if seen_sectors.get(sector, 0) >= 2:
            continue

        print(f"  ⏳ {ticker}...", end=" ", flush=True)

        details = get_yfinance_details(ticker)
        time.sleep(0.3)

        price = safe_get(row.get("close"), 0)
        score = int(row.get("score", 0))

        if score >= 70: sentiment = "🟢 Forte Acquisto"
        elif score >= 55: sentiment = "🟢 Acquista"
        elif score >= 40: sentiment = "🟡 Mantieni"
        else: sentiment = "🔴 Evita"

        results.append({
            "ticker": ticker,
            "company_name": details.get("company_name", ticker),
            "sector": sector,
            "score": score,
            "sentiment": sentiment,
            "price": price,
            "target": details.get("target", price * 1.10),
            "upside": details.get("upside", 10),
        })

        seen_sectors[sector] = seen_sectors.get(sector, 0) + 1
        print(f"✅ Score: {score}")

    return results


# ============================================================
# 4) EMAIL HTML (semplice + nota campi rimossi)
# ============================================================

def generate_html(stocks: list[dict], removed_fields: list[str]) -> str:
    """HTML minimale: tabella + nota finale campi rimossi."""
    today = datetime.now().strftime("%d/%m/%Y")

    rows = ""
    for i, s in enumerate(stocks, start=1):
        rows += (
            f"<tr>"
            f"<td>#{i}</td>"
            f"<td><b>{s['ticker']}</b></td>"
            f"<td>{s['sector']}</td>"
            f"<td>{s['score']}</td>"
            f"<td>{s['sentiment']}</td>"
            f"<td>${s['price']:.2f}</td>"
            f"<td>${s['target']:.2f}</td>"
            f"<td>+{s['upside']:.1f}%</td>"
            f"</tr>"
        )

    note = ""
    if removed_fields:
        note = (
            "<hr>"
            "<p style='font-size:12px;color:#6b7280;'>"
            "<b>Nota:</b> alcuni campi TradingView non erano disponibili e sono stati ignorati: "
            f"{', '.join(sorted(set(removed_fields)))}"
            "</p>"
        )

    return f"""
    <html>
      <body style="font-family:Arial, sans-serif;">
        <h2>Top 5 Azioni USA - {today}</h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
          <tr>
            <th>Rank</th><th>Ticker</th><th>Settore</th><th>Score</th><th>Giudizio</th>
            <th>Prezzo</th><th>Target</th><th>Upside</th>
          </tr>
          {rows}
        </table>
        {note}
      </body>
    </html>
    """


# ============================================================
# 5) MAIN
# ============================================================

def main():
    start = datetime.now()

    print("=" * 65)
    print("🇺🇸 STOCK SCREENER USA - TradingView API")
    print("=" * 65)
    print(f"⏰ Avvio: {start.strftime('%d/%m/%Y %H:%M')}")

    df_tv, removed_fields = run_tradingview_screener()

    if df_tv.empty:
        print("\n❌ Nessuna azione trovata o errore TradingView.")
        # comunque provo a mandare una mail “vuota” con nota errore
        subject = f"🇺🇸 Screener USA - Nessun risultato ({datetime.now().strftime('%d/%m/%Y')})"
        body = generate_html([], removed_fields) + "<p>Nessun risultato dallo screener.</p>"
        send_email(subject, body, is_html=True)
        return

    top5 = build_top5(df_tv)

    print(f"\n{'='*65}")
    print("🏆 TOP 5 AZIONI USA")
    print(f"{'='*65}")
    for i, s in enumerate(top5):
        print(f"#{i} {s['ticker']:6} | Score: {s['score']:3d} | ${s['price']:8.2f} → ${s['target']:8.2f} (+{s['upside']:5.1f}%) | {s['sector']}")

    today = datetime.now().strftime("%d/%m/%Y")
    subject = f"🇺🇸 Top 5 Azioni USA - {today}"
    html = generate_html(top5, removed_fields)

    print("\n📧 Invio email...", end=" ")
    ok = send_email(subject, html, is_html=True)
    print("OK" if ok else "KO")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"⏱️  Completato in {elapsed:.1f} secondi")
    print("=" * 65)


if __name__ == "__main__":
    main()
