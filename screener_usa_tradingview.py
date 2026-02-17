"""
╔══════════════════════════════════════════════════════════════╗
║         STOCK SCREENER USA - TradingView API                 ║
║         Filtri identici al tuo screener TradingView          ║
║         Invio automatico email con analisi dettagliata       ║
╚══════════════════════════════════════════════════════════════╝

REQUISITI (requirements.txt):
- tradingview-screener
- yfinance
- pandas
- numpy

EMAIL:
- usa common_utility/mailer.py
- legge da env/secrets:
  GMAIL_SENDER
  GMAIL_RECIPIENT
  GMAIL_PASSWORD
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

# ✅ usa SOLO questo (niente doppioni)
from common_utility.mailer import send_email


# ============================================================
# UTILS
# ============================================================

def safe_get(val, default=0.0) -> float:
    """Converte in float in modo sicuro (None/NaN/stringhe vuote)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val) if val != "" else default
    except Exception:
        return default


def extract_unknown_field(err: Exception) -> str | None:
    """
    Prova a estrarre il nome del campo dal messaggio TradingView:
    es: Unknown field "earnings_per_share_diluted_yoy"
    """
    msg = str(err)
    m = re.search(r'Unknown field\s+"([^"]+)"', msg)
    return m.group(1) if m else None


# ============================================================
# 1) TRADINGVIEW SCREENER (ROBUSTO)
# ============================================================

def run_tradingview_screener_robust() -> tuple[pd.DataFrame, list[str]]:
    """
    Esegue lo screener TradingView, ma se un campo non esiste:
    - lo rimuove da select
    - rimuove i filtri che dipendono da quel campo
    - riprova
    Ritorna:
    - DataFrame risultati
    - lista filtri/campi saltati (da mettere in fondo alla mail)
    """

    print("\n🔍 Interrogo TradingView Screener...")
    print("   (USA, Large Cap, Momentum + Quality)")

    # --- campi che vorresti leggere (se esistono) ---
    select_fields = [
        "name",
        "close",
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

    # --- filtri: li definiamo come "oggetti" così possiamo toglierli se manca un campo ---
    filters = [
        ("Market Cap > 10B", ["market_cap_basic"], lambda: Column("market_cap_basic") > 10_000_000_000),

        ("EPS YoY TTM > 10%", ["earnings_per_share_diluted_yoy_growth_ttm"],
         lambda: Column("earnings_per_share_diluted_yoy_growth_ttm") > 10),

        ("Revenue YoY TTM > 10%", ["total_revenue_yoy_growth_ttm"],
         lambda: Column("total_revenue_yoy_growth_ttm") > 10),

        ("Perf 6M > 5%", ["Perf.6M"], lambda: Column("Perf.6M") > 5),
        ("Perf 3M > 3%", ["Perf.3M"], lambda: Column("Perf.3M") > 3),

        ("ROE > 15%", ["return_on_equity"], lambda: Column("return_on_equity") > 15),
        ("ROIC > 12%", ["return_on_invested_capital"], lambda: Column("return_on_invested_capital") > 12),
        ("Op Margin > 10%", ["operating_margin"], lambda: Column("operating_margin") > 10),

        ("Debt/Equity < 1", ["debt_to_equity"], lambda: Column("debt_to_equity") < 1),

        ("Prezzo > SMA200", ["close", "SMA200"], lambda: Column("close") > Column("SMA200")),
        ("Prezzo > SMA50", ["close", "SMA50"], lambda: Column("close") > Column("SMA50")),

        ("RSI 45-75", ["RSI"], lambda: (Column("RSI") >= 45) & (Column("RSI") <= 75)),

        ("Volume > 2M", ["volume"], lambda: Column("volume") > 2_000_000),

        ("P/E 10-45", ["price_earnings_ttm"],
         lambda: (Column("price_earnings_ttm") >= 10) & (Column("price_earnings_ttm") <= 45)),
    ]

    skipped_notes: list[str] = []
    max_attempts = 8

    for attempt in range(1, max_attempts + 1):
        try:
            # costruisco query
            q = (
                Query()
                .set_markets("america")
                .select(*select_fields)
            )

            # applico filtri attivi
            active_conditions = [f[2]() for f in filters]
            if active_conditions:
                q = q.where(*active_conditions)

            _, df = (
                q.order_by("Perf.6M", ascending=False)
                 .limit(50)
                 .get_scanner_data()
            )

            print(f"✅ TradingView ha restituito {len(df)} azioni")
            return df, skipped_notes

        except Exception as e:
            unknown = extract_unknown_field(e)
            if not unknown:
                # errore non gestibile con fallback: lo mostro e chiudo
                print(f"❌ Errore TradingView (non gestibile): {e}")
                return pd.DataFrame(), skipped_notes

            # segno cosa ho saltato
            skipped_notes.append(f'Campo non disponibile su TradingView: "{unknown}" (saltato)')

            # rimuovo campo dalla select se presente
            if unknown in select_fields:
                select_fields = [x for x in select_fields if x != unknown]

            # rimuovo filtri che dipendono da quel campo
            new_filters = []
            for name, deps, builder in filters:
                if unknown in deps:
                    skipped_notes.append(f'Filtro non applicato: {name} (manca "{unknown}")')
                else:
                    new_filters.append((name, deps, builder))
            filters = new_filters

            print(f"⚠️ TradingView: campo sconosciuto '{unknown}'. Riprovo senza quel campo/filtro... ({attempt}/{max_attempts})")
            time.sleep(0.3)

    # se arrivo qui, ho finito i tentativi
    print("❌ Troppi fallback, stop.")
    return pd.DataFrame(), skipped_notes


# ============================================================
# 2) DETTAGLI YFINANCE (uguale a prima)
# ============================================================

def get_yfinance_details(ticker: str) -> dict:
    """Arricchisce un ticker con dati tecnici/fondamentali via yfinance."""
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
# 3) SCORING + TOP5 (qui tengo la tua logica base, senza cambiare troppo)
# ============================================================

def calculate_score(row: dict) -> int:
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
            "mcap": safe_get(row.get("market_cap_basic"), 0) / 1e9,
            "pe": safe_get(row.get("price_earnings_ttm"), 0),
            "roe": safe_get(row.get("return_on_equity"), 0),
            "roic": safe_get(row.get("return_on_invested_capital"), 0),
            "op_margin": safe_get(row.get("operating_margin"), 0),
            "debt_eq": safe_get(row.get("debt_to_equity"), 0),

            "rev_growth": safe_get(row.get("total_revenue_yoy_growth_ttm"), 0),
            "eps_growth": safe_get(row.get("earnings_per_share_diluted_yoy_growth_ttm"), 0),

            "perf_6m": safe_get(row.get("Perf.6M"), 0),
            "perf_3m": safe_get(row.get("Perf.3M"), 0),
            "rsi": safe_get(row.get("RSI"), 0),
            "volume": safe_get(row.get("volume"), 0),

            "div_yield": details.get("div_yield", safe_get(row.get("dividends_yield_current"), 0)),

            "target": details.get("target", price * 1.10),
            "target_source": details.get("target_source", "Stima"),
            "upside": details.get("upside", 10),
            "timeframe": details.get("timeframe", "12+ mesi"),
            "vol_annual": details.get("vol_annual", 30),

            "ma50": details.get("ma50", price),
            "ma200": details.get("ma200", price),

            "high_52w": details.get("high_52w", price),
            "low_52w": details.get("low_52w", price),
            "dist_52w": details.get("dist_52w", 0),

            "support_1": details.get("support_1", price * 0.95),
            "support_2": details.get("support_2", price * 0.85),
            "resistance_1": details.get("resistance_1", price * 1.05),
            "resistance_2": details.get("resistance_2", price * 1.15),

            "rsi_trend": details.get("rsi_trend", ""),
            "volume_trend": details.get("volume_trend", ""),

            "forward_pe": details.get("forward_pe", 0),
            "peg": details.get("peg", 0),
            "profit_margin": details.get("profit_margin", 0),
        })

        seen_sectors[sector] = seen_sectors.get(sector, 0) + 1
        print(f"✅ Score: {score}")

    return results


# ============================================================
# 4) HTML EMAIL
# ============================================================

def generate_html(stocks: list[dict], skipped_notes: list[str]) -> str:
    """
    Qui devi incollare la TUA generate_html completa.
    Io aggiungo solo una sezione finale "Filtri non applicati".
    """

    today = datetime.now().strftime("%d/%m/%Y")

    # --- qui metti la tua HTML completa, io ti lascio un placeholder ---
    html = f"""
    <html><body>
      <h1>🇺🇸 Top 5 Azioni USA - {today}</h1>
      <p>Qui incolla la tua generate_html completa.</p>
    """

    # ✅ Sezione finale: filtri saltati
    if skipped_notes:
        html += "<hr><h3>⚠️ Filtri/Campi non applicati</h3><ul>"
        for n in skipped_notes:
            html += f"<li>{n}</li>"
        html += "</ul>"

    html += "</body></html>"
    return html


# ============================================================
# MAIN
# ============================================================

def main():
    start = datetime.now()

    print("=" * 65)
    print("🇺🇸 STOCK SCREENER USA - TradingView API")
    print("=" * 65)
    print(f"⏰ Avvio: {start.strftime('%d/%m/%Y %H:%M')}")

    # 1) TradingView robusto
    df_tv, skipped = run_tradingview_screener_robust()

    # Se TradingView non ritorna nulla, NON mi fermo:
    # mando comunque una mail con nota (come vuoi tu: non deve piantarsi)
    top5 = build_top5(df_tv) if not df_tv.empty else []

    # 2) Email
    today = datetime.now().strftime("%d/%m/%Y")
    subject = f"🇺🇸 Top 5 Azioni USA - {today}"

    html = generate_html(top5, skipped)
    print("\n📧 Invio email...", end=" ")
    ok = send_email(subject, html, is_html=True)
    print("OK" if ok else "KO")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"⏱️  Completato in {elapsed:.1f} secondi")
    print("=" * 65)


if __name__ == "__main__":
    main()
