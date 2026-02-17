"""
╔══════════════════════════════════════════════════════════════╗
║         STOCK SCREENER USA - TradingView API                 ║
║         Filtri identici al tuo screener TradingView          ║
║         Invio automatico email con analisi dettagliata       ║
╚══════════════════════════════════════════════════════════════╝

REQUISITI:
- requirements.txt con:
  tradingview-screener
  yfinance
  pandas
  numpy

CONFIG EMAIL:
- NON mettere password nel codice.
- Usa mailer.py che legge:
  GMAIL_SENDER
  GMAIL_RECIPIENT
  GMAIL_PASSWORD
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# TradingView screener
from tradingview_screener import Query, Column
from common_utility.mailer import send_email

# Yahoo Finance
import yfinance as yf

# Modulo locale per invio email (password in env/secrets)
from mailer import send_email


# ============================================================
# UTILS
# ============================================================

def safe_get(val, default=0.0) -> float:
    """
    Converte in float in modo sicuro.
    - gestisce None
    - gestisce NaN
    - gestisce stringhe vuote
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val) if val != "" else default
    except Exception:
        return default


# ============================================================
# 1) TRADINGVIEW SCREENER
# ============================================================

def run_tradingview_screener() -> pd.DataFrame:
    """
    Esegue lo screener su TradingView via tradingview-screener.

    IMPORTANTISSIMO:
    I campi devono esistere nello scanner TradingView,
    altrimenti ottieni 400 Bad Request.

    Campi corretti usati qui:
    - earnings_per_share_diluted_yoy_growth_ttm
    - total_revenue_yoy_growth_ttm
    """

    print("\n🔍 Interrogo TradingView Screener...")
    print("   (USA, Large Cap, Momentum + Quality)")

    try:
        _, df = (
            Query()
            .set_markets("america")
            .select(
                # Identificativi e prezzo
                "name",
                "close",

                # Fondamentali
                "market_cap_basic",
                "price_earnings_ttm",
                "earnings_per_share_diluted_yoy_growth_ttm",  # FIX
                "total_revenue_yoy_growth_ttm",               # FIX
                "dividends_yield_current",
                "return_on_equity",
                "return_on_invested_capital",
                "operating_margin",
                "debt_to_equity",

                # Momentum / trend
                "Perf.6M",
                "Perf.3M",
                "RSI",
                "volume",
                "sector",
                "SMA200",
                "SMA50",
            )
            .where(
                # Size
                Column("market_cap_basic") > 10_000_000_000,

                # Growth
                Column("earnings_per_share_diluted_yoy_growth_ttm") > 10,
                Column("total_revenue_yoy_growth_ttm") > 10,

                # Momentum
                Column("Perf.6M") > 5,
                Column("Perf.3M") > 3,

                # Quality
                Column("return_on_equity") > 15,
                Column("return_on_invested_capital") > 12,
                Column("operating_margin") > 10,

                # Risk / leverage
                Column("debt_to_equity") < 1,

                # Trend
                Column("close") > Column("SMA200"),
                Column("close") > Column("SMA50"),

                # RSI
                Column("RSI") >= 45,
                Column("RSI") <= 75,

                # Liquidity
                Column("volume") > 2_000_000,

                # Valuation
                Column("price_earnings_ttm") >= 10,
                Column("price_earnings_ttm") <= 45,
            )
            .order_by("Perf.6M", ascending=False)
            .limit(50)
            .get_scanner_data()
        )

        print(f"✅ TradingView ha restituito {len(df)} azioni")
        return df

    except Exception as e:
        print(f"❌ Errore TradingView API: {e}")
        return pd.DataFrame()


# ============================================================
# 2) DETTAGLI YFINANCE
# ============================================================

def get_yfinance_details(ticker: str) -> dict:
    """
    Arricchisce un ticker con dati tecnici e fondamentali via yfinance:
    - supporti / resistenze
    - target (analisti o stima su volatilità)
    - trend RSI e volume
    - MA50 / MA200
    """

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info or {}

        # Se non ci sono abbastanza dati, skip
        if hist.empty or len(hist) < 50:
            return {}

        price = float(hist["Close"].iloc[-1])

        # Medie mobili
        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
        ma200 = float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else ma50

        # 52w + 20d
        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        high_20d = float(hist["High"].tail(20).max())
        low_20d = float(hist["Low"].tail(20).min())

        # Supporti / resistenze
        support_1 = max(low_20d, ma200)
        support_2 = low_52w
        resistance_1 = high_20d
        resistance_2 = high_52w

        # RSI trend
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss))
        rsi_trend = "↗️" if rsi_series.iloc[-1] > rsi_series.iloc[-5] else "↘️"

        # Volatilità annua
        returns = hist["Close"].pct_change().dropna()
        vol_annual = float(returns.std() * np.sqrt(252) * 100)

        # Target:
        # - se esiste targetMeanPrice e sta sopra il prezzo → uso quello
        # - altrimenti uso una stima su volatilità (max +40%)
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

        # Nome azienda
        company_name = info.get("longName") or info.get("shortName") or ticker

        # Volume trend
        avg_vol = float(hist["Volume"].tail(20).mean())
        volume_trend = "↗️" if float(hist["Volume"].iloc[-1]) > avg_vol else "↘️"

        # Extra
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
# 3) SCORING
# ============================================================

def calculate_score(row: dict) -> int:
    """
    Score 0-100 circa.
    - Momentum 40
    - Quality 35
    - Valuation 15
    - Volume 10
    """

    score = 0

    # -----------------------------
    # MOMENTUM (40)
    # -----------------------------
    perf6m = safe_get(row.get("Perf.6M"), 0)
    perf3m = safe_get(row.get("Perf.3M"), 0)
    rsi = safe_get(row.get("RSI"), 50)

    if perf6m > 20:
        score += 20
    elif perf6m > 10:
        score += 15
    elif perf6m > 5:
        score += 10

    if perf3m > 10:
        score += 10
    elif perf3m > 5:
        score += 7
    elif perf3m > 3:
        score += 4

    if 55 <= rsi <= 68:
        score += 10
    elif 45 <= rsi <= 75:
        score += 6

    # -----------------------------
    # QUALITY (35)
    # -----------------------------
    roe = safe_get(row.get("return_on_equity"), 0)
    roic = safe_get(row.get("return_on_invested_capital"), 0)
    op_margin = safe_get(row.get("operating_margin"), 0)

    # FIX: campo TradingView corretto
    eps_growth = safe_get(row.get("earnings_per_share_diluted_yoy_growth_ttm"), 0)

    if roe > 30:
        score += 10
    elif roe > 20:
        score += 7
    elif roe > 15:
        score += 4

    if roic > 25:
        score += 10
    elif roic > 15:
        score += 7
    elif roic > 12:
        score += 4

    if op_margin > 25:
        score += 10
    elif op_margin > 15:
        score += 7
    elif op_margin > 10:
        score += 4

    if eps_growth > 30:
        score += 5
    elif eps_growth > 15:
        score += 3

    # -----------------------------
    # VALUATION (15)
    # -----------------------------
    pe = safe_get(row.get("price_earnings_ttm"), 50)
    debt_eq = safe_get(row.get("debt_to_equity"), 2)

    if 10 <= pe <= 20:
        score += 8
    elif 20 < pe <= 30:
        score += 5
    elif 30 < pe <= 45:
        score += 2

    if debt_eq < 0.3:
        score += 7
    elif debt_eq < 0.5:
        score += 5
    elif debt_eq < 1:
        score += 2

    # -----------------------------
    # VOLUME (10)
    # -----------------------------
    volume = safe_get(row.get("volume"), 0)

    if volume > 10_000_000:
        score += 10
    elif volume > 5_000_000:
        score += 7
    elif volume > 2_000_000:
        score += 4

    return int(score)


# ============================================================
# 4) BUILD TOP 5
# ============================================================

def build_top5(df_tv: pd.DataFrame) -> list[dict]:
    """
    - calcola score su tutto il df TradingView
    - ordina per score
    - seleziona top 5
    - max 2 titoli per settore
    - arricchisce con yfinance
    """

    if df_tv.empty:
        print("❌ Nessun dato da TradingView")
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

        # max 2 per settore
        if seen_sectors.get(sector, 0) >= 2:
            continue

        print(f"  ⏳ {ticker}...", end=" ", flush=True)

        # yfinance details
        details = get_yfinance_details(ticker)
        time.sleep(0.3)  # rate limit

        price = safe_get(row.get("close"), 0)
        score = int(row.get("score", 0))

        # sentiment in base allo score
        if score >= 70:
            sentiment = "🟢 Forte Acquisto"
        elif score >= 55:
            sentiment = "🟢 Acquista"
        elif score >= 40:
            sentiment = "🟡 Mantieni"
        else:
            sentiment = "🔴 Evita"

        stock_data = {
            "ticker": ticker,
            "company_name": details.get("company_name", ticker),
            "sector": sector,
            "score": score,
            "sentiment": sentiment,

            # TradingView base
            "price": price,
            "mcap": safe_get(row.get("market_cap_basic"), 0) / 1e9,
            "pe": safe_get(row.get("price_earnings_ttm"), 0),
            "roe": safe_get(row.get("return_on_equity"), 0),
            "roic": safe_get(row.get("return_on_invested_capital"), 0),
            "op_margin": safe_get(row.get("operating_margin"), 0),
            "debt_eq": safe_get(row.get("debt_to_equity"), 0),

            # FIX: revenue / eps corretti
            "rev_growth": safe_get(row.get("total_revenue_yoy_growth_ttm"), 0),
            "eps_growth": safe_get(row.get("earnings_per_share_diluted_yoy_growth_ttm"), 0),

            "perf_6m": safe_get(row.get("Perf.6M"), 0),
            "perf_3m": safe_get(row.get("Perf.3M"), 0),
            "rsi": safe_get(row.get("RSI"), 0),
            "volume": safe_get(row.get("volume"), 0),

            # Dividendo: preferisco yfinance se presente
            "div_yield": details.get(
                "div_yield",
                safe_get(row.get("dividends_yield_current"), 0)
            ),

            # yfinance technical
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
        }

        results.append(stock_data)
        seen_sectors[sector] = seen_sectors.get(sector, 0) + 1

        print(f"✅ Score: {score}")

    return results


# ============================================================
# 5) HTML EMAIL (la tua funzione)
# ============================================================

def generate_html(stocks):
    """
    La tua funzione HTML è già ok.
    Non cambia nulla qui, se non che ora la mail viene inviata via mailer.py
    """

    # === INCOLLA QUI LA TUA generate_html COMPLETA ===
    # (è identica a quella che hai già)
    # -------------------------------------------------

    today = datetime.now().strftime("%d/%m/%Y")

    # Placeholder: sostituisci con la tua versione completa
    return f"""
    <html>
    <body>
      <h1>Top 5 USA - {today}</h1>
      <p>Hai dimenticato di incollare la generate_html completa.</p>
    </body>
    </html>
    """


# ============================================================
# 6) MAIN
# ============================================================

def main():
    start = datetime.now()

    print("=" * 65)
    print("🇺🇸 STOCK SCREENER USA - TradingView API")
    print("=" * 65)
    print(f"⏰ Avvio: {start.strftime('%d/%m/%Y %H:%M')}")

    # 1) TradingView
    df_tv = run_tradingview_screener()

    if df_tv.empty:
        print("\n❌ Nessuna azione trovata con i filtri impostati.")
        return

    # 2) Top 5
    top5 = build_top5(df_tv)

    if not top5:
        print("\n❌ Impossibile costruire la top 5.")
        return

    # 3) Output console
    print(f"\n{'='*65}")
    print("🏆 TOP 5 AZIONI USA")
    print(f"{'='*65}")
    for i, s in enumerate(top5):
        print(
            f"#{i+1} {s['ticker']:6} | Score: {s['score']:3d} | "
            f"${s['price']:8.2f} → ${s['target']:8.2f} (+{s['upside']:5.1f}%) | "
            f"{s['sector']}"
        )

    # 4) Email
    today = datetime.now().strftime("%d/%m/%Y")
    subject = f"🇺🇸 Top 5 Azioni USA - {today}"
    html = generate_html(top5)

    print("\n📧 Invio email...", end=" ")
    ok = send_email(subject, html, is_html=True)
    print("OK" if ok else "KO")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"⏱️  Completato in {elapsed:.1f} secondi")
    print("=" * 65)


if __name__ == "__main__":
    main()
