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

def generate_html(stocks: list[dict], skipped_notes: list[str]) -> str:
    from datetime import datetime
    today = datetime.now().strftime("%d/%m/%Y")

    # ── Helpers formato ──────────────────────────────────────────
    def fmt_price(v):
        return f"${v:,.2f}" if v is not None and v != 0.0 else None

    def fmt_pct(v):
        return f"{v:+.1f}%" if v is not None and v != 0.0 else None

    def fmt_num(v, decimals=1):
        return f"{v:.{decimals}f}" if v is not None and v != 0.0 else None

    def pct_color(v):
        if v is None: return "#374151"
        return "#16a34a" if v >= 0 else "#dc2626"

    def tv_link(ticker):
        return f"https://www.tradingview.com/chart/?symbol={ticker}"

    # Genera riga tabella SOLO se il valore è disponibile
    def row(label, value, color=None):
        if value is None:
            return ""  # nasconde la riga
        style = f"color:{color};" if color else ""
        return f"""
        <tr>
          <td style="font-size:12px;color:#6b7280;padding:3px 0;">{label}</td>
          <td style="font-size:12px;font-weight:600;text-align:right;{style}">{value}</td>
        </tr>"""

    def sentiment_badge(s):
        if "Forte Acquisto" in s:
            color, bg = "#16a34a", "#dcfce7"
        elif "Acquista" in s:
            color, bg = "#15803d", "#f0fdf4"
        elif "Mantieni" in s:
            color, bg = "#b45309", "#fef9c3"
        else:
            color, bg = "#dc2626", "#fee2e2"
        return f'<span style="background:{bg};color:{color};padding:4px 12px;border-radius:20px;font-weight:700;font-size:13px;">{s}</span>'

    def score_bar(score):
        pct = min(score, 100)
        if pct >= 70:   bar_color = "#16a34a"
        elif pct >= 55: bar_color = "#84cc16"
        elif pct >= 40: bar_color = "#f59e0b"
        else:           bar_color = "#ef4444"
        return f"""
        <div style="background:#e5e7eb;border-radius:6px;height:8px;width:100%;">
          <div style="background:{bar_color};width:{pct}%;height:8px;border-radius:6px;"></div>
        </div>
        <div style="font-size:11px;color:#6b7280;margin-top:3px;">Score: {score}/100</div>"""

    # ── Tutti i filtri applicati (sempre mostrati in fondo) ──────
    ALL_FILTERS = [
        ("Market Cap",             "> $10B"),
        ("EPS Growth YoY TTM",     "> 10%"),
        ("Revenue Growth YoY TTM", "> 10%"),
        ("Performance 6 Mesi",     "> 5%"),
        ("Performance 3 Mesi",     "> 3%"),
        ("ROE",                    "> 15%"),
        ("ROIC",                   "> 12%"),
        ("Operating Margin",       "> 10%"),
        ("Debt/Equity",            "< 1"),
        ("Prezzo > MA200",         "trend rialzista"),
        ("Prezzo > MA50",          "trend rialzista"),
        ("RSI",                    "tra 45 e 75"),
        ("Volume",                 "> 2.000.000"),
        ("P/E (TTM)",              "tra 10 e 45"),
    ]

    # ── Cards ────────────────────────────────────────────────────
    cards_html = ""
    for i, s in enumerate(stocks):
        rank     = i + 1
        ticker   = s.get("ticker", "")
        company  = s.get("company_name", ticker)
        sector   = s.get("sector", "N/A")
        score    = s.get("score", 0)
        sentiment = s.get("sentiment", "")

        price      = s.get("price") or 0
        target     = s.get("target") or 0
        upside     = s.get("upside") or 0
        target_src = s.get("target_source", "Stima")
        timeframe  = s.get("timeframe", "")
        vol        = s.get("vol_annual") or 0

        pe            = s.get("pe")
        fpe           = s.get("forward_pe")
        peg           = s.get("peg")
        roe           = s.get("roe")
        roic          = s.get("roic")
        op_margin     = s.get("op_margin")
        profit_margin = s.get("profit_margin")
        debt_eq       = s.get("debt_eq")
        rev_growth    = s.get("rev_growth")
        eps_growth    = s.get("eps_growth")
        div_yield     = s.get("div_yield")
        mcap          = s.get("mcap")

        ma50         = s.get("ma50")
        ma200        = s.get("ma200")
        rsi          = s.get("rsi")
        rsi_trend    = s.get("rsi_trend", "")
        perf_6m      = s.get("perf_6m")
        perf_3m      = s.get("perf_3m")
        dist_52w     = s.get("dist_52w")
        high_52w     = s.get("high_52w")
        low_52w      = s.get("low_52w")
        volume_trend = s.get("volume_trend", "")

        sup1 = s.get("support_1")
        sup2 = s.get("support_2")
        res1 = s.get("resistance_1")
        res2 = s.get("resistance_2")

        # Righe fondamentali (mostrate solo se hanno valore)
        fund_rows = (
            row("Market Cap",    f"${mcap:.1f}B" if mcap else None)
          + row("P/E (TTM)",     fmt_num(pe))
          + row("Forward P/E",   fmt_num(fpe))
          + row("PEG Ratio",     fmt_num(peg, 2))
          + row("ROE",           f"{fmt_num(roe)}%" if fmt_num(roe) else None)
          + row("ROIC",          f"{fmt_num(roic)}%" if fmt_num(roic) else None)
          + row("Op. Margin",    f"{fmt_num(op_margin)}%" if fmt_num(op_margin) else None)
          + row("Profit Margin", f"{fmt_num(profit_margin)}%" if fmt_num(profit_margin) else None)
          + row("Debt/Equity",   fmt_num(debt_eq, 2))
          + row("Rev. Growth YoY", fmt_pct(rev_growth), pct_color(rev_growth))
          + row("EPS Growth YoY",  fmt_pct(eps_growth), pct_color(eps_growth))
          + row("Dividend Yield",  f"{fmt_num(div_yield)}%" if fmt_num(div_yield) else None)
        )

        # Righe tecniche
        tech_rows = (
            row("Perf. 6 Mesi",    fmt_pct(perf_6m),  pct_color(perf_6m))
          + row("Perf. 3 Mesi",    fmt_pct(perf_3m),  pct_color(perf_3m))
          + row("RSI (14)",        f"{fmt_num(rsi)} {rsi_trend}" if fmt_num(rsi) else None)
          + row("MA 50",           fmt_price(ma50))
          + row("MA 200",          fmt_price(ma200))
          + row("Max 52 sett.",    fmt_price(high_52w))
          + row("Min 52 sett.",    fmt_price(low_52w))
          + row("Dist. da Max 52w",fmt_pct(dist_52w), pct_color(dist_52w))
          + row("Volume trend",    volume_trend if volume_trend else None)
        )

        # Livelli chiave (solo se disponibili)
        def level_row(emoji, label, color, value):
            v = fmt_price(value)
            if not v: return ""
            return f"""<tr>
              <td style="font-size:11px;color:{color};font-weight:600;">{emoji} {label}</td>
              <td style="font-size:11px;font-weight:700;text-align:right;color:{color};">{v}</td>
            </tr>"""

        livelli = (
            level_row("🔴", "Res. 2", "#dc2626", res2)
          + level_row("🟠", "Res. 1", "#f97316", res1)
          + f"""<tr><td colspan="2" style="text-align:center;padding:4px 0;">
              <span style="background:#dbeafe;color:#1d4ed8;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700;">
                ▶ PREZZO: {fmt_price(price) or "N/A"}
              </span></td></tr>"""
          + level_row("🟢", "Sup. 1", "#16a34a", sup1)
          + level_row("🟢", "Sup. 2", "#15803d", sup2)
        )

        cards_html += f"""
        <div style="background:#ffffff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:28px;overflow:hidden;border:1px solid #e5e7eb;">

          <!-- HEADER -->
          <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);padding:20px 24px;">
            <table width="100%"><tr>
              <td style="vertical-align:top;">
                <div style="color:#93c5fd;font-size:12px;font-weight:600;">#{rank} &nbsp;·&nbsp; {sector}</div>
                <div style="color:#ffffff;font-size:22px;font-weight:800;margin-top:4px;">{ticker}</div>
                <div style="color:#bfdbfe;font-size:13px;">{company}</div>
              </td>
              <td style="vertical-align:top;text-align:right;">
                <div style="color:#ffffff;font-size:26px;font-weight:800;">{fmt_price(price) or "N/A"}</div>
                <div style="margin-top:6px;">{sentiment_badge(sentiment)}</div>
                <div style="margin-top:8px;">
                  <a href="{tv_link(ticker)}" style="background:rgba(255,255,255,0.15);color:#ffffff;padding:5px 14px;border-radius:20px;text-decoration:none;font-size:12px;font-weight:600;">
                    📊 Apri su TradingView →
                  </a>
                </div>
              </td>
            </tr></table>
          </div>

          <!-- SCORE -->
          <div style="padding:12px 24px;background:#f8fafc;border-bottom:1px solid #e5e7eb;">
            {score_bar(score)}
          </div>

          <div style="padding:20px 24px;">

            <!-- TARGET -->
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:14px 18px;margin-bottom:18px;">
              <table width="100%"><tr>
                <td style="vertical-align:top;">
                  <div style="font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;">Target Price</div>
                  <div style="font-size:22px;font-weight:800;color:#15803d;">{fmt_price(target) or "N/A"}</div>
                  <div style="font-size:11px;color:#6b7280;">Fonte: {target_src} · {timeframe}</div>
                </td>
                <td style="vertical-align:top;text-align:right;">
                  <div style="font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;">Upside</div>
                  <div style="font-size:28px;font-weight:800;color:{pct_color(upside)};">{fmt_pct(upside) or "N/A"}</div>
                  <div style="font-size:11px;color:#6b7280;">Volatilità: {f"{vol:.1f}%" if vol else "N/A"}</div>
                </td>
              </tr></table>
            </div>

            <!-- 2 COLONNE -->
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="50%" style="padding-right:10px;vertical-align:top;">
                  <div style="background:#f8fafc;border-radius:8px;padding:14px;">
                    <div style="font-size:12px;font-weight:700;color:#1e3a5f;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #e5e7eb;margin-bottom:8px;">📊 Fondamentali</div>
                    <table width="100%">{fund_rows}</table>
                  </div>
                </td>
                <td width="50%" style="padding-left:10px;vertical-align:top;">
                  <div style="background:#f8fafc;border-radius:8px;padding:14px;">
                    <div style="font-size:12px;font-weight:700;color:#1e3a5f;text-transform:uppercase;letter-spacing:0.5px;padding-bottom:6px;border-bottom:1px solid #e5e7eb;margin-bottom:8px;">📈 Analisi Tecnica</div>
                    <table width="100%">{tech_rows}</table>
                    <div style="margin-top:12px;border-top:1px solid #e5e7eb;padding-top:8px;">
                      <div style="font-size:11px;font-weight:700;color:#1e3a5f;text-transform:uppercase;margin-bottom:6px;">Livelli Chiave</div>
                      <table width="100%">{livelli}</table>
                    </div>
                  </div>
                </td>
              </tr>
            </table>

          </div>
        </div>"""

    # ── Sezione filtri in fondo ───────────────────────────────────
    filtri_ok_rows = "".join(
        f"<tr><td style='font-size:12px;color:#374151;padding:4px 8px;'>✅ {name}</td>"
        f"<td style='font-size:12px;color:#6b7280;padding:4px 8px;'>{cond}</td></tr>"
        for name, cond in ALL_FILTERS
    )

    filtri_saltati_rows = ""
    if skipped_notes:
        filtri_saltati_rows = (
            "<tr><td colspan='2' style='padding:10px 8px 4px;'>"
            "<div style='font-size:12px;font-weight:700;color:#b45309;margin-bottom:4px;'>⚠️ Non applicati (campo non disponibile):</div>"
            + "".join(
                f"<div style='font-size:12px;color:#92400e;margin-bottom:3px;'>• {n}</div>"
                for n in skipped_notes
            )
            + "</td></tr>"
        )

    filtri_section = f"""
    <div style="background:#ffffff;border-radius:12px;border:1px solid #e5e7eb;padding:20px 24px;margin-top:8px;">
      <div style="font-size:14px;font-weight:700;color:#1e3a5f;margin-bottom:12px;">🔍 Filtri Screener Applicati</div>
      <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
        <tr style="background:#f1f5f9;">
          <td style="font-size:11px;font-weight:700;color:#6b7280;padding:6px 8px;text-transform:uppercase;width:60%;">Filtro</td>
          <td style="font-size:11px;font-weight:700;color:#6b7280;padding:6px 8px;text-transform:uppercase;">Condizione</td>
        </tr>
        {filtri_ok_rows}
        {filtri_saltati_rows}
      </table>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Segoe UI',Helvetica,Arial,sans-serif;">
<div style="max-width:680px;margin:0 auto;padding:24px 16px;">

  <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);border-radius:12px;padding:28px 32px;margin-bottom:24px;text-align:center;">
    <div style="font-size:32px;margin-bottom:8px;">🇺🇸</div>
    <div style="color:#ffffff;font-size:24px;font-weight:800;">Top 5 Azioni USA</div>
    <div style="color:#93c5fd;font-size:14px;margin-top:6px;">{today} &nbsp;·&nbsp; Large Cap &nbsp;·&nbsp; Momentum + Quality</div>
  </div>

  {cards_html}

  {filtri_section}

  <div style="text-align:center;padding:20px;color:#9ca3af;font-size:11px;margin-top:16px;border-top:1px solid #e5e7eb;">
    Report generato automaticamente · Solo a scopo informativo · Non costituisce consulenza finanziaria<br>
    <span style="font-style:italic;color:#6b7280;">Script ideato e scritto da Antonio Larocca · Tutti i diritti riservati 😄</span>
  </div>

</div>
</body>
</html>"""

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
