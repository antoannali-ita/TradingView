"""
╔══════════════════════════════════════════════════════════════╗
║         STOCK SCREENER USA - TradingView API                 ║
║         Filtri identici al tuo screener TradingView          ║
║         Invio automatico email con analisi dettagliata        ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
1. pip install tradingview-screener yfinance pandas numpy
2. Imposta variabile ambiente: GMAIL_PASSWORD=tuapasswordapp
3. python screener_usa_tradingview.py

PER GITHUB ACTIONS (automazione giornaliera):
- Aggiungi GMAIL_PASSWORD nei GitHub Secrets
- Usa il file .github/workflows/screener.yml incluso
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAZIONE
# ============================================================

EMAIL_CONFIG = {
    'sender': 'antoannali@gmail.com',
    'password': os.getenv('GMAIL_PASSWORD', ''),
    'recipient': 'antoannali@gmail.com',
}

# ============================================================
# INSTALLAZIONE AUTOMATICA DIPENDENZE
# ============================================================

def install_deps():
    import subprocess
    deps = ['tradingview-screener', 'yfinance', 'pandas', 'numpy']
    for dep in deps:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', dep, '-q'],
            capture_output=True
        )

try:
    from tradingview_screener import Query, Column
    import yfinance as yf
except ImportError:
    print("📦 Installazione dipendenze...")
    install_deps()
    from tradingview_screener import Query, Column
    import yfinance as yf

# ============================================================
# SCREENER TRADINGVIEW (filtri identici al tuo screenshot)
# ============================================================

def run_tradingview_screener():
    """
    Esegue lo screener con gli STESSI filtri del tuo TradingView:
    - Market cap > 10B USD
    - EPS dil growth TTM YoY > 10%
    - Perf 6M > 5%
    - Revenue growth TTM YoY > 10%
    - ROE TTM > 15%
    - SMA 200 > 0 (prezzo sopra MA200)
    - SMA 50 > 0 (prezzo sopra MA50)
    - RSI 14: 45 to 75
    - Volume > 2M
    - P/E: 10 to 45
    - Perf 3M > 3%
    - Debt/Equity FY < 1
    - Operating margin FY > 10%
    - ROIC TTM > 12%
    """
    print("\n🔍 Interrogo TradingView Screener...")
    print("   (Filtri: USA, Large Cap, Momentum + Quality)")
    
    try:
        _, df = (
            Query()
            .set_markets('america')
            .select(
                'name',
                'close',
                'market_cap_basic',
                'price_earnings_ttm',
                'earnings_per_share_diluted_yoy',
                'revenue_growth_rate_5y',
                'dividends_yield_current',
                'return_on_equity',
                'Perf.6M',
                'Perf.3M',
                'RSI',
                'volume',
                'sector',
                'SMA200',
                'SMA50',
                'debt_to_equity',
                'operating_margin',
                'return_on_invested_capital',
                'price_target_mean_1y',
                'analyst_rating_agr',
                'beta_1_year',
                'revenue_annual_yoy',
                'gross_profit_margin_annual_yoy',
                'EPS.Diluted.Yoy',
                'earnings_per_share_diluted_ttm',
                'dividends_per_share_annual',
                'net_income_annual_yoy',
            )
            .where(
                Column('market_cap_basic') > 10_000_000_000,           # Market cap > 10B
                Column('earnings_per_share_diluted_yoy') > 10,         # EPS growth > 10%
                Column('Perf.6M') > 5,                                  # Perf 6M > 5%
                Column('revenue_annual_yoy') > 10,                     # Revenue growth > 10%
                Column('return_on_equity') > 15,                       # ROE > 15%
                Column('RSI') >= 45,                                    # RSI >= 45
                Column('RSI') <= 75,                                    # RSI <= 75
                Column('volume') > 2_000_000,                           # Volume > 2M
                Column('price_earnings_ttm') >= 10,                    # P/E >= 10
                Column('price_earnings_ttm') <= 45,                    # P/E <= 45
                Column('Perf.3M') > 3,                                  # Perf 3M > 3%
                Column('debt_to_equity') < 1,                          # Debt/Equity < 1
                Column('operating_margin') > 10,                       # Op margin > 10%
                Column('return_on_invested_capital') > 12,             # ROIC > 12%
                Column('close') > Column('SMA200'),                    # Prezzo > MA200
                Column('close') > Column('SMA50'),                     # Prezzo > MA50
            )
            .order_by('Perf.6M', ascending=False)
            .limit(50)
            .get_scanner_data()
        )
        
        print(f"✅ TradingView ha restituito {len(df)} azioni")
        return df
        
    except Exception as e:
        print(f"❌ Errore TradingView API: {e}")
        return pd.DataFrame()

# ============================================================
# ANALISI DETTAGLIATA CON YFINANCE
# ============================================================

def safe_get(val, default=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val) if val != '' else default
    except:
        return default

def get_yfinance_details(ticker):
    """Dati extra da yfinance: target, supporti, RSI trend"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        info = stock.info or {}
        
        if hist.empty or len(hist) < 50:
            return {}
        
        price = hist['Close'].iloc[-1]
        
        # Medie Mobili
        ma50 = hist['Close'].rolling(50).mean().iloc[-1]
        ma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else ma50
        
        # 52w
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        high_20d = hist['High'].tail(20).max()
        low_20d = hist['Low'].tail(20).min()
        
        # Supporti/Resistenze
        support_1 = max(low_20d, ma200)
        support_2 = low_52w
        resistance_1 = high_20d
        resistance_2 = high_52w
        
        # RSI trend
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain/loss))
        rsi_trend = "↗️" if rsi_series.iloc[-1] > rsi_series.iloc[-5] else "↘️"
        
        # Volatilità & Target
        returns = hist['Close'].pct_change().dropna()
        vol_annual = returns.std() * np.sqrt(252) * 100
        
        analyst_target = safe_get(info.get('targetMeanPrice'), 0)
        if analyst_target > price:
            target = analyst_target
            target_source = "Analisti"
        else:
            target = price * (1 + (1.5 * vol_annual / 100))
            target_source = "Volatilità"
        target = min(target, price * 1.40)
        upside = ((target / price) - 1) * 100
        
        timeframe = "6-9 mesi" if upside > 20 else ("9-12 mesi" if upside > 10 else "12+ mesi")
        
        # Company name
        company_name = info.get('longName') or info.get('shortName') or ticker
        
        # Volume trend
        avg_vol = hist['Volume'].tail(20).mean()
        volume_trend = "↗️" if hist['Volume'].iloc[-1] > avg_vol else "↘️"
        
        # Forward PE e PEG
        forward_pe = safe_get(info.get('forwardPE'), 0)
        peg = safe_get(info.get('pegRatio'), 0)
        div_yield = safe_get(info.get('dividendYield'), 0) * 100
        profit_margin = safe_get(info.get('profitMargins'), 0) * 100
        
        return {
            'company_name': company_name,
            'target': target,
            'target_source': target_source,
            'upside': upside,
            'timeframe': timeframe,
            'vol_annual': vol_annual,
            'ma50': ma50,
            'ma200': ma200,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'dist_52w': ((price / high_52w) - 1) * 100,
            'support_1': support_1,
            'support_2': support_2,
            'resistance_1': resistance_1,
            'resistance_2': resistance_2,
            'rsi_trend': rsi_trend,
            'volume_trend': volume_trend,
            'forward_pe': forward_pe,
            'peg': peg,
            'div_yield': div_yield,
            'profit_margin': profit_margin,
        }
    except:
        return {}

# ============================================================
# SCORING & SELEZIONE TOP 5
# ============================================================

def calculate_score(row):
    score = 0
    
    # Momentum (40pt)
    perf6m = safe_get(row.get('Perf.6M'), 0)
    perf3m = safe_get(row.get('Perf.3M'), 0)
    rsi = safe_get(row.get('RSI'), 50)
    
    if perf6m > 20: score += 20
    elif perf6m > 10: score += 15
    elif perf6m > 5: score += 10
    
    if perf3m > 10: score += 10
    elif perf3m > 5: score += 7
    elif perf3m > 3: score += 4
    
    if 55 <= rsi <= 68: score += 10
    elif 45 <= rsi <= 75: score += 6
    
    # Quality (35pt)
    roe = safe_get(row.get('return_on_equity'), 0)
    roic = safe_get(row.get('return_on_invested_capital'), 0)
    op_margin = safe_get(row.get('operating_margin'), 0)
    eps_growth = safe_get(row.get('earnings_per_share_diluted_yoy'), 0)
    
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
    
    # Valuation (15pt)
    pe = safe_get(row.get('price_earnings_ttm'), 50)
    debt_eq = safe_get(row.get('debt_to_equity'), 2)
    
    if 10 <= pe <= 20: score += 8
    elif 20 < pe <= 30: score += 5
    elif 30 < pe <= 45: score += 2
    
    if debt_eq < 0.3: score += 7
    elif debt_eq < 0.5: score += 5
    elif debt_eq < 1: score += 2
    
    # Volume (10pt)
    volume = safe_get(row.get('volume'), 0)
    if volume > 10_000_000: score += 10
    elif volume > 5_000_000: score += 7
    elif volume > 2_000_000: score += 4
    
    return score

def build_top5(df_tv):
    """Combina dati TradingView + yfinance e seleziona top 5"""
    
    if df_tv.empty:
        print("❌ Nessun dato da TradingView")
        return []
    
    results = []
    seen_sectors = {}
    
    # Calcola score per tutti
    df_tv['score'] = df_tv.apply(calculate_score, axis=1)
    df_tv = df_tv.sort_values('score', ascending=False)
    
    print(f"\n📊 Analisi dettagliata top candidati...")
    
    for _, row in df_tv.iterrows():
        if len(results) >= 5:
            break
        
        ticker = row.get('name', '')
        sector = row.get('sector', 'Altro')
        
        # Max 2 per settore
        if seen_sectors.get(sector, 0) >= 2:
            continue
        
        print(f"  ⏳ {ticker}...", end=' ', flush=True)
        
        # Dettagli extra da yfinance
        details = get_yfinance_details(ticker)
        time.sleep(0.3)  # Rate limit
        
        price = safe_get(row.get('close'), 0)
        score = int(row.get('score', 0))
        
        if score >= 70:
            sentiment = "🟢 Forte Acquisto"
        elif score >= 55:
            sentiment = "🟢 Acquista"
        elif score >= 40:
            sentiment = "🟡 Mantieni"
        else:
            sentiment = "🔴 Evita"
        
        stock_data = {
            'ticker': ticker,
            'company_name': details.get('company_name', ticker),
            'sector': sector,
            'score': score,
            'sentiment': sentiment,
            'price': price,
            'mcap': safe_get(row.get('market_cap_basic'), 0) / 1e9,
            'pe': safe_get(row.get('price_earnings_ttm'), 0),
            'roe': safe_get(row.get('return_on_equity'), 0),
            'roic': safe_get(row.get('return_on_invested_capital'), 0),
            'op_margin': safe_get(row.get('operating_margin'), 0),
            'debt_eq': safe_get(row.get('debt_to_equity'), 0),
            'rev_growth': safe_get(row.get('revenue_annual_yoy'), 0),
            'eps_growth': safe_get(row.get('earnings_per_share_diluted_yoy'), 0),
            'perf_6m': safe_get(row.get('Perf.6M'), 0),
            'perf_3m': safe_get(row.get('Perf.3M'), 0),
            'rsi': safe_get(row.get('RSI'), 0),
            'volume': safe_get(row.get('volume'), 0),
            'div_yield': details.get('div_yield', safe_get(row.get('dividends_yield_current'), 0)),
            # Da yfinance
            'target': details.get('target', price * 1.10),
            'target_source': details.get('target_source', 'Stima'),
            'upside': details.get('upside', 10),
            'timeframe': details.get('timeframe', '12+ mesi'),
            'vol_annual': details.get('vol_annual', 30),
            'ma50': details.get('ma50', price),
            'ma200': details.get('ma200', price),
            'high_52w': details.get('high_52w', price),
            'low_52w': details.get('low_52w', price),
            'dist_52w': details.get('dist_52w', 0),
            'support_1': details.get('support_1', price * 0.95),
            'support_2': details.get('support_2', price * 0.85),
            'resistance_1': details.get('resistance_1', price * 1.05),
            'resistance_2': details.get('resistance_2', price * 1.15),
            'rsi_trend': details.get('rsi_trend', ''),
            'volume_trend': details.get('volume_trend', ''),
            'forward_pe': details.get('forward_pe', 0),
            'peg': details.get('peg', 0),
            'profit_margin': details.get('profit_margin', 0),
        }
        
        results.append(stock_data)
        seen_sectors[sector] = seen_sectors.get(sector, 0) + 1
        
        print(f"✅ Score: {score}")
    
    return results

# ============================================================
# EMAIL HTML
# ============================================================

def generate_html(stocks):
    today = datetime.now().strftime('%d/%m/%Y')
    
    html = f"""
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
  h1 {{ color: #1a1a1a; border-bottom: 4px solid #1a73e8; padding-bottom: 15px; margin-bottom: 10px; }}
  .subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 30px; }}
  h2 {{ color: #374151; margin-top: 40px; margin-bottom: 20px; }}
  
  .filters-box {{ background: #f0f4ff; border: 1px solid #c7d7fb; border-radius: 8px; padding: 15px 20px; margin-bottom: 30px; font-size: 13px; color: #374151; }}
  .filter-tag {{ display: inline-block; background: #1a73e8; color: white; padding: 3px 10px; border-radius: 12px; margin: 3px; font-size: 12px; font-weight: 600; }}
  
  .comparison-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  .comparison-table th {{ background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); color: white; padding: 12px 10px; text-align: left; font-size: 13px; }}
  .comparison-table td {{ padding: 11px 10px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
  .comparison-table tr:hover {{ background: #f3f4f6; }}
  
  .ticker-name {{ font-weight: bold; color: #1f2937; font-size: 15px; }}
  .sector-tag {{ background: #e0e7ff; color: #3730a3; padding: 3px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .score-a {{ color: #059669; font-weight: bold; font-size: 18px; }}
  .score-b {{ color: #10b981; font-weight: bold; font-size: 18px; }}
  .score-c {{ color: #f59e0b; font-weight: bold; font-size: 18px; }}
  .up {{ color: #059669; font-weight: bold; }}
  .down {{ color: #dc2626; font-weight: bold; }}
  
  .stock-card {{ border: 2px solid #e5e7eb; border-radius: 12px; padding: 28px; margin: 28px 0; background: linear-gradient(135deg, #fafafa 0%, #fff 100%); }}
  .stock-header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; margin-bottom: 22px; }}
  .ticker-large {{ font-size: 26px; font-weight: bold; color: #1f2937; }}
  .company-sub {{ font-size: 15px; color: #6b7280; margin-top: 5px; }}
  .badge {{ font-size: 13px; padding: 5px 12px; border-radius: 6px; font-weight: 600; margin-top: 6px; display: inline-block; }}
  .badge-a {{ background: #d1fae5; color: #065f46; }}
  .badge-b {{ background: #dcfce7; color: #166534; }}
  .badge-c {{ background: #fef3c7; color: #92400e; }}
  .chart-btn {{ display: inline-block; margin-top: 10px; padding: 7px 15px; background: #1a73e8; color: white; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: 600; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 13px; margin: 18px 0; }}
  .metric-box {{ padding: 12px; background: white; border-radius: 8px; border-left: 4px solid #1a73e8; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .metric-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; font-weight: 600; margin-bottom: 5px; }}
  .metric-value {{ font-size: 17px; font-weight: bold; color: #1f2937; }}
  .metric-sub {{ font-size: 12px; color: #9ca3af; margin-top: 3px; }}
  
  .perf-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }}
  .perf-box {{ padding: 10px; background: #f0fdf4; border-radius: 6px; text-align: center; border: 1px solid #bbf7d0; }}
  .perf-label {{ font-size: 11px; color: #6b7280; font-weight: 600; }}
  .perf-value {{ font-size: 16px; font-weight: bold; color: #059669; margin-top: 4px; }}
  
  .target-box {{ background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 18px; border-radius: 10px; margin: 15px 0; border-left: 5px solid #28a745; }}
  .tech-box {{ background: #e7f3ff; padding: 18px; border-radius: 10px; margin: 15px 0; border-left: 5px solid #1a73e8; }}
  .rec-box {{ background: #f0fdf4; border: 2px solid #86efac; padding: 14px; border-radius: 8px; margin: 14px 0; }}
  .risk-box {{ background: #fef2f2; border: 2px solid #fca5a5; padding: 14px; border-radius: 8px; margin: 14px 0; }}
  .warn-box {{ background: #fffbeb; border: 2px solid #fcd34d; padding: 14px; border-radius: 8px; margin: 14px 0; }}
  
  .dtable {{ width: 100%; margin: 10px 0; }}
  .dtable td {{ padding: 7px 4px; border-bottom: 1px dotted #ddd; }}
  .dtable td:first-child {{ width: 40%; color: #6b7280; font-weight: 600; }}
  .dtable td:last-child {{ font-weight: bold; color: #1f2937; }}
  
  .hl {{ color: #059669; font-weight: bold; font-size: 20px; }}
  
  .disclaimer {{ margin-top: 50px; padding: 22px; background: #fef3c7; border-radius: 10px; border-left: 5px solid #f59e0b; font-size: 13px; line-height: 1.6; }}
  .footer {{ text-align: center; margin-top: 35px; padding-top: 18px; border-top: 2px solid #e5e7eb; color: #6b7280; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🇺🇸 Top 5 Azioni USA - {today}</h1>
  <div class="subtitle">Analisi automatica via TradingView Screener API + Yahoo Finance</div>
  
  <div class="filters-box">
    <strong>🔍 Filtri applicati (identici al tuo screener TradingView):</strong><br><br>
    <span class="filter-tag">Market Cap &gt; 10B</span>
    <span class="filter-tag">EPS Growth &gt; 10%</span>
    <span class="filter-tag">Perf 6M &gt; 5%</span>
    <span class="filter-tag">Perf 3M &gt; 3%</span>
    <span class="filter-tag">Revenue Growth &gt; 10%</span>
    <span class="filter-tag">ROE &gt; 15%</span>
    <span class="filter-tag">ROIC &gt; 12%</span>
    <span class="filter-tag">SMA 200 ✅</span>
    <span class="filter-tag">SMA 50 ✅</span>
    <span class="filter-tag">RSI 45-75</span>
    <span class="filter-tag">P/E 10-45</span>
    <span class="filter-tag">Debt/Eq &lt; 1</span>
    <span class="filter-tag">Op. Margin &gt; 10%</span>
    <span class="filter-tag">Volume &gt; 2M</span>
  </div>
  
  <h2>📈 Tabella Comparativa</h2>
  <table class="comparison-table">
    <tr>
      <th>Rank</th><th>Ticker</th><th>Settore</th><th>Score</th><th>Giudizio</th>
      <th>Prezzo</th><th>Target</th><th>Upside</th><th>P/E</th><th>ROE</th><th>6M Perf</th>
    </tr>
"""
    
    for i, s in enumerate(stocks):
        rank = i + 1
        sc = s['score']
        sc_class = 'score-a' if sc >= 70 else ('score-b' if sc >= 55 else 'score-c')
        pe_val = f"{s['pe']:.1f}" if s['pe'] > 0 else 'N/D'
        
        html += f"""
    <tr>
      <td><strong>#{rank}</strong></td>
      <td class="ticker-name">{s['ticker']}</td>
      <td><span class="sector-tag">{s['sector']}</span></td>
      <td class="{sc_class}">{sc}</td>
      <td>{s['sentiment']}</td>
      <td>${s['price']:.2f}</td>
      <td>${s['target']:.2f}</td>
      <td class="up">+{s['upside']:.1f}%</td>
      <td>{pe_val}</td>
      <td>{s['roe']:.1f}%</td>
      <td class="up">+{s['perf_6m']:.1f}%</td>
    </tr>"""
    
    html += "</table><h2>📋 Analisi Dettagliata</h2>"
    
    for i, s in enumerate(stocks):
        rank = i + 1
        sc = s['score']
        sc_class = 'score-a' if sc >= 70 else ('score-b' if sc >= 55 else 'score-c')
        badge_class = 'badge-a' if sc >= 70 else ('badge-b' if sc >= 55 else 'badge-c')
        
        tv_url = f"https://www.tradingview.com/chart/?symbol=NASDAQ%3A{s['ticker']}"
        
        pe_val = f"{s['pe']:.1f}" if s['pe'] > 0 else 'N/D'
        fpe_val = f"{s['forward_pe']:.1f}" if s['forward_pe'] > 0 else 'N/D'
        peg_val = f"{s['peg']:.2f}" if s['peg'] > 0 else 'N/D'
        div_val = f"{s['div_yield']:.2f}%" if s['div_yield'] > 0 else 'N/D'
        debt_status = '✅ Basso' if s['debt_eq'] < 0.3 else ('🟡 Medio' if s['debt_eq'] < 0.7 else '🔴 Alto')
        
        # Raccomandazione
        if sc >= 65 and s['upside'] > 8:
            action = f"✅ COMPRA vicino supporto ${s['support_1']:.2f}"
            rec_class = "rec-box"
        elif sc >= 45:
            action = "⚠️ MANTIENI - Attendi conferma"
            rec_class = "warn-box"
        else:
            action = "❌ EVITA - Score insufficiente"
            rec_class = "risk-box"
        
        # Rischi
        risks = []
        if s['pe'] > 35: risks.append("Valutazione elevata")
        if s['rsi'] > 70: risks.append("RSI overbought")
        if s['debt_eq'] > 0.7: risks.append("Debito moderato")
        if s['vol_annual'] > 50: risks.append("Alta volatilità")
        if s['dist_52w'] < -20: risks.append("Lontano dai massimi")
        risk_text = " · ".join(risks) if risks else "Nessun rischio significativo rilevato"
        
        html += f"""
  <div class="stock-card">
    <div class="stock-header">
      <div>
        <div class="ticker-large">{rank}. {s['ticker']}</div>
        <div class="company-sub">{s['company_name']}</div>
        <div style="margin-top:8px"><span class="sector-tag">{s['sector']}</span></div>
        <a href="{tv_url}" target="_blank" class="chart-btn">📈 Grafico TradingView</a>
      </div>
      <div style="text-align:right">
        <div class="{sc_class}" style="font-size:32px">{sc}</div>
        <div class="badge {badge_class}">{s['sentiment']}</div>
      </div>
    </div>
    
    <div class="perf-grid">
      <div class="perf-box">
        <div class="perf-label">Perf 3 Mesi</div>
        <div class="perf-value">+{s['perf_3m']:.1f}%</div>
      </div>
      <div class="perf-box">
        <div class="perf-label">Perf 6 Mesi</div>
        <div class="perf-value">+{s['perf_6m']:.1f}%</div>
      </div>
      <div class="perf-box">
        <div class="perf-label">RSI (14)</div>
        <div class="perf-value">{s['rsi']:.0f} {s['rsi_trend']}</div>
      </div>
      <div class="perf-box">
        <div class="perf-label">Da 52w High</div>
        <div class="perf-value" style="color:{'#059669' if s['dist_52w'] > -5 else '#f59e0b'}">{s['dist_52w']:.1f}%</div>
      </div>
    </div>
    
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-label">💰 Prezzo</div>
        <div class="metric-value">${s['price']:.2f}</div>
        <div class="metric-sub">Cap: ${s['mcap']:.1f}B</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">📊 ROE / ROIC</div>
        <div class="metric-value">{s['roe']:.1f}% / {s['roic']:.1f}%</div>
        <div class="metric-sub">Redditività</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">📈 Revenue Growth</div>
        <div class="metric-value">+{s['rev_growth']:.1f}%</div>
        <div class="metric-sub">EPS Growth: +{s['eps_growth']:.1f}%</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">💳 Debt/Equity</div>
        <div class="metric-value">{s['debt_eq']:.2f}</div>
        <div class="metric-sub">{debt_status}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">📉 P/E</div>
        <div class="metric-value">{pe_val}</div>
        <div class="metric-sub">Forward: {fpe_val}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">💼 Op. Margin</div>
        <div class="metric-value">{s['op_margin']:.1f}%</div>
        <div class="metric-sub">Net: {s['profit_margin']:.1f}%</div>
      </div>
    </div>
    
    <div class="target-box">
      <strong>🎯 PRICE TARGET ({s['target_source']})</strong>
      <table class="dtable">
        <tr><td>🎯 Target:</td><td class="hl">${s['target']:.2f}</td></tr>
        <tr><td>📈 Upside:</td><td class="hl">+{s['upside']:.1f}%</td></tr>
        <tr><td>⏰ Timeframe:</td><td>{s['timeframe']}</td></tr>
        <tr><td>📊 Volatilità annua:</td><td>{s['vol_annual']:.1f}%</td></tr>
      </table>
    </div>
    
    <div class="tech-box">
      <strong>📊 LIVELLI TECNICI</strong>
      <table class="dtable">
        <tr><td>🔴 R2 (52w High):</td><td>${s['resistance_2']:.2f}</td></tr>
        <tr><td>🟠 R1 (20d High):</td><td>${s['resistance_1']:.2f}</td></tr>
        <tr style="background:#fffbeb"><td>━━ Prezzo corrente:</td><td><strong>${s['price']:.2f}</strong></td></tr>
        <tr><td>🟢 S1 (200MA):</td><td>${s['support_1']:.2f}</td></tr>
        <tr><td>🟢 S2 (52w Low):</td><td>${s['support_2']:.2f}</td></tr>
      </table>
      <div style="margin-top:10px; padding:10px; background:white; border-radius:6px; font-size:13px;">
        <strong>📍 Indicatori:</strong> RSI {s['rsi']:.0f} {s['rsi_trend']} · 
        Volume {s['volume_trend']} · MA50: ${s['ma50']:.2f} · MA200: ${s['ma200']:.2f}
      </div>
    </div>
    
    <div class="{rec_class}">
      <strong>💡 RACCOMANDAZIONE:</strong> {action}<br>
      <strong>📍 Entry ideale:</strong> ${s['support_1']:.2f} – ${(s['support_1']+s['price'])/2:.2f}<br>
      <strong>🛑 Stop Loss:</strong> ${s['support_2']:.2f}<br>
      <strong>🎯 Take Profit:</strong> 50% a ${s['resistance_1']:.2f} · 50% a ${s['target']:.2f}
    </div>
    
    <div class="risk-box">
      <strong>⚠️ RISCHI DA MONITORARE:</strong> {risk_text}
    </div>
    
    <div style="padding:12px; background:#f9fafb; border-radius:6px; font-size:13px; margin-top:12px;">
      <strong>📌 Extra:</strong> PEG {peg_val} · 
      Dividendo {div_val} · 
      Profit Margin {s['profit_margin']:.1f}%
    </div>
  </div>"""
    
    html += f"""
  <div class="disclaimer">
    <strong>⚠️ DISCLAIMER</strong><br><br>
    Analisi automatica via TradingView API + Yahoo Finance — <strong>NON è consulenza finanziaria</strong>.
    Verifica sempre bilanci recenti, earnings e contesto macro prima di investire.
    Diversifica, usa stop loss, investi solo capitale che puoi permetterti di perdere.<br>
    <em>I rendimenti passati non garantiscono risultati futuri.</em>
  </div>
  <div class="footer">
    🤖 Stock Screener USA Pro · {today} · Fonte: TradingView API + Yahoo Finance<br>
    Filtri: TradingView screener_3-6_mesi_USA
  </div>
</div>
</body>
</html>"""
    
    return html

# ============================================================
# INVIO EMAIL
# ============================================================

def send_email(subject, html_body):
    if not EMAIL_CONFIG['password']:
        print("\n❌ GMAIL_PASSWORD non trovata!")
        print("   Imposta variabile: export GMAIL_PASSWORD='tua_password_app'")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['recipient']
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            smtp.send_message(msg)
        
        print("✅ Email inviata con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore email: {e}")
        return False

# ============================================================
# MAIN
# ============================================================

def main():
    start = datetime.now()
    
    print("=" * 65)
    print("🇺🇸 STOCK SCREENER USA - TradingView API")
    print("=" * 65)
    print(f"⏰ Avvio: {start.strftime('%d/%m/%Y %H:%M')}")
    
    # Valida email password
    if not EMAIL_CONFIG['password']:
        print("\n⚠️  ATTENZIONE: GMAIL_PASSWORD non impostata")
        print("   export GMAIL_PASSWORD='tuapasswordapp16caratteri'")
    
    # Step 1: Screener TradingView
    df_tv = run_tradingview_screener()
    
    if df_tv.empty:
        print("\n❌ Nessuna azione trovata con i filtri impostati.")
        print("   Possibile causa: mercato chiuso o filtri troppo restrittivi.")
        return
    
    # Step 2: Arricchisci con yfinance e calcola score
    top5 = build_top5(df_tv)
    
    if not top5:
        print("\n❌ Impossibile costruire la top 5.")
        return
    
    # Step 3: Stampa risultati console
    print(f"\n{'='*65}")
    print("🏆 TOP 5 AZIONI USA")
    print(f"{'='*65}")
    for i, s in enumerate(top5):
        print(f"#{i+1} {s['ticker']:6} | Score: {s['score']:3d} | "
              f"${s['price']:8.2f} → ${s['target']:8.2f} (+{s['upside']:5.1f}%) | "
              f"{s['sector']}")
    
    # Step 4: Genera email
    today = datetime.now().strftime('%d/%m/%Y')
    subject = f"🇺🇸 Top 5 Azioni USA - {today}"
    html = generate_html(top5)
    
    print(f"\n📧 Invio email a {EMAIL_CONFIG['recipient']}...", end=' ')
    send_email(subject, html)
    
    elapsed = (datetime.now() - start).total_seconds()
    print(f"⏱️  Completato in {elapsed:.1f} secondi")
    print("=" * 65)

if __name__ == "__main__":
    main()
