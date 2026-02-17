# 📊 Stock Screener USA - TradingView API

Screener automatico azioni USA con gli stessi filtri del tuo screener TradingView `screener_3-6_mesi_USA`.

## 🔍 Filtri Applicati

| Filtro | Valore |
|--------|--------|
| Market Cap | > 10B USD |
| EPS Growth TTM YoY | > 10% |
| Perf 6M | > 5% |
| Perf 3M | > 3% |
| Revenue Growth YoY | > 10% |
| ROE TTM | > 15% |
| ROIC TTM | > 12% |
| SMA 200 | Prezzo sopra ✅ |
| SMA 50 | Prezzo sopra ✅ |
| RSI (14) | 45 – 75 |
| P/E | 10 – 45 |
| Debt/Equity | < 1 |
| Operating Margin | > 10% |
| Volume | > 2M |

## 🚀 Setup Rapido

### 1. Installa dipendenze
```bash
pip install tradingview-screener yfinance pandas numpy
```

### 2. Imposta password Gmail
```bash
export GMAIL_PASSWORD="tuapasswordapp16caratteri"
```

> ℹ️ Serve una **App Password Gmail** (non la password normale)
> Crea su: https://myaccount.google.com/apppasswords

### 3. Esegui
```bash
python screener_usa_tradingview.py
```

## 📧 Output Email

Ricevi ogni giorno un report con:
- **Tabella comparativa** Top 5 azioni
- **Analisi dettagliata** per ogni azione:
  - Score qualità/momentum
  - Price Target (Analisti o Volatilità)
  - Livelli tecnici (Supporti/Resistenze)
  - Raccomandazione (Compra/Mantieni/Evita)
  - Link diretto grafico TradingView
  - Analisi rischi

## ⚙️ GitHub Actions (automazione)

Per ricevere il report **automaticamente ogni giorno** alle 18:00 CET:

1. Vai in `Settings → Secrets → Actions`
2. Crea secret: `GMAIL_PASSWORD` = tua password app
3. Il workflow `.github/workflows/screener.yml` si attiva automaticamente

## 📁 File

```
├── screener_usa_tradingview.py    # Script principale
├── .github/
│   └── workflows/
│       └── screener.yml           # Automazione GitHub Actions
└── README.md
```

## ⚠️ Disclaimer

Analisi automatica — NON è consulenza finanziaria.
Verifica sempre prima di investire. Diversifica, usa stop loss.
