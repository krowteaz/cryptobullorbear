# Crypto Bull or Bear — v9.1 (Streamlit-Safe)

**What’s fixed**
- Logging path is now **safe in the cloud**:
  - Uses `LOG_DIR` env if set
  - Else `~/crypto_logs`
  - Else system temp (e.g., `/tmp`)
- Sidebar shows **status**: 🟢 Logs OK / 🔴 No Write Access, plus the active path.

**Still included**
- Unified Settings + Smart Coin Search
- Alerts (Discord/Telegram/Email) with Test Alert
- CSV + SQLite logs
- EMA cross, Bollinger Bands, coin-specific news sentiment
- XGBoost predictor, entry/target/stop, risk meter

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

**Tip:** On Streamlit Cloud, you can set a custom log dir via environment:  
`LOG_DIR=/app/logs`