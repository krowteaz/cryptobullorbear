# Crypto Bull or Bear — v8 (Watchlist + Logs + Test Alerts)

**What’s new**
- 🗂️ **Multi-coin watchlist** — choose **Tabs** or **Stacked panels** layout.
- 🔔 **One‑click Test Alerts** (Discord, Telegram, Email).
- 🧾 **Persistent logs**: CSV + SQLite (alerts & signals). Built-in log viewer.
- ✅ Keeps v7: EMA cross markers, Bollinger Bands, coin-specific news sentiment, XGBoost predictor, risk meter logic.

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

**Tip:** Use CoinGecko IDs in the watchlist (e.g., `bitcoin,ethereum,solana`). Alerts send on **signal flips** per coin.
Logs saved under `/mnt/data/logs`.