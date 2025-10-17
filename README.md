# Crypto Bull or Bear — v9 (Unified Settings + Smart Search)

**What’s new**
- 🧭 **Unified Settings** in a single sidebar expander: API & Market, Watchlist & Display, Alerts, Performance.
- 🔎 **Smarter coin search** with CoinGecko `/search`, multi-select builder, and saved selections.
- ▶️ **One RUN button** triggers analysis on all selected coins with progress feedback.
- ✅ Keeps v8 features: stacked/tabs layouts, alerts, logs (CSV + SQLite), EMA/Bollinger, coin-specific news, XGBoost predictor, risk meter.

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

**Tips**
- Use the **Search** button to fetch live suggestions, tick the ones you want, then hit **RUN / REFRESH**.
- Alerts fire on signal flips; test them with **Send Test Alert** inside Settings.