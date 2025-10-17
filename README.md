# Crypto Bull or Bear — v6 Lite (Fast)

**Optimized for speed**
- Cached per-coin XGBoost model (trained once, then reused).
- Dark theme, candlestick chart first, instant prediction.
- Coin-specific news loads asynchronously in the background.
- Fast auto-refresh option (10s).

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app_fast.py
```

## Notes
- The first run per coin/currency/lookback will train a lightweight model and cache it to `model_cache/…`.
- News uses RSS (CoinDesk, CoinTelegraph, The Block) filtered by coin **name or ticker**; you can extend to CryptoNews/NewsAPI easily.
- Educational only. Not financial advice.
