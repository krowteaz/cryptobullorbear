# Crypto Bull or Bear — v6 (Coin-Specific News)

**New in v6**
- Coin-specific news using **CryptoNews API** (`tickers=SYMBOL`) or **NewsAPI.org** (`"Name" OR TICKER`), with RSS fallback filtered by coin.
- Sentiment influences AI **entry/target/stop** suggestions.
- Candlesticks (green/red) + high/low trend lines.
- XGBoost predictor, backtest, metrics.

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
Enter optional API keys in the sidebar (CryptoNews or NewsAPI) for better coin-specific headlines.
