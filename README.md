# Crypto Bull or Bear — v5 (XGBoost + News + Candles)

**New in v5**
- 📰 **Crypto news sentiment** from CoinDesk, CoinTelegraph, The Block (RSS, no API key). Simple keyword-based score → bullish/bearish/neutral.
- 🕯️ **Green/Red Candlesticks** using CoinGecko OHLC + high/low trend lines.
- 🤖 **AI trade suggestions** (entry/target/stop) from model probability + RSI/MACD + swing levels.
- ⚡ Same XGBoost predictor and backtests, CoinGecko without Enterprise interval.

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
> If xgboost install fails on Windows, try `pip install --only-binary=:all: xgboost`.

## Notes
- News sentiment is heuristic and for **education only**.
- OHLC endpoint supports fixed day ranges (7, 14, 30, 90, 180, 365).
- Suggestions are **not financial advice**.
