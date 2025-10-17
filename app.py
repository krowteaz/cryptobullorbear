import streamlit as st
import pandas as pd
import numpy as np
import requests, time, math, re
from datetime import datetime
from typing import Tuple, List, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import plotly.graph_objects as go
import feedparser

# ===================== CoinGecko Helpers =====================
def cg_get(path, params=None, api_key=None, base_url="https://api.coingecko.com/api/v3", max_retries=5):
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    backoff = 1.0
    for attempt in range(max_retries):
        r = requests.get(url, params=params or {}, headers=headers, timeout=30)
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            ra = r.headers.get("retry-after")
            wait = float(ra) if ra else backoff
            time.sleep(wait)
            backoff = min(backoff * 2, 16)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

@st.cache_data(show_spinner=False, ttl=600)
def cg_ping(api_key=None, base_url="https://api.coingecko.com/api/v3"):
    return cg_get("/ping", api_key=api_key, base_url=base_url)

@st.cache_data(show_spinner=False, ttl=600)
def cg_search_coins(query, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get("/search", params={"query": query}, api_key=api_key, base_url=base_url)
    return data.get("coins", [])

@st.cache_data(show_spinner=False, ttl=600)
def fetch_market_chart(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/market_chart",
                  params={"vs_currency": vs_currency, "days": days},
                  api_key=api_key, base_url=base_url)
    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    dfp = pd.DataFrame(prices, columns=["ts", "price"])
    dfv = pd.DataFrame(vols, columns=["ts", "volume"])
    df = pd.merge(dfp, dfv, on="ts", how="left")
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","price","volume"]]

@st.cache_data(show_spinner=False, ttl=600)
def fetch_ohlc(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/ohlc",
                  params={"vs_currency": vs_currency, "days": days},
                  api_key=api_key, base_url=base_url)
    cols = ["ts","open","high","low","close"]
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","open","high","low","close"]]

@st.cache_data(show_spinner=False, ttl=600)
def fetch_coin_symbol(coin_id: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}", params={"localization": "false", "tickers": "false", "market_data": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}, api_key=api_key, base_url=base_url)
    sym = data.get("symbol","").upper()
    name = data.get("name","").strip()
    return sym or None, name or coin_id

# ===================== Indicators =====================
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_features(df: pd.DataFrame):
    df = df.copy()
    df["ret"] = df["price"].pct_change()
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["price"].rolling(w).mean()
        df[f"ema_{w}"] = ema(df["price"], w)
        df[f"mom_{w}"] = df["price"].pct_change(w)
        df[f"vol_{w}"] = df["ret"].rolling(w).std()
        df[f"v_chg_{w}"] = df["volume"].pct_change(w)
    df["rsi_14"] = rsi(df["price"], 14)
    macd_line, signal_line, hist = macd(df["price"], 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    for c in ["sma_5","sma_10","sma_20","sma_50","ema_5","ema_10","ema_20","ema_50"]:
        df[c] = df[c] / df["price"]
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [
        "ret",
        "mom_5","mom_10","mom_20",
        "vol_5","vol_10","vol_20",
        "v_chg_5","v_chg_10","v_chg_20",
        "rsi_14","macd","macd_signal","macd_hist",
        "ema_5","ema_10","ema_20","ema_50",
        "sma_5","sma_10","sma_20","sma_50",
    ]
    return df, feature_cols

# ===================== Modeling (XGBoost) =====================
@st.cache_resource(show_spinner=False)
def train_xgb(X_train, y_train, X_valid, y_valid):
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist"
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model

def time_series_split(df_feat, feature_cols, split_ratio=0.8):
    X = df_feat[feature_cols].values
    y = df_feat["target"].values
    split_idx = int(len(df_feat) * split_ratio)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], split_idx

def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["bearish","bullish"], zero_division=0)
    return acc, cm, report, y_prob, y_pred

# ===================== Coin-Specific News =====================
NEWS_RSS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss",
]

BULL_WORDS = ["surge","rally","record","all-time high","spike","bull","buy","breakout","soars","jumps","uptrend","green"]
BEAR_WORDS = ["drop","dump","selloff","bear","crash","plunge","falls","downtrend","red","decline","slump"]

def contains_coin(text: str, coin_name: str, ticker: str) -> bool:
    t = text.lower()
    if coin_name.lower() in t: 
        return True
    if re.search(rf"\\b{re.escape(ticker.lower())}\\b", t):
        return True
    return False

@st.cache_data(show_spinner=False, ttl=300)
def fetch_coin_news_cryptonews(ticker: str, api_key: str, limit: int=20):
    url = "https://cryptonews-api.com/api/v1"
    params = {"tickers": ticker, "items": limit, "token": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("data", [])

@st.cache_data(show_spinner=False, ttl=300)
def fetch_coin_news_newsapi(query: str, api_key: str, limit: int=20):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": limit, "language": "en", "sortBy": "publishedAt", "apiKey": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])

@st.cache_data(show_spinner=False, ttl=300)
def fetch_coin_news_rss(coin_name: str, ticker: str, max_items=50):
    items = []
    for url in NEWS_RSS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_items//len(NEWS_RSS)+1]:
                title = e.get("title","").strip()
                summary = re.sub("<.*?>","", e.get("summary","")).strip()
                link = e.get("link","")
                text = f"{title} {summary}"
                if contains_coin(text, coin_name, ticker):
                    items.append({"source": feed.feed.get("title",""), "title": title, "summary": summary, "link": link})
        except Exception:
            continue
    return items

def score_sentiment(text: str) -> float:
    t = text.lower()
    score = 0
    for w in BULL_WORDS: 
        if w in t: score += 1
    for w in BEAR_WORDS:
        if w in t: score -= 1
    return score

def summarize_news_coin(items):
    if not items:
        return {"score": 0, "trend": "neutral", "top_pos": [], "top_neg": [], "all": []}
    scored = []
    for it in items:
        s = score_sentiment(it["title"] + " " + it.get("summary",""))
        scored.append((s, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    total = sum(s for s,_ in scored)
    trend = "bullish" if total > 1 else ("bearish" if total < -1 else "neutral")
    top_pos = [it for s,it in scored if s>0][:3]
    top_neg = [it for s,it in scored if s<0][:3]
    return {"score": total, "trend": trend, "top_pos": top_pos, "top_neg": top_neg, "all": [it for _,it in scored]}

# ===================== Entry/Exit Heuristics =====================
def atr_like(series: pd.Series, window: int = 14):
    ret = series.pct_change()
    vol = ret.rolling(window).std().iloc[-1]
    return float(vol)

def swing_levels(df_ohlc: pd.DataFrame, lookback: int = 10):
    recent = df_ohlc.tail(lookback)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    last_close = float(recent["close"].iloc[-1])
    return swing_low, swing_high, last_close

def ai_suggestions(prob_bull: float, rsi_val: float, macd_hist: float, dfo: pd.DataFrame, news_score: float):
    swing_low, swing_high, last_close = swing_levels(dfo, lookback=min(20, len(dfo)))
    vol = atr_like(dfo["close"])
    buf = last_close * max(0.005, min(0.03, vol*1.5))
    news_bias = 0.05 if news_score > 0 else (-0.05 if news_score < 0 else 0.0)
    p_adj = np.clip(prob_bull + news_bias, 0, 1)

    if p_adj >= 0.6 and macd_hist >= 0:
        entry = max(swing_low, last_close - buf)
        target = swing_high
        stop = min(swing_low * 0.99, last_close - 2*buf)
        text = "Bullish setup with supportive news — consider buying pullbacks toward support."
        tag = "buy"
    elif p_adj <= 0.4 and macd_hist <= 0:
        entry = min(swing_high, last_close + buf)
        target = swing_low
        stop = max(swing_high * 1.01, last_close + 2*buf)
        text = "Bearish pressure and weak news — consider selling/hedging near resistance."
        tag = "sell"
    else:
        entry = last_close
        target = swing_high if p_adj >= 0.5 else swing_low
        stop = swing_low if p_adj >= 0.5 else swing_high
        text = "Mixed signals — wait for a cleaner break or reduce position size."
        tag = "wait"
    return {
        "mood": tag,
        "entry": float(entry),
        "target": float(target),
        "stop": float(stop),
        "last": float(last_close),
        "explain": text,
        "context": {
            "swing_low": swing_low, "swing_high": swing_high, "vol_proxy": vol,
            "prob_bull_raw": float(prob_bull), "prob_bull_news_adjusted": float(p_adj),
            "rsi": float(rsi_val), "macd_hist": float(macd_hist), "news_score": float(news_score)
        }
    }

# ===================== UI =====================
st.set_page_config(page_title="Crypto Bull or Bear • v6", page_icon="📈", layout="wide")
st.title("📈 Crypto Bull or Bear — Coin-Specific News + Signals (v6)")
st.caption("Educational demo. Uses CoinGecko + XGBoost + coin-specific news sentiment. Not financial advice.")

with st.sidebar:
    st.header("Data Sources")
    api_key = st.text_input("CoinGecko API Key (optional)", type="password")
    base_url = st.selectbox("CG API Base", ["https://api.coingecko.com/api/v3", "https://pro-api.coingecko.com/api/v3"], index=0)
    st.markdown("---")
    news_key_crn = st.text_input("CryptoNews API Key (optional)", type="password")
    news_key_napi = st.text_input("NewsAPI.org Key (optional)", type="password")
    st.divider()

    st.header("Symbol & Window")
    coin_query = st.text_input("Search coin", value="bitcoin")
    if st.button("Search"):
        try:
            results = cg_search_coins(coin_query, api_key=api_key, base_url=base_url)
            st.session_state["search_results"] = [(c.get("id"), f"{c.get('name')} ({c.get('symbol')})") for c in results[:20]]
        except Exception as e:
            st.error(f"Search error: {e}")
    options = st.session_state.get("search_results", [("bitcoin","Bitcoin (btc)")])
    coin_id = st.selectbox("CoinGecko ID", options=[x[0] for x in options], format_func=lambda x: dict(options).get(x, x))
    vs_currency = st.selectbox("Currency", ["usd","eur","php","jpy"], index=0)
    days_choice = st.selectbox("Lookback", ["7","14","30","90","180","365"], index=2)
    refresh = st.checkbox("Auto-refresh (60s)", value=False)
    run_btn = st.button("Run / Refresh", type="primary")

try:
    _ = cg_ping(api_key=api_key, base_url=base_url)
    st.toast("CoinGecko OK ✅", icon="✅")
except Exception as e:
    st.error(f"Cannot reach CoinGecko API: {e}")

st.info("Tip: Enter a CryptoNews or NewsAPI key for higher-quality, coin-filtered headlines. Otherwise RSS fallback filters by coin name/ticker.")

def build_and_run():
    df = fetch_market_chart(coin_id, vs_currency, days_choice, api_key=api_key, base_url=base_url)
    dfo = fetch_ohlc(coin_id, vs_currency, days_choice, api_key=api_key, base_url=base_url)
    if df.empty or len(df) < 120 or dfo.empty:
        st.warning("Not enough price/OHLC data. Try a different lookback.")
        return

    ticker, coin_name = fetch_coin_symbol(coin_id, api_key=api_key, base_url=base_url)
    ticker = ticker or coin_id.upper()
    coin_name = coin_name or coin_id

    st.subheader(f"Candlesticks for {coin_name} ({ticker}) — green up / red down")
    fig = go.Figure(data=[go.Candlestick(
        x=dfo["time"], open=dfo["open"], high=dfo["high"], low=dfo["low"], close=dfo["close"],
        increasing_line_color="green", decreasing_line_color="red",
        increasing_fillcolor="green", decreasing_fillcolor="red", name="OHLC"
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=10,r=10,t=30,b=10))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["high"].rolling(5).max(), mode="lines", name="High trend"))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["low"].rolling(5).min(), mode="lines", name="Low trend"))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2,1], vertical_alignment="top")

    with left:
        st.subheader("Signals & Prediction")
        df_feat, feature_cols = build_features(df)
        X_train, y_train, X_test, y_test, split_idx = time_series_split(df_feat, feature_cols, split_ratio=0.8)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = train_xgb(X_train_s, y_train, X_test_s, y_test)
        acc, cm, report, y_prob, y_pred = evaluate_model(model, X_test_s, y_test)

        last_row = df_feat.iloc[[-1]][feature_cols].values
        last_scaled = scaler.transform(last_row)
        prob_bull = float(model.predict_proba(last_scaled)[0,1])
        rsi_val = float(df_feat.iloc[-1]["rsi_14"])
        macd_hist = float(df_feat.iloc[-1]["macd_hist"])

        st.metric("Model Accuracy (test)", f"{acc*100:.1f}%")
        st.metric("Bullish Probability", f"{prob_bull*100:.1f}%")
        st.metric("RSI(14)", f"{rsi_val:.1f}")
        st.metric("MACD Hist", f"{macd_hist:.5f}")

        with st.expander("Backtest vs Buy & Hold (Test)"):
            df_sig = df_feat.copy()
            df_sig.loc[split_idx:, "signal"] = np.where(y_pred==1, 1, -1)
            df_sig["strategy_ret"] = df_sig["signal"].shift(1) * df_sig["ret"]
            df_sig["equity"] = (1 + df_sig["strategy_ret"].fillna(0)).cumprod()
            df_sig["bh_equity"] = (1 + df_sig["ret"].fillna(0)).cumprod()
            st.line_chart(df_sig.set_index("time")[["equity","bh_equity"]])
            st.dataframe(pd.DataFrame(cm, index=["Actual Bear","Actual Bull"], columns=["Pred Bear","Pred Bull"]), use_container_width=True)
            st.code(report)

    with right:
        st.subheader(f"News Sentiment — {coin_name} ({ticker})")
        news_items = []
        if news_key_crn:
            try:
                news_items = fetch_coin_news_cryptonews(ticker, news_key_crn, limit=20)
                news_items = [{"source": n.get("source_name",""), "title": n.get("title",""), "summary": n.get("text",""), "link": n.get("news_url","")} for n in news_items]
            except Exception as e:
                st.warning(f"CryptoNews API error: {e}")
        if not news_items and news_key_napi:
            try:
                arts = fetch_coin_news_newsapi(f'"{coin_name}" OR {ticker}', news_key_napi, limit=20)
                news_items = [{"source": a.get("source",{}).get("name",""), "title": a.get("title",""), "summary": a.get("description",""), "link": a.get("url","")} for a in arts if contains_coin((a.get("title","")+" "+a.get("description","")), coin_name, ticker)]
            except Exception as e:
                st.warning(f"NewsAPI error: {e}")
        if not news_items:
            news_items = fetch_coin_news_rss(coin_name, ticker, max_items=50)

        summary = summarize_news_coin(news_items)
        trend = summary["trend"]
        score = summary["score"]
        if trend == "bullish":
            st.success(f"News trend: BULLISH (score {score:+d})")
        elif trend == "bearish":
            st.error(f"News trend: BEARISH (score {score:+d})")
        else:
            st.warning(f"News trend: NEUTRAL (score {score:+d})")
        for it in (summary.get("top_pos", []) + summary.get("top_neg", []))[:6]:
            st.write(f"• [{it['title']}]({it['link']})")

        st.subheader("AI Trade Suggestion")
        ideas = ai_suggestions(prob_bull, rsi_val, macd_hist, dfo, summary["score"])
        if ideas["mood"] == "buy":
            st.success("Suggested Action: BUY / ADD")
        elif ideas["mood"] == "sell":
            st.error("Suggested Action: SELL / HEDGE")
        else:
            st.info("Suggested Action: WAIT / NEUTRAL")

        st.metric("Entry (guide)", f"{ideas['entry']:.4f} {vs_currency.upper()}")
        st.metric("Target", f"{ideas['target']:.4f} {vs_currency.upper()}")
        st.metric("Stop", f"{ideas['stop']:.4f} {vs_currency.upper()}")
        st.caption(ideas["explain"])
        with st.expander("Context"):
            st.json(ideas["context"])

if run_btn:
    build_and_run()

if 'last_auto' not in st.session_state:
    st.session_state['last_auto'] = 0.0
if refresh:
    import time as _t
    now = _t.time()
    if now - st.session_state['last_auto'] > 60:
        st.session_state['last_auto'] = now
        st.rerun()
