import streamlit as st
import pandas as pd
import numpy as np
import requests, time, re, os, sqlite3
from datetime import datetime
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import plotly.graph_objects as go
import feedparser

# ---------------- Config ----------------
LOG_DIR = "/mnt/data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- CoinGecko Helpers ----------------
def cg_get(path, params=None, api_key=None, base_url="https://api.coingecko.com/api/v3", max_retries=4):
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.get(url, params=params or {}, headers=headers, timeout=20)
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            wait = float(r.headers.get("retry-after") or backoff)
            time.sleep(wait)
            backoff = min(backoff * 2, 8)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

@st.cache_data(show_spinner=False, ttl=600)
def cg_ping(api_key=None, base_url="https://api.coingecko.com/api/v3"):
    return cg_get("/ping", api_key=api_key, base_url=base_url)

@st.cache_data(show_spinner=False, ttl=600)
def fetch_market_chart(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/market_chart",
                  params={"vs_currency": vs_currency, "days": days},
                  api_key=api_key, base_url=base_url)
    prices = data.get("prices", []); vols = data.get("total_volumes", [])
    dfp = pd.DataFrame(prices, columns=["ts", "price"])
    dfv = pd.DataFrame(vols, columns=["ts", "volume"])
    df = pd.merge(dfp, dfv, on="ts", how="left")
    if df.empty: return df
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
    if df.empty: return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","open","high","low","close"]]

@st.cache_data(show_spinner=False, ttl=600)
def fetch_coin_symbol(coin_id: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}", params={"localization": "false", "tickers": "false", "market_data": "false", "community_data": "false", "developer_data": "false", "sparkline": "false"}, api_key=api_key, base_url=base_url)
    sym = data.get("symbol","").upper()
    name = data.get("name","").strip()
    return sym or coin_id.upper(), name or coin_id

# ---------------- Indicators ----------------
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

def bollinger(series: pd.Series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma, ma + num_std*std, ma - num_std*std

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
    df["macd"] = macd_line; df["macd_signal"] = signal_line; df["macd_hist"] = hist
    for c in ["sma_5","sma_10","sma_20","sma_50","ema_5","ema_10","ema_20","ema_50"]:
        df[c] = df[c] / df["price"]
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    feature_cols = ["ret","mom_5","mom_10","mom_20","vol_5","vol_10","vol_20",
                    "v_chg_5","v_chg_10","v_chg_20","rsi_14","macd","macd_signal","macd_hist",
                    "ema_5","ema_10","ema_20","ema_50","sma_5","sma_10","sma_20","sma_50"]
    return df, feature_cols

# ---------------- Model (with caching) ----------------
@st.cache_resource(show_spinner=False)
def train_xgb_cached(key: tuple, X_train, y_train, X_valid, y_valid):
    # key = (coin_id, vs_currency, days, fast_mode_flag) used only for caching
    model = XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist"
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model

def time_series_split(df_feat, feature_cols, split_ratio=0.8):
    X = df_feat[feature_cols].values; y = df_feat["target"].values
    split_idx = int(len(df_feat) * split_ratio)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], split_idx

def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["bearish","bullish"], zero_division=0)
    return acc, cm, report, y_prob, y_pred

# ---------------- News (coin-filtered RSS) ----------------
NEWS_RSS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss",
]
BULL_WORDS = ["surge","rally","record","all-time high","spike","bull","buy","breakout","soars","jumps","uptrend","green"]
BEAR_WORDS = ["drop","dump","selloff","bear","crash","plunge","falls","downtrend","red","decline","slump"]

def contains_coin(text: str, coin_name: str, ticker: str) -> bool:
    t = text.lower()
    if coin_name.lower() in t: return True
    import re as _re
    return bool(_re.search(rf"\\b{ticker.lower()}\\b", t))

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
                if contains_coin(f"{title} {summary}", coin_name, ticker):
                    items.append({"source": feed.feed.get("title",""), "title": title, "summary": summary, "link": link})
        except Exception:
            continue
    return items

def score_sentiment(text: str) -> float:
    t = text.lower(); score = 0
    for w in BULL_WORDS: 
        if w in t: score += 1
    for w in BEAR_WORDS:
        if w in t: score -= 1
    return score

def summarize_news_coin(items):
    if not items: return {"score": 0, "trend": "neutral", "all": []}
    total = sum(score_sentiment(i["title"] + " " + i.get("summary","")) for i in items)
    trend = "bullish" if total > 1 else ("bearish" if total < -1 else "neutral")
    return {"score": total, "trend": trend, "all": items}

# ---------------- Suggestions ----------------
def atr_like(series: pd.Series, window: int = 14):
    return float(series.pct_change().rolling(window).std().iloc[-1])

def swing_levels(df_ohlc: pd.DataFrame, lookback: int = 20):
    recent = df_ohlc.tail(lookback)
    return float(recent["low"].min()), float(recent["high"].max()), float(recent["close"].iloc[-1])

def ai_suggestions(prob_bull: float, rsi_val: float, macd_hist: float, dfo: pd.DataFrame, news_score: float, model_acc: float):
    swing_low, swing_high, last_close = swing_levels(dfo)
    vol = atr_like(dfo["close"])
    buf = last_close * max(0.005, min(0.03, vol*1.5))
    news_bias = 0.05 if news_score > 0 else (-0.05 if news_score < 0 else 0.0)
    p_adj = float(np.clip(prob_bull + news_bias, 0, 1))

    if p_adj >= 0.6 and macd_hist >= 0:
        entry = max(swing_low, last_close - buf); target = swing_high
        stop = min(swing_low * 0.99, last_close - 2*buf); mood = "buy"; msg = "Bullish + supportive news."
    elif p_adj <= 0.4 and macd_hist <= 0:
        entry = min(swing_high, last_close + buf); target = swing_low
        stop = max(swing_high * 1.01, last_close + 2*buf); mood = "sell"; msg = "Bearish + weak news."
    else:
        entry = last_close; target = swing_high if p_adj >= 0.5 else swing_low
        stop = swing_low if p_adj >= 0.5 else swing_high; mood = "wait"; msg = "Mixed — wait/size smaller."

    # position size quick calc
    conf = max(0.0, (p_adj - 0.5) * 2) * (0.5 + 0.5*model_acc)
    vol_penalty = 1.0 / (1.0 + 10.0*vol)
    size_pct = 0.02 + (0.1 - 0.02) * conf * vol_penalty
    size_pct = float(min(0.1, max(0.02, size_pct)))

    return {"mood": mood, "entry": float(entry), "target": float(target), "stop": float(stop),
            "last": float(last_close), "pos_size_pct": size_pct, "explain": msg,
            "context": {"prob_bull_raw": prob_bull, "prob_bull_news_adjusted": p_adj,
                        "rsi": rsi_val, "macd_hist": macd_hist, "news_score": news_score,
                        "swing_low": swing_low, "swing_high": swing_high, "vol_proxy": vol}}

# ---------------- UI: Fast Mode ----------------
st.set_page_config(page_title="Crypto Bull or Bear • v10 (Fast Mode)", page_icon="⚡", layout="wide")
st.title("⚡ Crypto Bull or Bear — v10 (Fast Mode)")
st.caption("Optimized for speed. Uses CoinGecko + XGBoost + coin-specific news. Not financial advice.")

with st.sidebar:
    st.header("Mode & Data")
    fast_mode = st.toggle("Fast Mode", value=True, help="Default ON: fewer UI updates, cached model, async fetch.")
    minimal_chart = st.toggle("Minimal Chart (line only)", value=False)
    api_key = st.text_input("CoinGecko API Key (optional)", type="password")
    base_url = st.selectbox("CG API Base", ["https://api.coingecko.com/api/v3", "https://pro-api.coingecko.com/api/v3"], index=0)

    st.divider()
    st.header("Watchlist")
    coins_text = st.text_input("CoinGecko IDs", value="bitcoin,ethereum,solana")
    vs_currency = st.selectbox("Currency", ["usd","eur","php","jpy"], index=0)
    days_choice = st.selectbox("Lookback", ["7","14","30","90","180","365"], index=2)
    refresh = st.checkbox("Auto-refresh (60s)", value=False)
    run_btn = st.button("Run / Refresh", type="primary")

# Quick connectivity check (cached)
try:
    _ = cg_ping(api_key=api_key, base_url=base_url)
    st.toast("CoinGecko OK ✅", icon="✅")
except Exception as e:
    st.error(f"Cannot reach CoinGecko: {e}")

def fetch_all_async(coin_id, vs_currency, days_choice, api_key, base_url, coin_name=None, ticker=None):
    with ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(fetch_market_chart, coin_id, vs_currency, days_choice, api_key, base_url)
        f2 = ex.submit(fetch_ohlc, coin_id, vs_currency, days_choice, api_key, base_url)
        # Need names for news filter; fetch after
        df = f1.result(); dfo = f2.result()
    if coin_name is None or ticker is None:
        ticker, coin_name = fetch_coin_symbol(coin_id, api_key=api_key, base_url=base_url)
    # Fetch news (separate, cached)
    items = fetch_coin_news_rss(coin_name, ticker, max_items=50)
    return df, dfo, ticker, coin_name, items

def render_coin_fast(coin_id: str):
    start = time.time()
    prog = st.progress(5, text=f"[{coin_id}] Starting…")
    df, dfo, ticker, coin_name, news_items = fetch_all_async(coin_id, vs_currency, days_choice, api_key, base_url)
    if df.empty or len(df)<120 or dfo.empty:
        prog.progress(100, text=f"[{coin_id}] Not enough data"); st.warning(f"[{coin_id}] Not enough price/OHLC data."); return

    prog.progress(25, text=f"[{coin_id}] Building features…")
    df_feat, feature_cols = build_features(df)
    X_train, y_train, X_test, y_test, split_idx = time_series_split(df_feat, feature_cols)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

    cache_key = (coin_id, vs_currency, days_choice, fast_mode)
    model = train_xgb_cached(cache_key, X_train_s, y_train, X_test_s, y_test)
    prog.progress(55, text=f"[{coin_id}] Predicting…")
    acc, cm, report, y_prob, y_pred = evaluate_model(model, X_test_s, y_test)
    last_row = df_feat.iloc[[-1]][feature_cols].values
    last_scaled = scaler.transform(last_row)
    prob_bull = float(model.predict_proba(last_scaled)[0,1])
    rsi_val = float(df_feat.iloc[-1]["rsi_14"]); macd_hist = float(df_feat.iloc[-1]["macd_hist"])

    news_summary = summarize_news_coin(news_items)

    prog.progress(75, text=f"[{coin_id}] Rendering…")
    # Chart
    sym, name = ticker, coin_name
    label = f"{name} ({sym})"
    st.subheader(label)
    if minimal_chart:
        st.line_chart(dfo.set_index("time")["close"])
    else:
        # overlays
        dfo = dfo.copy()
        dfo["ema_5"] = dfo["close"].ewm(span=5, adjust=False).mean()
        dfo["ema_20"] = dfo["close"].ewm(span=20, adjust=False).mean()
        ma, up, dn = bollinger(dfo["close"], 20, 2)
        dfo["bb_mid"], dfo["bb_up"], dfo["bb_dn"] = ma, up, dn

        fig = go.Figure(data=[go.Candlestick(
            x=dfo["time"], open=dfo["open"], high=dfo["high"], low=dfo["low"], close=dfo["close"],
            increasing_line_color="green", decreasing_line_color="red",
            increasing_fillcolor="green", decreasing_fillcolor="red", name="OHLC"
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=10,r=10,t=30,b=10))
        fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["ema_5"], mode="lines", name="EMA 5"))
        fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["ema_20"], mode="lines", name="EMA 20"))
        fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_up"], mode="lines", name="BB Upper", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_mid"], mode="lines", name="BB Mid", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_dn"], mode="lines", name="BB Lower", line=dict(dash="dot")))
        st.plotly_chart(fig, use_container_width=True)

    # Quick metrics + suggestion
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{acc*100:.1f}%")
    cols[1].metric("Bullish Prob", f"{prob_bull*100:.1f}%")
    cols[2].metric("RSI(14)", f"{rsi_val:.1f}")
    cols[3].metric("MACD Hist", f"{macd_hist:.5f}")

    ideas = ai_suggestions(prob_bull, rsi_val, macd_hist, dfo if not minimal_chart else pd.DataFrame({
        "low":[dfo["low"].tail(20).min()], "high":[dfo["high"].tail(20).max()], "close":[dfo["close"].iloc[-1]]
    }), news_summary["score"], acc)

    if ideas["mood"] == "buy":
        st.success(f"Action: BUY — Entry {ideas['entry']:.4f} • Target {ideas['target']:.4f} • Stop {ideas['stop']:.4f}")
    elif ideas["mood"] == "sell":
        st.error(f"Action: SELL — Entry {ideas['entry']:.4f} • Target {ideas['target']:.4f} • Stop {ideas['stop']:.4f}")
    else:
        st.info(f"Action: WAIT — Guide Entry {ideas['entry']:.4f} • Target {ideas['target']:.4f} • Stop {ideas['stop']:.4f}")
    st.caption(ideas["explain"])

    if news_summary["trend"] == "bullish":
        st.success(f"News: BULLISH ({news_summary['score']:+d})")
    elif news_summary["trend"] == "bearish":
        st.error(f"News: BEARISH ({news_summary['score']:+d})")
    else:
        st.warning(f"News: NEUTRAL ({news_summary['score']:+d})")

    dur = time.time() - start
    prog.progress(100, text=f"[{coin_id}] Done in {dur:.1f}s")
    st.caption(f"Completed in {dur:.1f}s (Fast Mode {'ON' if fast_mode else 'OFF'})")

def run_watchlist():
    coins = [c.strip() for c in st.session_state.get("coins_text_cache", coins_text).split(",") if c.strip()]
    if not coins:
        st.warning("Please enter at least one CoinGecko ID (e.g., bitcoin).")
        return
    st.session_state["coins_text_cache"] = coins_text

    if fast_mode:
        # tabs to keep UI clean
        tabs = st.tabs(coins)
        for tab, coin in zip(tabs, coins):
            with tab:
                render_coin_fast(coin)
    else:
        # Detailed mode could call a more verbose renderer (omitted for brevity)
        tabs = st.tabs(coins)
        for tab, coin in zip(tabs, coins):
            with tab:
                render_coin_fast(coin)

if run_btn:
    run_watchlist()

# Auto-refresh
if 'last_auto' not in st.session_state: st.session_state['last_auto'] = 0.0
if refresh:
    import time as _t
    now = _t.time()
    if now - st.session_state['last_auto'] > 60:
        st.session_state['last_auto'] = now
        st.rerun()
