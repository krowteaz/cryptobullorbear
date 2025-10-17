import streamlit as st
import pandas as pd
import numpy as np
import requests, time, re, threading, queue
from datetime import datetime
from typing import Tuple, List, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

import plotly.graph_objects as go
import feedparser
import os

# ===================== UI THEME (Dark) =====================
st.set_page_config(page_title="Crypto Bull/Bear • v6-Lite", page_icon="📈", layout="wide")
st.markdown("""
<style>
:root { --bg:#0f1117; --card:#11131c; --text:#e6e6e6; --muted:#9aa4ad; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; }
.stMetric { background: var(--card); border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

# ===================== Helpers =====================
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

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_chart(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/market_chart", params={"vs_currency": vs_currency, "days": days}, api_key=api_key, base_url=base_url)
    dfp = pd.DataFrame(data.get("prices", []), columns=["ts","price"])
    dfv = pd.DataFrame(data.get("total_volumes", []), columns=["ts","volume"])
    df = pd.merge(dfp, dfv, on="ts", how="left")
    if df.empty: return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    return df[["time","price","volume"]].sort_values("time").reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlc(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/ohlc", params={"vs_currency": vs_currency, "days": days}, api_key=api_key, base_url=base_url)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close"])
    if df.empty: return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    return df[["time","open","high","low","close"]].sort_values("time").reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coin_symbol(coin_id: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}", params={"localization":"false","tickers":"false","market_data":"false","community_data":"false","developer_data":"false","sparkline":"false"}, api_key=api_key, base_url=base_url)
    return (data.get("symbol","").upper() or coin_id.upper(), data.get("name","").strip() or coin_id)

def ema(x, span): return x.ewm(span=span, adjust=False).mean()
def rsi(series, period=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100/(1+rs))
def macd(series, f=12, s=26, sig=9):
    m = ema(series,f)-ema(series,s); sg = ema(m,sig); return m, sg, m-sg

def build_features(df):
    df = df.copy()
    df["ret"] = df["price"].pct_change()
    for w in [5,10,20,50]:
        df[f"sma_{w}"]=df["price"].rolling(w).mean()
        df[f"ema_{w}"]=ema(df["price"],w)
        df[f"mom_{w}"]=df["price"].pct_change(w)
        df[f"vol_{w}"]=df["ret"].rolling(w).std()
    df["rsi_14"]=rsi(df["price"],14)
    m, s, h = macd(df["price"])
    df["macd"]=m; df["macd_signal"]=s; df["macd_hist"]=h
    for c in ["sma_5","sma_10","sma_20","sma_50","ema_5","ema_10","ema_20","ema_50"]:
        df[c] = df[c] / df["price"]
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    feats = ["ret","mom_5","mom_10","mom_20","vol_5","vol_10","vol_20","rsi_14","macd","macd_signal","macd_hist","ema_5","ema_10","ema_20","ema_50","sma_5","sma_10","sma_20","sma_50"]
    return df, feats

def model_cache_path(coin_id, vs_currency, days):
    safe = f"{coin_id}_{vs_currency}_{days}".replace("/","-")
    os.makedirs("model_cache", exist_ok=True)
    return os.path.join("model_cache", f"{safe}.joblib")

def get_or_train_model(df_feat, feats, coin_id, vs_currency, days):
    path = model_cache_path(coin_id, vs_currency, days)
    if os.path.exists(path):
        try:
            obj = joblib.load(path)
            return obj["model"], obj["scaler"]
        except Exception:
            pass
    # train tiny fast model
    X = df_feat[feats].values; y = df_feat["target"].values
    split = int(len(df_feat)*0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
    model.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=False)
    try:
        joblib.dump({"model":model,"scaler":scaler}, path)
    except Exception:
        pass
    return model, scaler

def atr_like(close, window=14):
    return float(close.pct_change().rolling(window).std().iloc[-1])

def swing_levels(dfo, lookback=10):
    rec = dfo.tail(lookback)
    return float(rec["low"].min()), float(rec["high"].max()), float(rec["close"].iloc[-1])

# ===================== Async coin-specific news =====================
NEWS_RSS = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss","https://www.theblock.co/rss"]
BULL = ["surge","rally","record","all-time high","spike","bull","buy","breakout","soars","jumps","uptrend","green"]
BEAR = ["drop","dump","selloff","bear","crash","plunge","falls","downtrend","red","decline","slump"]

def contains_coin(text, name, ticker):
    t = text.lower()
    if name.lower() in t: return True
    return re.search(rf"\\b{re.escape(ticker.lower())}\\b", t) is not None

def score_text(t):
    t = t.lower(); s=0
    for w in BULL: s += (w in t)
    for w in BEAR: s -= (w in t)
    return int(s)

def fetch_news_filtered(name, ticker, limit=30):
    items=[]
    for url in NEWS_RSS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit//len(NEWS_RSS)+1]:
                title=e.get("title",""); summary=re.sub("<.*?>","",e.get("summary",""))
                if contains_coin(title+" "+summary, name, ticker):
                    items.append({"title":title,"summary":summary,"link":e.get("link",""),"src":feed.feed.get("title","")})
        except Exception: 
            continue
    return items

def news_worker(name, ticker, q):
    try:
        it = fetch_news_filtered(name, ticker, limit=36)
        score = sum(score_text(x["title"]+" "+x["summary"]) for x in it)
        trend = "bullish" if score>1 else ("bearish" if score<-1 else "neutral")
        q.put({"items":it[:8], "score":score, "trend":trend})
    except Exception as e:
        q.put({"items":[], "score":0, "trend":"neutral"})

# ===================== App =====================
st.title("📈 Crypto Bull/Bear — v6 Lite (Fast)")
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("CoinGecko API Key (optional)", type="password")
    base_url = st.selectbox("API Base", ["https://api.coingecko.com/api/v3","https://pro-api.coingecko.com/api/v3"], index=0)
    coin_id = st.text_input("CoinGecko ID", value="bitcoin")
    vs_currency = st.selectbox("Currency", ["usd","eur","php","jpy"], index=0)
    days = st.selectbox("Lookback", ["30","90","180","365"], index=0)
    fast_refresh = st.checkbox("Fast refresh (every 10s)", value=False)
    go = st.button("Run / Refresh", type="primary")

if go:
    try:
        df = fetch_market_chart(coin_id, vs_currency, days, api_key=api_key, base_url=base_url)
        dfo = fetch_ohlc(coin_id, vs_currency, days, api_key=api_key, base_url=base_url)
        if df.empty or len(df)<120 or dfo.empty:
            st.warning("Not enough data. Try a longer lookback.")
        else:
            ticker, name = fetch_coin_symbol(coin_id, api_key=api_key, base_url=base_url)

            # Chart first (fast)
            st.subheader(f"Candlesticks — {name} ({ticker})")
            fig = go.Figure(data=[go.Candlestick(
                x=dfo["time"], open=dfo["open"], high=dfo["high"], low=dfo["low"], close=dfo["close"],
                increasing_line_color="green", decreasing_line_color="red",
                increasing_fillcolor="green", decreasing_fillcolor="red"
            )])
            fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font_color="#e6e6e6")
            st.plotly_chart(fig, use_container_width=True)

            # Build features and get cached model (no retrain if exists)
            df_feat, feats = build_features(df)
            model, scaler = get_or_train_model(df_feat, feats, coin_id, vs_currency, days)

            lastX = scaler.transform(df_feat.iloc[[-1]][feats].values)
            prob_bull = float(model.predict_proba(lastX)[0,1])
            rsi_val = float(df_feat.iloc[-1]["rsi_14"]); macd_hist=float(df_feat.iloc[-1]["macd_hist"])

            # Quick suggestion (fast)
            s_low, s_high, last_close = swing_levels(dfo, lookback=min(20,len(dfo)))
            vol = atr_like(dfo["close"])
            buf = last_close * max(0.005, min(0.03, vol*1.5))
            if prob_bull>=0.6 and macd_hist>=0:
                action="BUY / ADD"; entry=max(s_low, last_close-buf); target=s_high; stop=min(s_low*0.99,last_close-2*buf); mood="bull"
            elif prob_bull<=0.4 and macd_hist<=0:
                action="SELL / HEDGE"; entry=min(s_high, last_close+buf); target=s_low; stop=max(s_high*1.01,last_close+2*buf); mood="bear"
            else:
                action="WAIT / NEUTRAL"; entry=last_close; target=s_high if prob_bull>=0.5 else s_low; stop=s_low if prob_bull>=0.5 else s_high; mood="neutral"

            cols = st.columns(4)
            cols[0].metric("Bullish Prob", f"{prob_bull*100:.1f}%")
            cols[1].metric("RSI(14)", f"{rsi_val:.1f}")
            cols[2].metric("MACD Hist", f"{macd_hist:.5f}")
            cols[3].metric("Action", action)

            st.caption("Guide prices")
            g1,g2,g3 = st.columns(3)
            g1.metric("Entry", f"{entry:.4f} {vs_currency.upper()}")
            g2.metric("Target", f"{target:.4f} {vs_currency.upper()}")
            g3.metric("Stop", f"{stop:.4f} {vs_currency.upper()}")

            # Async coin-specific news (non-blocking)
            st.subheader("News (coin-specific; loads in background)")
            q = queue.Queue()
            t = threading.Thread(target=news_worker, args=(name, ticker, q), daemon=True)
            t.start()

            placeholder = st.empty()
            start = time.time()
            while time.time()-start < 0.5:  # brief wait for fast responses
                if not q.empty():
                    break
                time.sleep(0.05)
            if q.empty():
                placeholder.info("Fetching headlines…")
            else:
                data = q.get(False)
                render = "bullish" if data["trend"]=="bullish" else ("bearish" if data["trend"]=="bearish" else "neutral")
                if render=="bullish": st.success(f"News trend: BULLISH (score {data['score']:+d})")
                elif render=="bearish": st.error(f"News trend: BEARISH (score {data['score']:+d})")
                else: st.warning(f"News trend: NEUTRAL (score {data['score']:+d})")
                for itm in data["items"]:
                    st.write(f"• [{itm['title']}]({itm['link']})")

            # finalize async fetch if still running
            def finalize_news():
                try:
                    data = q.get(timeout=3)
                    placeholder.empty()
                    if data:
                        if data["trend"]=="bullish": st.success(f"News trend: BULLISH (score {data['score']:+d})")
                        elif data["trend"]=="bearish": st.error(f"News trend: BEARISH (score {data['score']:+d})")
                        else: st.warning(f"News trend: NEUTRAL (score {data['score']:+d})")
                        for itm in data["items"]:
                            st.write(f"• [{itm['title']}]({itm['link']})")
                except Exception:
                    pass
            threading.Timer(1.2, finalize_news).start()

    except Exception as e:
        st.exception(e)

# Fast auto-refresh
if 'last_auto' not in st.session_state: st.session_state['last_auto']=0.0
if fast_refresh:
    now = time.time()
    if now - st.session_state['last_auto'] > 10:
        st.session_state['last_auto'] = now
        st.rerun()
