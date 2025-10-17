import streamlit as st
import pandas as pd
import numpy as np
import requests, time, math, re, smtplib, sqlite3, os
from email.message import EmailMessage
from datetime import datetime
from typing import Tuple, List, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

import plotly.graph_objects as go
import feedparser

LOG_DIR = "/mnt/data/logs"
CSV_PATH = os.path.join(LOG_DIR, "signals.csv")
DB_PATH = os.path.join(LOG_DIR, "signals.db")

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

def bollinger(series: pd.Series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std*std
    lower = ma - num_std*std
    return ma, upper, lower

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

# ===================== Coin-Specific News (RSS filtered) =====================
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

# ===================== Entry/Exit & Risk Meter =====================
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

def position_size_percent(prob_adj: float, vol_proxy: float, model_acc: float, cap_min=0.02, cap_max=0.1):
    conf = (prob_adj - 0.5) * 2
    conf = max(0.0, conf)
    conf *= (0.5 + 0.5*model_acc)
    vol_penalty = 1.0 / (1.0 + 10.0*vol_proxy)
    size = cap_min + (cap_max - cap_min) * conf * vol_penalty
    return float(min(cap_max, max(cap_min, size)))

def ai_suggestions(prob_bull: float, rsi_val: float, macd_hist: float, dfo: pd.DataFrame, news_score: float, model_acc: float):
    swing_low, swing_high, last_close = swing_levels(dfo, lookback=min(20, len(dfo)))
    vol = atr_like(dfo["close"])
    buf = last_close * max(0.005, min(0.03, vol*1.5))
    news_bias = 0.05 if news_score > 0 else (-0.05 if news_score < 0 else 0.0)
    p_adj = np.clip(prob_bull + news_bias, 0, 1)

    if p_adj >= 0.6 and macd_hist >= 0:
        entry = max(swing_low, last_close - buf)
        target = swing_high
        stop = min(swing_low * 0.99, last_close - 2*buf)
        tag = "buy"; expl = "Bullish setup with supportive news — consider buying pullbacks toward support."
    elif p_adj <= 0.4 and macd_hist <= 0:
        entry = min(swing_high, last_close + buf)
        target = swing_low
        stop = max(swing_high * 1.01, last_close + 2*buf)
        tag = "sell"; expl = "Bearish pressure and weak news — consider selling/hedging near resistance."
    else:
        entry = last_close
        target = swing_high if p_adj >= 0.5 else swing_low
        stop = swing_low if p_adj >= 0.5 else swing_high
        tag = "wait"; expl = "Mixed signals — wait for a cleaner break or reduce position size."

    size_pct = position_size_percent(p_adj, vol, model_acc, cap_min=0.02, cap_max=0.1)
    return {"mood": tag, "entry": float(entry), "target": float(target), "stop": float(stop),
            "last": float(last_close), "pos_size_pct": float(size_pct), "explain": expl,
            "context": {"swing_low": swing_low, "swing_high": swing_high, "vol_proxy": vol,
                        "prob_bull_raw": float(prob_bull), "prob_bull_news_adjusted": float(p_adj),
                        "rsi": float(rsi_val), "macd_hist": float(macd_hist), "news_score": float(news_score),
                        "model_acc": float(model_acc)}}

# ===================== Alerts =====================
def send_discord(webhook_url: str, content: str):
    try:
        r = requests.post(webhook_url, json={"content": content}, timeout=15)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)

def send_telegram(bot_token: str, chat_id: str, message: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=15)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)

def send_email_alert(host, port, username, password, from_addr, to_addr, subject, body, use_tls=True):
    try:
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)
        if use_tls:
            with smtplib.SMTP(host, int(port)) as s:
                s.starttls()
                if username:
                    s.login(username, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, int(port)) as s:
                if username:
                    s.login(username, password)
                s.send_message(msg)
        return 200, "sent"
    except Exception as e:
        return -1, str(e)

def try_alerts(signal_str, ideas, coin_label, price, alert_opts: dict):
    message = f"[{coin_label}] Signal: {signal_str}\nPrice: {price:.4f}\nEntry: {ideas['entry']:.4f}\nTarget: {ideas['target']:.4f}\nStop: {ideas['stop']:.4f}\nPos Size: {ideas['pos_size_pct']*100:.1f}%\nNote: {ideas['explain']}"
    results = {}
    if alert_opts.get("discord_webhook"):
        code, text = send_discord(alert_opts["discord_webhook"], message)
        results["discord"] = code
    if alert_opts.get("tg_token") and alert_opts.get("tg_chat"):
        code, text = send_telegram(alert_opts["tg_token"], alert_opts["tg_chat"], message)
        results["telegram"] = code
    if alert_opts.get("smtp_host") and alert_opts.get("smtp_to") and alert_opts.get("smtp_from"):
        code, text = send_email_alert(
            alert_opts.get("smtp_host"), alert_opts.get("smtp_port", 587),
            alert_opts.get("smtp_user"), alert_opts.get("smtp_pass"),
            alert_opts.get("smtp_from"), alert_opts.get("smtp_to"),
            f"[{coin_label}] Signal: {signal_str}", message, use_tls=True
        )
        results["email"] = code
    return results

# ===================== Logging (CSV + SQLite) =====================
def init_db(path=DB_PATH):
    os.makedirs(LOG_DIR, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
            ts TEXT, coin TEXT, signal TEXT, price REAL, entry REAL, target REAL, stop REAL,
            pos_size REAL, channel TEXT, result_code INTEGER, note TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            ts TEXT, coin TEXT, mood TEXT, prob REAL, rsi REAL, macd REAL, news REAL, model_acc REAL
        )""")
        conn.commit()

def log_alert(coin, signal, price, ideas, results):
    ts = datetime.utcnow().isoformat()
    # CSV
    row = {
        "ts": ts, "coin": coin, "signal": signal, "price": price,
        "entry": ideas["entry"], "target": ideas["target"], "stop": ideas["stop"],
        "pos_size": ideas["pos_size_pct"], "channels": "|".join(results.keys() or []),
        "codes": "|".join(str(v) for v in results.values() or [])
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame([row]).to_csv(CSV_PATH, index=False)
    else:
        pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    # SQLite
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        for ch, code in results.items():
            conn.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (ts, coin, signal, float(price), float(ideas["entry"]), float(ideas["target"]),
                          float(ideas["stop"]), float(ideas["pos_size_pct"]), ch, int(code), ideas["explain"]))
        conn.commit()

def log_signal(coin, mood, prob, rsi, macd, news, model_acc):
    ts = datetime.utcnow().isoformat()
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",
                     (ts, coin, mood, float(prob), float(rsi), float(macd), float(news), float(model_acc)))
        conn.commit()
    # Append to CSV as well (optional)
    row = {"ts": ts, "coin": coin, "mood": mood, "prob": prob, "rsi": rsi, "macd": macd, "news": news, "model_acc": model_acc}
    sig_csv = os.path.join(LOG_DIR, "signals_stream.csv")
    if not os.path.exists(sig_csv):
        pd.DataFrame([row]).to_csv(sig_csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(sig_csv, mode="a", header=False, index=False)

# ===================== UI =====================
st.set_page_config(page_title="Crypto Bull or Bear • v8", page_icon="📈", layout="wide")
st.title("📈 Crypto Bull or Bear — Watchlist + Logs + Test Alerts (v8)")
st.caption("Educational demo. Uses CoinGecko + XGBoost + coin-specific news. Not financial advice.")

with st.sidebar:
    st.header("CoinGecko")
    api_key = st.text_input("API Key (optional)", type="password")
    base_url = st.selectbox("CG API Base", ["https://api.coingecko.com/api/v3", "https://pro-api.coingecko.com/api/v3"], index=0)

    st.divider()
    st.header("Watchlist")
    coins_text = st.text_input("CoinGecko IDs (comma-separated)", value="bitcoin,ethereum,solana")
    layout_choice = st.radio("Layout", ["Tabs","Stacked panels"], index=0)

    st.divider()
    st.header("Alerts")
    discord_webhook = st.text_input("Discord Webhook URL")
    tg_token = st.text_input("Telegram Bot Token")
    tg_chat = st.text_input("Telegram Chat ID")
    smtp_host = st.text_input("SMTP Host")
    smtp_port = st.number_input("SMTP Port", value=587, step=1)
    smtp_user = st.text_input("SMTP Username")
    smtp_pass = st.text_input("SMTP Password", type="password")
    smtp_from = st.text_input("From Email")
    smtp_to = st.text_input("To Email")
    if st.button("Send Test Alert"):
        test_res = try_alerts("TEST", {"entry":0,"target":0,"stop":0,"pos_size_pct":0,"explain":"config test","last":0}, "TEST", 0.0, {
            "discord_webhook": discord_webhook, "tg_token": tg_token, "tg_chat": tg_chat,
            "smtp_host": smtp_host, "smtp_port": smtp_port, "smtp_user": smtp_user, "smtp_pass": smtp_pass,
            "smtp_from": smtp_from, "smtp_to": smtp_to
        })
        st.write("Test results:", test_res)

    st.divider()
    st.header("Run")
    vs_currency = st.selectbox("Currency", ["usd","eur","php","jpy"], index=0)
    days_choice = st.selectbox("Lookback", ["7","14","30","90","180","365"], index=2)
    refresh = st.checkbox("Auto-refresh (60s)", value=False)
    run_btn = st.button("Run / Refresh", type="primary")

try:
    _ = cg_ping(api_key=api_key, base_url=base_url)
    st.toast("CoinGecko OK ✅", icon="✅")
except Exception as e:
    st.error(f"Cannot reach CoinGecko API: {e}")

def render_coin_block(coin_id: str):
    df = fetch_market_chart(coin_id, vs_currency, days_choice, api_key=api_key, base_url=base_url)
    dfo = fetch_ohlc(coin_id, vs_currency, days_choice, api_key=api_key, base_url=base_url)
    if df.empty or len(df) < 120 or dfo.empty:
        st.warning(f"[{coin_id}] Not enough price/OHLC data.")
        return

    ticker, coin_name = fetch_coin_symbol(coin_id, api_key=api_key, base_url=base_url)
    ticker = ticker or coin_id.upper()
    coin_name = coin_name or coin_id
    coin_label = f"{coin_name} ({ticker})"

    # OHLC + overlays
    dfo = dfo.copy()
    dfo["ema_5"] = dfo["close"].ewm(span=5, adjust=False).mean()
    dfo["ema_20"] = dfo["close"].ewm(span=20, adjust=False).mean()
    ma20, bb_up, bb_dn = bollinger(dfo["close"], window=20, num_std=2)
    dfo["bb_mid"] = ma20; dfo["bb_up"] = bb_up; dfo["bb_dn"] = bb_dn

    st.subheader(coin_label)
    fig = go.Figure(data=[go.Candlestick(
        x=dfo["time"], open=dfo["open"], high=dfo["high"], low=dfo["low"], close=dfo["close"],
        increasing_line_color="green", decreasing_line_color="red",
        increasing_fillcolor="green", decreasing_fillcolor="red", name="OHLC"
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=10,r=10,t=30,b=10))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["high"].rolling(5).max(), mode="lines", name="High trend"))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["low"].rolling(5).min(), mode="lines", name="Low trend"))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["ema_5"], mode="lines", name="EMA 5"))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["ema_20"], mode="lines", name="EMA 20"))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_up"], mode="lines", name="BB Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_mid"], mode="lines", name="BB Mid", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=dfo["time"], y=dfo["bb_dn"], mode="lines", name="BB Lower", line=dict(dash="dot")))
    cross_up = (dfo["ema_5"] > dfo["ema_20"]) & (dfo["ema_5"].shift(1) <= dfo["ema_20"].shift(1))
    cross_dn = (dfo["ema_5"] < dfo["ema_20"]) & (dfo["ema_5"].shift(1) >= dfo["ema_20"].shift(1))
    fig.add_trace(go.Scatter(x=dfo["time"][cross_up], y=dfo["close"][cross_up], mode="markers", name="Bullish X", marker_symbol="triangle-up", marker_size=10))
    fig.add_trace(go.Scatter(x=dfo["time"][cross_dn], y=dfo["close"][cross_dn], mode="markers", name="Bearish X", marker_symbol="triangle-down", marker_size=10))
    st.plotly_chart(fig, use_container_width=True)

    # Signals
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

    # News Sentiment (RSS filtered)
    news_items = fetch_coin_news_rss(coin_name, ticker, max_items=60)
    summary = summarize_news_coin(news_items)

    st.metric("Bullish Probability", f"{prob_bull*100:.1f}%")
    st.metric("RSI(14)", f"{rsi_val:.1f}")
    st.metric("MACD Hist", f"{macd_hist:.5f}")
    if summary["trend"] == "bullish":
        st.success(f"News: BULLISH ({summary['score']:+d})")
    elif summary["trend"] == "bearish":
        st.error(f"News: BEARISH ({summary['score']:+d})")
    else:
        st.warning(f"News: NEUTRAL ({summary['score']:+d})")

    ideas = ai_suggestions(prob_bull, rsi_val, macd_hist, dfo, summary["score"], acc)
    if ideas["mood"] == "buy":
        st.success("Action: BUY / ADD")
    elif ideas["mood"] == "sell":
        st.error("Action: SELL / HEDGE")
    else:
        st.info("Action: WAIT / NEUTRAL")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Entry", f"{ideas['entry']:.4f} {vs_currency.upper()}")
    colB.metric("Target", f"{ideas['target']:.4f} {vs_currency.upper()}")
    colC.metric("Stop", f"{ideas['stop']:.4f} {vs_currency.upper()}")
    colD.metric("Pos Size", f"{ideas['pos_size_pct']*100:.1f}%")

    # Log current signal
    log_signal(coin_label, ideas["mood"], prob_bull, rsi_val, macd_hist, summary["score"], acc)

    # Flip detection + alerts
    key = f"last_signal_{coin_id}"
    current_signal = ideas["mood"]
    if key not in st.session_state:
        st.session_state[key] = current_signal
    flipped = (current_signal != st.session_state[key])
    st.caption(f"Signal: **{current_signal.upper()}** {'(flipped)' if flipped else ''}")
    if flipped and (discord_webhook or tg_token or (smtp_host and smtp_from and smtp_to)):
        results = try_alerts(current_signal, ideas, coin_label, ideas['last'], {
            "discord_webhook": discord_webhook, "tg_token": tg_token, "tg_chat": tg_chat,
            "smtp_host": smtp_host, "smtp_port": smtp_port, "smtp_user": smtp_user, "smtp_pass": smtp_pass,
            "smtp_from": smtp_from, "smtp_to": smtp_to
        })
        log_alert(coin_label, current_signal, ideas['last'], ideas, results)
        st.toast(f"Alert sent for {coin_label}: {current_signal}", icon="🔔")
        st.write("Alert results:", results)
        st.session_state[key] = current_signal

def render_logs():
    st.subheader("📜 Logs")
    # CSV signals summary
    sig_csv = os.path.join(LOG_DIR, "signals_stream.csv")
    if os.path.exists(sig_csv):
        df = pd.read_csv(sig_csv)
        st.dataframe(df.tail(200), use_container_width=True)
    else:
        st.info("No signal logs yet.")
    if os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as conn:
            dfA = pd.read_sql_query("SELECT * FROM alerts ORDER BY ts DESC LIMIT 200", conn)
            st.write("Alerts (latest 200)")
            st.dataframe(dfA, use_container_width=True)
    else:
        st.info("No alert DB yet.")

if st.button("Show Logs"):
    render_logs()

def run_watchlist():
    coins = [c.strip() for c in coins_text.split(",") if c.strip()]
    if not coins:
        st.warning("Please enter at least one CoinGecko ID (e.g., bitcoin).")
        return
    if layout_choice == "Tabs":
        tabs = st.tabs([c for c in coins])
        for tab, coin in zip(tabs, coins):
            with tab:
                render_coin_block(coin)
    else:
        for coin in coins:
            with st.container():
                render_coin_block(coin)
                st.divider()

if run_btn:
    run_watchlist()

if 'last_auto' not in st.session_state:
    st.session_state['last_auto'] = 0.0
if refresh:
    import time as _t
    now = _t.time()
    if now - st.session_state['last_auto'] > 60:
        st.session_state['last_auto'] = now
        st.rerun()
