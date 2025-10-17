import streamlit as st
import pandas as pd
import numpy as np
import requests, os, re, time, sqlite3, smtplib, tempfile
from email.message import EmailMessage
from datetime import datetime
from typing import Dict, List, Tuple
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import feedparser

# ============== Safe log dir ==============
def _resolve_log_dir() -> str:
    env_path = os.getenv("LOG_DIR")
    if env_path:
        try:
            os.makedirs(env_path, exist_ok=True)
            if os.access(env_path, os.W_OK):
                return env_path
        except Exception:
            pass
    home = os.path.join(os.path.expanduser("~"), "crypto_logs")
    try:
        os.makedirs(home, exist_ok=True)
        if os.access(home, os.W_OK):
            return home
    except Exception:
        pass
    return tempfile.gettempdir()

LOG_DIR = _resolve_log_dir()
CSV_PATH = os.path.join(LOG_DIR, "signals.csv")
DB_PATH = os.path.join(LOG_DIR, "signals.db")

# ============== CoinGecko helpers ==============
def cg_get(path, params=None, api_key=None, base_url="https://api.coingecko.com/api/v3", max_retries=5):
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {}
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.get(url, params=params or {}, headers=headers, timeout=30)
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            wait = float(r.headers.get("retry-after", backoff))
            time.sleep(wait)
            backoff = min(backoff*2, 16)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

@st.cache_data(ttl=300, show_spinner=False)
def cg_ping(api_key=None, base_url="https://api.coingecko.com/api/v3"):
    return cg_get("/ping", api_key=api_key, base_url=base_url)

@st.cache_data(ttl=300, show_spinner=False)
def cg_search_coins(query, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get("/search", params={"query": query}, api_key=api_key, base_url=base_url)
    return data.get("coins", [])

@st.cache_data(ttl=180, show_spinner=False)
def fetch_market_chart(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/market_chart", params={"vs_currency": vs_currency, "days": days}, api_key=api_key, base_url=base_url)
    dfp = pd.DataFrame(data.get("prices", []), columns=["ts","price"])
    dfv = pd.DataFrame(data.get("total_volumes", []), columns=["ts","volume"])
    if dfp.empty: 
        return pd.DataFrame()
    df = pd.merge(dfp, dfv, on="ts", how="left")
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    return df.sort_values("time")[["time","price","volume"]].reset_index(drop=True)

@st.cache_data(ttl=180, show_spinner=False)
def fetch_ohlc(coin_id: str, vs_currency: str, days: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}/ohlc", params={"vs_currency": vs_currency, "days": days}, api_key=api_key, base_url=base_url)
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts","open","high","low","close"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Asia/Manila")
    return df.sort_values("time")[["time","open","high","low","close"]].reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_coin_symbol(coin_id: str, api_key=None, base_url="https://api.coingecko.com/api/v3"):
    data = cg_get(f"/coins/{coin_id}", params={"localization":"false","tickers":"false","market_data":"false","community_data":"false","developer_data":"false","sparkline":"false"}, api_key=api_key, base_url=base_url)
    return (data.get("symbol","").upper() or coin_id[:4].upper(), data.get("name", coin_id))

# ============== Indicators & features ==============
def ema(series: pd.Series, span:int): return series.ewm(span=span, adjust=False).mean()
def rsi(series: pd.Series, period:int=14):
    d=series.diff()
    up=d.clip(lower=0).rolling(period).mean()
    dn=(-d.clip(upper=0)).rolling(period).mean()
    rs=up/(dn.replace(0,np.nan))
    return 100-(100/(1+rs))
def macd(series: pd.Series, fast=12, slow=26, signal=9):
    m=ema(series,fast)-ema(series,slow); s=ema(m,signal); h=m-s; return m,s,h
def bollinger(series: pd.Series, window=20, num_std=2):
    ma=series.rolling(window).mean(); sd=series.rolling(window).std()
    return ma, ma+num_std*sd, ma-num_std*sd

def build_features(df):
    df=df.copy()
    df["ret"]=df["price"].pct_change()
    for w in [5,10,20,50]:
        df[f"sma_{w}"]=df["price"].rolling(w).mean()
        df[f"ema_{w}"]=ema(df["price"], w)
        df[f"mom_{w}"]=df["price"].pct_change(w)
        df[f"vol_{w}"]=df["ret"].rolling(w).std()
        df[f"v_chg_{w}"]=df["volume"].pct_change(w)
    df["rsi_14"]=rsi(df["price"],14)
    m,s,h=macd(df["price"],12,26,9)
    df["macd"]=m; df["macd_signal"]=s; df["macd_hist"]=h
    for c in ["sma_5","sma_10","sma_20","sma_50","ema_5","ema_10","ema_20","ema_50"]:
        df[c]=df[c]/df["price"]
    df["target"]=(df["price"].shift(-1)>df["price"]).astype(int)
    df=df.dropna().reset_index(drop=True)
    cols=["ret","mom_5","mom_10","mom_20","vol_5","vol_10","vol_20","v_chg_5","v_chg_10","v_chg_20",
          "rsi_14","macd","macd_signal","macd_hist","ema_5","ema_10","ema_20","ema_50","sma_5","sma_10","sma_20","sma_50"]
    return df, cols

# ============== Model training & caching ==============
@st.cache_resource(show_spinner=False)
def get_model_bundle(key: str, Xtr, ytr, Xva, yva):
    model=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,
                        reg_lambda=1.0,objective="binary:logistic",eval_metric="logloss",tree_method="hist")
    scaler=StandardScaler()
    Xtr_s=scaler.fit_transform(Xtr)
    Xva_s=scaler.transform(Xva)
    model.fit(Xtr_s,ytr,eval_set=[(Xva_s,yva)],verbose=False)
    return {"model":model, "scaler":scaler}

def ts_split(df, cols, ratio=0.8):
    X=df[cols].values; y=df["target"].values; i=int(len(df)*ratio)
    return X[:i], y[:i], X[i:], y[i:], i

def eval_model(model, Xt_s, yt):
    p=model.predict_proba(Xt_s)[:,1]; y=(p>=0.5).astype(int)
    return accuracy_score(yt,y), confusion_matrix(yt,y), classification_report(yt,y,zero_division=0), p, y

# ============== News sentiment (coin-specific RSS) ==============
NEWS_RSS=["https://www.coindesk.com/arc/outboundfeeds/rss/",
          "https://cointelegraph.com/rss",
          "https://www.theblock.co/rss"]
BULL=["surge","rally","record","all-time high","spike","bull","buy","breakout","soars","jumps","uptrend","green"]
BEAR=["drop","dump","selloff","bear","crash","plunge","falls","downtrend","red","decline","slump"]

def _mentions(text,name,ticker):
    t=text.lower()
    return (name.lower() in t) or (re.search(rf"\\b{re.escape(ticker.lower())}\\b", t) is not None)

@st.cache_data(ttl=180, show_spinner=False)
def coin_news(name,ticker,max_items=50):
    out=[]
    for url in NEWS_RSS:
        try:
            f=feedparser.parse(url)
            for e in f.entries[:max_items//len(NEWS_RSS)+1]:
                title=e.get("title","").strip()
                summary=re.sub("<.*?>","", e.get("summary","")).strip()
                if _mentions(title+" "+summary,name,ticker):
                    out.append({"title":title,"summary":summary,"link":e.get("link","")})
        except Exception:
            continue
    return out

def news_summary(items):
    if not items: return {"score":0,"trend":"neutral","top": []}
    def sc(t):
        t=t.lower(); s=0
        for w in BULL: 
            if w in t: s+=1
        for w in BEAR:
            if w in t: s-=1
        return s
    scored=[(sc(i["title"]+" "+i.get("summary","")),i) for i in items]
    total=sum(s for s,_ in scored)
    trend="bullish" if total>1 else ("bearish" if total<-1 else "neutral")
    top=[i for s,i in sorted(scored,key=lambda x:x[0],reverse=True)][:5]
    return {"score": total, "trend": trend, "top": top}

# ============== AI suggestion & risk ==============
def atr_like(series, window=14):
    return float(series.pct_change().rolling(window).std().iloc[-1])
def swing_levels(ohlc, lookback=20):
    r=ohlc.tail(min(lookback,len(ohlc)))
    return float(r["low"].min()), float(r["high"].max()), float(r["close"].iloc[-1])
def pos_size(prob_adj, vol, acc, cap_min=0.02, cap_max=0.1):
    conf=max(0.0,(prob_adj-0.5)*2)*(0.5+0.5*acc)
    size=cap_min+(cap_max-cap_min)*conf*(1.0/(1.0+10.0*vol))
    return float(min(cap_max,max(cap_min,size)))
def suggest(prob, rsi_val, macd_hist, ohlc, news_score, acc):
    lo,hi,last=swing_levels(ohlc)
    vol=atr_like(ohlc["close"])
    bias=0.05 if news_score>0 else (-0.05 if news_score<0 else 0.0)
    p=np.clip(prob+bias,0,1)
    buf=last*max(0.005,min(0.03,vol*1.5))
    if p>=0.6 and macd_hist>=0:
        entry=max(lo,last-buf); target=hi; stop=min(lo*0.99,last-2*buf); mood="buy"; msg="Bullish setup — buy pullbacks to support."
    elif p<=0.4 and macd_hist<=0:
        entry=min(hi,last+buf); target=lo; stop=max(hi*1.01,last+2*buf); mood="sell"; msg="Bearish pressure — consider selling near resistance."
    else:
        entry=last; target=hi if p>=0.5 else lo; stop=lo if p>=0.5 else hi; mood="wait"; msg="Mixed signals — wait for break or reduce size."
    size=pos_size(p,vol,acc)
    return {"mood":mood,"entry":float(entry),"target":float(target),"stop":float(stop),"last":float(last),"size":float(size),
            "context":{"prob_raw":float(prob),"prob_adj":float(p),"rsi":float(rsi_val),"macd_hist":float(macd_hist),"news":float(news_score),"acc":float(acc)},"note":msg}

# ============== Alerts ==============
def alert_discord(url, content):
    try: r=requests.post(url,json={"content":content},timeout=15); return r.status_code
    except Exception: return -1
def alert_tg(token, chat, text):
    try: r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",data={"chat_id":chat,"text":text},timeout=15); return r.status_code
    except Exception: return -1
def alert_email(host,port,user,pw,from_addr,to_addr,subject,body,use_tls=True):
    try:
        msg=EmailMessage(); msg["From"]=from_addr; msg["To"]=to_addr; msg["Subject"]=subject; msg.set_content(body)
        if use_tls:
            with smtplib.SMTP(host,int(port)) as s:
                s.starttls(); 
                if user: s.login(user,pw)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host,int(port)) as s:
                if user: s.login(user,pw)
                s.send_message(msg)
        return 200
    except Exception:
        return -1

def send_alerts(signal, ideas, coin_label, opts):
    msg=f"[{coin_label}] Signal: {signal}\\nPrice: {ideas['last']:.4f}\\nEntry:{ideas['entry']:.4f}  Target:{ideas['target']:.4f}  Stop:{ideas['stop']:.4f}\\nPos Size:{ideas['size']*100:.1f}%\\nNote: {ideas['note']}"
    res={}
    if opts.get("discord"): res["discord"]=alert_discord(opts["discord"], msg)
    if opts.get("tg_token") and opts.get("tg_chat"): res["telegram"]=alert_tg(opts["tg_token"], opts["tg_chat"], msg)
    if opts.get("smtp_host") and opts.get("smtp_from") and opts.get("smtp_to"):
        res["email"]=alert_email(opts["smtp_host"], opts.get("smtp_port",587), opts.get("smtp_user",""), opts.get("smtp_pass",""),
                                 opts["smtp_from"], opts["smtp_to"], f"[{coin_label}] {signal}", msg, use_tls=True)
    return res

# ============== Logs (CSV + SQLite) ==============
def init_db():
    os.makedirs(LOG_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS alerts(ts TEXT, coin TEXT, signal TEXT, price REAL, entry REAL, target REAL, stop REAL, size REAL, channel TEXT, code INTEGER, note TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS signals(ts TEXT, coin TEXT, mood TEXT, prob REAL, rsi REAL, macd REAL, news REAL, acc REAL)""")
        c.commit()

def log_signal(coin, mood, prob, rsi, macd, news, acc):
    init_db(); ts=datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",(ts,coin,mood,float(prob),float(rsi),float(macd),float(news),float(acc))); c.commit()
    row={"ts":ts,"coin":coin,"mood":mood,"prob":prob,"rsi":rsi,"macd":macd,"news":news,"acc":acc}
    sig_csv=os.path.join(LOG_DIR,"signals_stream.csv")
    pd.DataFrame([row]).to_csv(sig_csv, mode="a", header=not os.path.exists(sig_csv), index=False)

def log_alerts(coin, signal, ideas, results):
    init_db(); ts=datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        for ch,code in results.items():
            c.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?)",(ts,coin,signal,ideas['last'],ideas['entry'],ideas['target'],ideas['stop'],ideas['size'],ch,int(code),ideas['note']))
        c.commit()
    row={"ts":ts,"coin":coin,"signal":signal,"price":ideas['last'],"entry":ideas['entry'],"target":ideas['target'],"stop":ideas['stop'],"size":ideas['size'],"channels":"|".join(results.keys()),"codes":"|".join(map(str,results.values()))}
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)

# ============== UI (collapsible, speed mode) ==============
st.set_page_config(page_title="Crypto Bull or Bear • v9.4", page_icon="📈", layout="wide")
st.title("📈 Crypto Bull or Bear — v9.4")
st.caption("Educational demo. CoinGecko + XGBoost + RSI/MACD + coin‑specific news. Not financial advice.")

with st.sidebar:
    st.subheader("Last session")
    st.info(f"Coin: {st.session_state.get('last_coin','—')}  |  Signal: {st.session_state.get('last_signal','—').upper() if st.session_state.get('last_signal') else '—'}")

    with st.expander("⚙️ Settings", expanded=False):
        api_key = st.text_input("CoinGecko API Key (optional)", type="password")
        base_url = st.selectbox("CG API", ["https://api.coingecko.com/api/v3","https://pro-api.coingecko.com/api/v3"], index=0)
        vs_currency = st.selectbox("Quote", ["usd","eur","php","jpy"], index=0)
        days = st.selectbox("Lookback", ["7","14","30","90","180","365"], index=2)
        speed_mode = st.toggle("⚡ Speed Mode (skip retrain if model cached)", value=True)
        auto_refresh = st.checkbox("Auto‑refresh every 60s", value=False)

    with st.expander("🔍 Search", expanded=True):
        query = st.text_input("Search coin (name or ticker)", value=st.session_state.get("last_query","bitcoin"))
        if st.button("Search"):
            try:
                res = cg_search_coins(query, api_key=api_key, base_url=base_url)
                st.session_state["search_hits"]=[(c.get("id"), f"{c.get('name')} ({c.get('symbol').upper()})") for c in res[:15]]
                st.success(f"Found {len(st.session_state['search_hits'])} result(s).")
            except Exception as e:
                st.error(f"Search error: {e}")
        hits = st.session_state.get("search_hits",[("bitcoin","Bitcoin (BTC)")])
        coin_id = st.selectbox("Pick coin", [h[0] for h in hits], format_func=lambda x: dict(hits).get(x,x))
        st.session_state["last_query"] = query

    with st.expander("🔔 Alerts", expanded=False):
        discord = st.text_input("Discord Webhook URL")
        tg_token = st.text_input("Telegram Bot Token")
        tg_chat = st.text_input("Telegram Chat ID")
        smtp_host = st.text_input("SMTP Host")
        smtp_port = st.number_input("SMTP Port", value=587, step=1)
        smtp_user = st.text_input("SMTP User")
        smtp_pass = st.text_input("SMTP Pass", type="password")
        smtp_from = st.text_input("From Email")
        smtp_to = st.text_input("To Email")
        if st.button("Send Test Alert"):
            res=send_alerts("TEST", {"last":0,"entry":0,"target":0,"stop":0,"size":0,"note":"config test"}, "TEST", 
                            {"discord":discord,"tg_token":tg_token,"tg_chat":tg_chat,"smtp_host":smtp_host,"smtp_port":smtp_port,"smtp_user":smtp_user,"smtp_pass":smtp_pass,"smtp_from":smtp_from,"smtp_to":smtp_to})
            st.write("Test results:", res)

    with st.expander("ℹ️ Logs & Status", expanded=False):
        st.caption(f"Logs path: {LOG_DIR}")

run = st.button("Run / Refresh", type="primary", use_container_width=True)

# Connectivity ping
try:
    _ = cg_ping(api_key=api_key, base_url=base_url)
    st.toast("CoinGecko OK ✅", icon="✅")
except Exception as e:
    st.error(f"CoinGecko unreachable: {e}")

def progress_step(pbar, txt, pct):
    pbar.progress(pct, text=txt)

def run_once(coin_id):
    st.session_state["last_coin_id"]=coin_id
    sym,name=fetch_coin_symbol(coin_id, api_key=api_key, base_url=base_url)
    label=f"{name} ({sym})"

    pbar = st.progress(0, text="Starting…")
    progress_step(pbar, "1️⃣ Fetching data…", 10)
    df=fetch_market_chart(coin_id, vs_currency, days, api_key=api_key, base_url=base_url)
    ohlc=fetch_ohlc(coin_id, vs_currency, days, api_key=api_key, base_url=base_url)
    if df.empty or ohlc.empty or len(df)<120:
        pbar.empty()
        st.warning("Not enough data; try a longer lookback.")
        return None

    progress_step(pbar, "2️⃣ Computing indicators…", 35)
    ohlc=ohlc.copy()
    ohlc["ema5"]=ohlc["close"].ewm(span=5,adjust=False).mean()
    ohlc["ema20"]=ohlc["close"].ewm(span=20,adjust=False).mean()
    bb_mid, bb_up, bb_dn = bollinger(ohlc["close"],20,2)
    ohlc["bb_mid"]=bb_mid; ohlc["bb_up"]=bb_up; ohlc["bb_dn"]=bb_dn

    feats, cols = build_features(df)

    progress_step(pbar, "3️⃣ Modeling…", 60)
    key=f"{coin_id}_{vs_currency}_{days}"
    Xtr,ytr,Xt,yt,idx = ts_split(feats, cols, 0.8)

    # Cache model bundle per key; if Speed Mode and cache exists, skip retrain
    bundle=None
    if speed_mode and key in st.session_state.get("model_cache", {}):
        bundle = st.session_state["model_cache"][key]
    else:
        bundle=get_model_bundle(key, Xtr, ytr, Xt, yt)
        cache=st.session_state.get("model_cache", {})
        cache[key]=bundle
        st.session_state["model_cache"]=cache

    scaler=bundle["scaler"]; model=bundle["model"]
    Xt_s=scaler.transform(Xt)
    acc, cm, report, yprob, ypred = eval_model(model, Xt_s, yt)

    progress_step(pbar, "4️⃣ Predicting latest bar…", 80)
    last = feats.iloc[[-1]][cols].values
    last_p = float(model.predict_proba(scaler.transform(last))[0,1])
    rsi_val = float(feats.iloc[-1]["rsi_14"])
    macd_hist = float(feats.iloc[-1]["macd_hist"])

    items = coin_news(name, sym, 60)
    ns = news_summary(items)

    progress_step(pbar, "5️⃣ Rendering…", 95)

    # Chart
    fig = go.Figure(data=[go.Candlestick(x=ohlc["time"],open=ohlc["open"],high=ohlc["high"],low=ohlc["low"],close=ohlc["close"],
                                         increasing_line_color="green",decreasing_line_color="red",
                                         increasing_fillcolor="green",decreasing_fillcolor="red",name="OHLC")])
    fig.update_layout(xaxis_rangeslider_visible=False, height=430, margin=dict(l=6,r=6,t=30,b=6))
    fig.add_trace(go.Scatter(x=ohlc["time"], y=ohlc["ema5"], mode="lines", name="EMA5"))
    fig.add_trace(go.Scatter(x=ohlc["time"], y=ohlc["ema20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=ohlc["time"], y=ohlc["bb_up"], mode="lines", name="BB Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ohlc["time"], y=ohlc["bb_mid"], mode="lines", name="BB Mid", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ohlc["time"], y=ohlc["bb_dn"], mode="lines", name="BB Lower", line=dict(dash="dot")))

    st.subheader(label)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bullish Prob", f"{last_p*100:.1f}%")
    c2.metric("RSI(14)", f"{rsi_val:.1f}")
    c3.metric("MACD Hist", f"{macd_hist:.5f}")
    c4.metric("News", f"{ns['trend'].upper()} ({ns['score']:+d})")

    idea = suggest(last_p, rsi_val, macd_hist, ohlc, ns["score"], acc)
    if idea["mood"]=="buy": st.success("Action: BUY / ADD")
    elif idea["mood"]=="sell": st.error("Action: SELL / HEDGE")
    else: st.info("Action: WAIT / NEUTRAL")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Entry", f"{idea['entry']:.4f} {vs_currency.upper()}")
    d2.metric("Target", f"{idea['target']:.4f} {vs_currency.upper()}")
    d3.metric("Stop", f"{idea['stop']:.4f} {vs_currency.upper()}")
    d4.metric("Pos Size", f"{idea['size']*100:.1f}%")
    with st.expander("Top Headlines"):
        for it in ns["top"]:
            st.write(f"• [{it['title']}]({it['link']})")

    # Save last signal
    st.session_state["last_coin"]=label
    st.session_state["last_signal"]=idea["mood"]

    # Alerts on flip
    key_sig="last_signal_runtime"
    prev=st.session_state.get(key_sig)
    flipped = prev is not None and prev != idea["mood"]
    st.caption(f"Signal: **{idea['mood'].upper()}** {'(flipped)' if flipped else ''}")
    st.session_state[key_sig]=idea["mood"]

    # fire alerts if configured
    opts={"discord":st.session_state.get("discord",""),
          "tg_token":st.session_state.get("tg_token",""),
          "tg_chat":st.session_state.get("tg_chat",""),
          "smtp_host":st.session_state.get("smtp_host",""),
          "smtp_port":st.session_state.get("smtp_port",587),
          "smtp_user":st.session_state.get("smtp_user",""),
          "smtp_pass":st.session_state.get("smtp_pass",""),
          "smtp_from":st.session_state.get("smtp_from",""),
          "smtp_to":st.session_state.get("smtp_to","")}
    if flipped and any([opts["discord"], opts["tg_token"] and opts["tg_chat"], opts["smtp_host"] and opts["smtp_from"] and opts["smtp_to"]]):
        res=send_alerts(idea["mood"], idea, label, opts)
        log_alerts(label, idea["mood"], idea, res)
        st.toast("Alert sent.", icon="🔔")

    # Log signal
    log_signal(label, idea["mood"], last_p, rsi_val, macd_hist, ns["score"], acc)

    pbar.progress(100, text="✅ Done")
    time.sleep(0.2)
    pbar.empty()
    return True

# Persist alert settings
def _persist_alerts_to_session():
    for k in ["discord","tg_token","tg_chat","smtp_host","smtp_port","smtp_user","smtp_pass","smtp_from","smtp_to"]:
        if k in st.session_state: continue  # don't overwrite
    # no-op; values are written when user opens Alerts and interacts

if run:
    # capture latest alert fields for session
    st.session_state["discord"]=st.session_state.get("discord","")
    st.session_state["tg_token"]=st.session_state.get("tg_token","")
    st.session_state["tg_chat"]=st.session_state.get("tg_chat","")
    st.session_state["smtp_host"]=st.session_state.get("smtp_host","")
    st.session_state["smtp_port"]=st.session_state.get("smtp_port",587)
    st.session_state["smtp_user"]=st.session_state.get("smtp_user","")
    st.session_state["smtp_pass"]=st.session_state.get("smtp_pass","")
    st.session_state["smtp_from"]=st.session_state.get("smtp_from","")
    st.session_state["smtp_to"]=st.session_state.get("smtp_to","")
    run_once(coin_id)

# Auto-refresh (uses cached data/model for speed)
if auto_refresh:
    if "last_auto" not in st.session_state: st.session_state["last_auto"]=0.0
    import time as _t
    now=_t.time()
    if now-st.session_state["last_auto"]>60:
        st.session_state["last_auto"]=now
        st.rerun()
