# Crypto Bull or Bear — v10 (Fast Mode)

**Built for speed**  
- Fast Mode **default**: fewer UI updates, single progress bar, cached models, async fetching.
- Optional **Minimal Chart** (line) for ultra-fast runs.
- Keeps v9 features conceptually but trims per‑step UI overhead.

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
- Use **30–90 days** lookback for richer signals while staying quick.
- If you want full progress visuals, toggle **Fast Mode** off.