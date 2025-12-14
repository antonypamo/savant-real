# Savant RRF — Production-like Bench Repo (Real Models)

FastAPI + real HF models + production-like benchmark suite.

- Embedder: `antonypamo/RRFSAVANTMADE`
- Meta-logit: `antonypamo/RRFSavantMetaLogit` (`logreg_rrf_savant_15.joblib`)

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Run suite:
```bash
python bench/run_bench.py --base-url http://127.0.0.1:8000 --endpoint /judge --out artifacts
```

## Remote (Space)
```bash
python bench/run_bench.py --base-url https://antonypamo-apisavant2.hf.space --endpoint /judge --out artifacts
```

## k6
```bash
BASE_URL=http://127.0.0.1:8000 k6 run bench/k6_savant.js
```

## Auth
If your models are private, set `HF_TOKEN` (env var) or copy `.env.example` to `.env`.
