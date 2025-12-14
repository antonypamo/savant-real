from __future__ import annotations

import os, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import joblib

from savant_core.agi_rrf_core import AGIRRFCore

app = FastAPI(title="Savant RRF API", version="0.2.0")

class JudgeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)

HF_TOKEN = os.getenv("HF_TOKEN", None)

print("🔄 [Startup] Loading embedder: antonypamo/RRFSAVANTMADE", flush=True)
embedder = SentenceTransformer("antonypamo/RRFSAVANTMADE")
print("✅ [Startup] Embedder loaded.", flush=True)

print("🔄 [Startup] Downloading meta-logit: antonypamo/RRFSavantMetaLogicV2/logreg_rrf_savant.joblib", flush=True)
meta_path = hf_hub_download(repo_id="antonypamo/RRFSavantMetaLogicV2", filename="logreg_rrf_savant.joblib", token=HF_TOKEN)
meta_logit = joblib.load(meta_path)
print("✅ [Startup] Meta-logit loaded.", flush=True)

expected = 15
try:
    if hasattr(meta_logit, "coef_"):
        expected = int(meta_logit.coef_.shape[1])
except Exception:
    pass

core = AGIRRFCore(embedder=embedder, meta_logit=meta_logit, expected_features=expected)

@app.get("/")
def root():
    return {"name":"Savant RRF API","status":"ok","version":"0.2.0"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/judge")
def judge(req: JudgeRequest):
    t0 = time.time()
    try:
        out = core.predict(req.prompt, req.answer)
        return {
            "scores": {
                "p_good": out["p_good"],
                "SRRF": out["SRRF"],
                "CRRF": out["CRRF"],
                "E_phi": out["E_phi"],
                "cosine": out["cosine"],
                "phi": out["phi"],
            },
            "features": out["features"],
            "meta": {
                "latency_s": float(time.time() - t0),
                "embedder": "antonypamo/RRFSAVANTMADE",
                "meta_logit": "antonypamo/RRFSavantMetaLogicV2/logreg_rrf_savant.joblib",
                "expected_features": expected
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
