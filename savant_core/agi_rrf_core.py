from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import math
import numpy as np
from numpy.linalg import norm
from scipy.fft import rfft

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def token_entropy(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter
    c = Counter(text)
    total = sum(c.values())
    ent = 0.0
    for _, v in c.items():
        p = v / total
        ent -= p * math.log(p + 1e-12)
    return float(ent)

def spectrum_features(vec: np.ndarray) -> Dict[str, float]:
    x = vec.astype(np.float64)
    y = np.abs(rfft(x))
    if y.size == 0:
        return {"freq_low": 0.0, "freq_mid": 0.0, "freq_high": 0.0, "dominant_frequency": 0.0}
    n = y.size
    low = y[: max(1, n//6)].mean()
    mid = y[max(1, n//6) : max(2, n//3)].mean() if n >= 3 else 0.0
    high = y[max(2, n//3) :].mean() if n >= 3 else 0.0
    dom = float(np.argmax(y)) / float(max(1, n-1))
    return {"freq_low": float(low), "freq_mid": float(mid), "freq_high": float(high), "dominant_frequency": dom}

@dataclass
class AGIRRFCore:
    embedder: Any
    meta_logit: Any
    expected_features: int = 15

    def _feature_names(self) -> List[str]:
        if hasattr(self.meta_logit, "feature_names_in_"):
            try:
                return list(self.meta_logit.feature_names_in_)
            except Exception:
                pass
        return [
            "semantic_margin",
            "cosine_prompt_answer",
            "token_entropy",
            "dirac_energy",
            "dirac_shell_std",
            "freq_low",
            "freq_mid",
            "freq_high",
            "coherence_ratio",
            "phi",
            "omega",
            "S_RRF",
            "C_RRF",
            "dominant_frequency",
            "Phi1_geometric",
        ]

    def extract_features(self, prompt: str, answer: str) -> Dict[str, float]:
        ep = np.array(self.embedder.encode(prompt, normalize_embeddings=False), dtype=np.float64)
        ea = np.array(self.embedder.encode(answer, normalize_embeddings=False), dtype=np.float64)

        cos = safe_cosine(ep, ea)
        semantic_margin = float(cos - 0.5)
        ent = float(token_entropy(prompt + answer))

        dirac_energy = float(norm(ep - ea))
        dirac_shell_std = float(np.std(np.abs(ep)) + np.std(np.abs(ea)))

        sf = spectrum_features(ea)
        coherence_ratio = float((abs(cos) + 1e-9) / (1.0 + ent))

        phi = 0.6321205588
        omega = float(2.0 * math.pi * max(1e-6, sf["dominant_frequency"]))

        S_RRF = float(sigmoid(2.0 * semantic_margin))
        C_RRF = float(1.0 - abs(0.5 - cos))
        Phi1_geometric = float(sigmoid(1.5 * cos))

        return {
            "semantic_margin": semantic_margin,
            "cosine_prompt_answer": cos,
            "token_entropy": ent,
            "dirac_energy": dirac_energy,
            "dirac_shell_std": dirac_shell_std,
            "freq_low": sf["freq_low"],
            "freq_mid": sf["freq_mid"],
            "freq_high": sf["freq_high"],
            "coherence_ratio": coherence_ratio,
            "phi": float(phi),
            "omega": omega,
            "S_RRF": S_RRF,
            "C_RRF": C_RRF,
            "dominant_frequency": sf["dominant_frequency"],
            "Phi1_geometric": Phi1_geometric,
        }

    def predict(self, prompt: str, answer: str) -> Dict[str, float]:
        feats = self.extract_features(prompt, answer)
        names = self._feature_names()
        x = np.array([feats.get(n, 0.0) for n in names], dtype=np.float64).reshape(1, -1)

        if x.shape[1] < self.expected_features:
            x = np.pad(x, ((0,0),(0, self.expected_features - x.shape[1])), mode="constant")
        elif x.shape[1] > self.expected_features:
            x = x[:, : self.expected_features]

        try:
            p_good = float(self.meta_logit.predict_proba(x)[0, 1])
        except Exception:
            s = float(self.meta_logit.decision_function(x)[0])
            p_good = sigmoid(s)

        return {
            "p_good": p_good,
            "SRRF": p_good,
            "CRRF": float(feats.get("C_RRF", 0.0)),
            "E_phi": float(0.25 + 0.5 * feats.get("cosine_prompt_answer", 0.0)),
            "cosine": float(feats.get("cosine_prompt_answer", 0.0)),
            "phi": float(feats.get("phi", 0.6321205588)),
            "features": {k: float(v) for k, v in feats.items()},
        }
