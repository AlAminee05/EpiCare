"""Personal layer (§4.2), fusion and calibration (§4.4)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass
class PersonalLayer:
    """Logistic regression on the shared representation, trained on one patient only.

    Strong L2 (small C) keeps the effective number of parameters low so 10-20 events suffice.
    No class re-weighting: the output must stay a calibrated probability.
    """
    C: float = 0.1
    model: object = None
    train_prior: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersonalLayer":
        self.train_prior = float(np.mean(y))
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=self.C, max_iter=2000))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def top_features(self, names: list[str], x: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Features pushing the current hour's risk up the most (coef * standardised value)."""
        sc, lr = self.model.steps[0][1], self.model.steps[1][1]
        contrib = lr.coef_[0] * (x - sc.mean_) / np.where(sc.scale_ > 0, sc.scale_, 1)
        order = np.argsort(contrib)[::-1][:k]
        return [(names[i], float(contrib[i])) for i in order if contrib[i] > 0]


def prior_shift_fusion(p_wear: np.ndarray, wear_train_prior: float, p_cycle: np.ndarray) -> np.ndarray:
    """SeFEF LR-vMe baseline: shift the personal layer's intercept by the cycle log-odds ratio.

    logit p = logit p_wear - logit(prior used to train p_wear) + logit p_cycle
    """
    return sigmoid(logit(p_wear) - logit(wear_train_prior) + logit(p_cycle))


def fusion_inputs(p_wear, p_cycle, hours_since_seizure) -> np.ndarray:
    cols = [logit(p_cycle), np.log1p(hours_since_seizure)]
    if p_wear is not None:
        cols.insert(0, logit(p_wear))
    return np.column_stack(cols)


@dataclass
class LearnedFusion:
    """Our improvement (§4.4): per-patient logistic regression on
    [logit p_wear, logit p_cycle, log(1 + hours since last seizure)], fitted on held-out
    (later-in-time) training data, which also makes it a Platt-type calibration."""
    C: float = 1.0
    model: LogisticRegression = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LearnedFusion":
        self.model = LogisticRegression(C=self.C, max_iter=1000).fit(Z, y)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(Z)[:, 1]

    @property
    def weights(self) -> dict:
        return {"coef": self.model.coef_[0].round(3).tolist(), "intercept": round(float(self.model.intercept_[0]), 3)}


@dataclass
class Calibrator:
    method: str = "platt"   # "platt" | "isotonic" | "none"
    model: object = field(default=None, repr=False)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self.model = LogisticRegression(C=1e3, max_iter=1000).fit(logit(p)[:, None], y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.model is None or self.method == "none":
            return p
        if self.method == "platt":
            return self.model.predict_proba(logit(p)[:, None])[:, 1]
        return self.model.predict(p)
