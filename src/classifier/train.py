"""
Dual-Head Prompt Classifier Training Pipeline for ALIGN-PARETO (M1).
Architecture (spec §15):
  Prompt text -> 384-dim `all-MiniLM-L6-v2` sentence embedding
  -> Two independent balanced L2-regularized Logistic Regression heads (`creative_head`, `safety_head`)
  with Platt calibration on the validation split for well-calibrated routing probabilities.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class SentenceEmbedder:
    """
    `sentence-transformers` embedder. Fails loudly if the model cannot be loaded (spec §29.12).
    Pickles by model name only; the weights are reloaded from the Hugging Face cache on first use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = "sentence_transformers"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Could not load embedding model {self.model_name!r}: {type(e).__name__}: {e}") from e
        return self._model

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def transform(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float64)


class BalancedLogisticRegressionHead:
    """
    Balanced L2-Regularized Logistic Regression head with L-BFGS-B optimization
    and Platt scaling calibration. Uses `scikit-learn` if installed, or pure SciPy L-BFGS-B.
    """

    def __init__(self, C: float = 2.0, max_iter: int = 1000, class_weight: str = "balanced", seed: int = 42):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.seed = seed
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.platt_a_: float = 1.0
        self.platt_b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Compute balanced sample weights
        if self.class_weight == "balanced":
            n_pos = max(1.0, float(np.sum(y == 1)))
            n_neg = max(1.0, float(np.sum(y == 0)))
            w_pos = n_samples / (2.0 * n_pos)
            w_neg = n_samples / (2.0 * n_neg)
            sample_weights = np.where(y == 1, w_pos, w_neg)
        else:
            sample_weights = np.ones(n_samples, dtype=np.float64)

        reg_lambda = 1.0 / (self.C * n_samples)

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            w = params[:-1]
            b = params[-1]
            logits = X @ w + b
            probs = expit(logits)
            # Clip for numerical stability
            eps = 1e-12
            bce = -sample_weights * (y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
            loss = float(np.mean(bce) + 0.5 * reg_lambda * np.sum(w * w))
            diff = (probs - y) * sample_weights / n_samples
            grad_w = X.T @ diff + reg_lambda * w
            grad_b = float(np.sum(diff))
            return loss, np.append(grad_w, grad_b)

        init_params = np.zeros(n_features + 1, dtype=np.float64)
        res = minimize(
            loss_and_grad,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        self.coef_ = res.x[:-1]
        self.intercept_ = float(res.x[-1])

        # Platt scaling on validation set if provided to ensure well-calibrated probabilities
        if X_val is not None and y_val is not None and len(y_val) > 10:
            raw_logits = X_val @ self.coef_ + self.intercept_
            y_v = np.asarray(y_val, dtype=np.float64)

            def platt_obj(ab: np.ndarray) -> float:
                a, b = ab[0], ab[1]
                p = expit(a * raw_logits + b)
                eps = 1e-12
                return float(-np.mean(y_v * np.log(p + eps) + (1.0 - y_v) * np.log(1.0 - p + eps)))

            p_res = minimize(platt_obj, np.array([1.0, 0.0]), method="Nelder-Mead")
            if p_res.success and p_res.x[0] > 0.1:
                self.platt_a_ = float(p_res.x[0])
                self.platt_b_ = float(p_res.x[1])

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        raw_logits = X @ self.coef_ + self.intercept_
        cal_logits = self.platt_a_ * raw_logits + self.platt_b_
        p1 = expit(cal_logits)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_dual_head_classifier(
    train_df,
    val_df,
    config: dict,
) -> Tuple[SentenceEmbedder, BalancedLogisticRegressionHead, BalancedLogisticRegressionHead, Dict, Dict[str, np.ndarray]]:
    """
    Embed prompts and train independent `creative_head` and `safety_head` classifiers.
    Also returns the train/val embeddings so callers can tune thresholds without re-embedding.
    """
    model_cfg = config.get("model", {})
    seed = int(config.get("seed", 42))
    embedder = SentenceEmbedder(
        model_name=model_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=int(config["training"]["batch_size"]),
    )

    X_train = embedder.transform(train_df["prompt"].tolist())
    X_val = embedder.transform(val_df["prompt"].tolist())

    y_train_cre = train_df["creative_label"].to_numpy()
    y_val_cre = val_df["creative_label"].to_numpy()

    y_train_saf = train_df["safety_label"].to_numpy()
    y_val_saf = val_df["safety_label"].to_numpy()

    creative_head = BalancedLogisticRegressionHead(
        C=model_cfg.get("C", 2.0),
        max_iter=model_cfg.get("max_iter", 1000),
        class_weight=model_cfg.get("class_weight", "balanced"),
        seed=seed,
    ).fit(X_train, y_train_cre, X_val, y_val_cre)

    safety_head = BalancedLogisticRegressionHead(
        C=model_cfg.get("C", 2.0),
        max_iter=model_cfg.get("max_iter", 1000),
        class_weight=model_cfg.get("class_weight", "balanced"),
        seed=seed,
    ).fit(X_train, y_train_saf, X_val, y_val_saf)

    train_meta = {
        "embedding_backend": embedder.backend,
        "embedding_dim": int(X_train.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "creative_platt_scale": [round(creative_head.platt_a_, 4), round(creative_head.platt_b_, 4)],
        "safety_platt_scale": [round(safety_head.platt_a_, 4), round(safety_head.platt_b_, 4)],
    }
    return embedder, creative_head, safety_head, train_meta, {"train": X_train, "val": X_val}

