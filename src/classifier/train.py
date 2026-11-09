"""
Dual-Head Prompt Classifier Training Pipeline for ALIGN-PARETO (M1).
Architecture (spec §15):
  Prompt text -> frozen sentence embedding (all-MiniLM-L6-v2 by default; with `embedding_candidates`
  set, the encoder and each head's C are chosen on the validation split)
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

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 256, prefix: str = ""):
        self.model_name = model_name
        self.batch_size = batch_size
        self.prefix = prefix  # e.g. "query: " for e5 models
        self.backend = "sentence_transformers"
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}  # text -> embedding, so repeated prompts are embedded once

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
        state["_cache"] = {}
        return state

    def __setstate__(self, state):
        state.setdefault("prefix", "")  # bundles pickled before prefixes existed
        state.setdefault("_cache", {})
        self.__dict__.update(state)

    def transform(self, texts: List[str]) -> np.ndarray:
        new = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if new:
            emb = self.model.encode(
                [self.prefix + t for t in new], batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True
            )
            for t, e in zip(new, np.asarray(emb, dtype=np.float64)):
                self._cache[t] = e
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 0))


class BalancedLogisticRegressionHead:
    """
    Balanced L2-Regularized Logistic Regression head, fitted with SciPy L-BFGS-B, plus Platt scaling on the
    validation split. The objective matches scikit-learn's LogisticRegression(C, class_weight="balanced")
    divided by C*n: mean(weighted log-loss) + ||w||^2 / (2*C*n), with an unpenalized intercept.
    Platt scaling is fitted to the validation split's label mix, so probabilities reflect the pilot
    dataset's class balance, not real-world prompt traffic.
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


def _labelled(df, col: str) -> np.ndarray:
    """Boolean mask of rows that have a label in `col` (safety-only rows have no creative label)."""
    return df[col].notna().to_numpy()


def _make_head(model_cfg: dict, C: float, seed: int) -> "BalancedLogisticRegressionHead":
    return BalancedLogisticRegressionHead(
        C=C, max_iter=model_cfg.get("max_iter", 1000), class_weight=model_cfg.get("class_weight", "balanced"), seed=seed
    )


def select_embedder_and_C(train_df, val_df, config: dict, max_candidates: Optional[int] = None) -> Tuple[SentenceEmbedder, Dict[str, float], List[Dict]]:
    """
    Choose the sentence encoder and each head's regularization C on the validation split.
    Each (encoder, head, C) is scored by validation ROC-AUC (threshold-free, and unaffected by Platt
    scaling); an encoder's score is the mean of its two heads' best ROC-AUCs. The test split is never used.
    Without `embedding_candidates` in the config, returns the single configured encoder and C.
    """
    from src.classifier.evaluate import compute_roc_curve_and_auc

    model_cfg = config.get("model", {})
    seed = int(config.get("seed", 42))
    batch_size = int(config["training"]["batch_size"])
    candidates = model_cfg.get("embedding_candidates") or [
        {"name": model_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"), "prefix": ""}
    ]
    if max_candidates:
        candidates = candidates[:max_candidates]
    c_grid = [float(c) for c in (model_cfg.get("C_grid") or [model_cfg.get("C", 2.0)])]
    cols = {"creative": "creative_label", "safety": "safety_label"}

    table, best = [], None
    for cand in candidates:
        emb = SentenceEmbedder(cand["name"], batch_size=batch_size, prefix=cand.get("prefix", ""))
        X_tr = emb.transform(train_df["prompt"].tolist())
        X_va = emb.transform(val_df["prompt"].tolist())
        row = {"embedding_model": cand["name"], "prefix": emb.prefix, "embedding_dim": int(X_tr.shape[1]), "val_roc_auc": {}}
        best_c = {}
        for head, col in cols.items():
            m_tr, m_va = _labelled(train_df, col), _labelled(val_df, col)
            y_tr = train_df[col].to_numpy(dtype=float)[m_tr]
            y_va = val_df[col].to_numpy(dtype=float)[m_va].astype(int)
            scores = {}
            for C in c_grid:
                h = _make_head(model_cfg, C, seed).fit(X_tr[m_tr], y_tr)
                scores[C] = round(compute_roc_curve_and_auc(y_va, h.predict_proba(X_va[m_va])[:, 1])[2], 4)
            best_c[head] = max(scores, key=lambda c: (scores[c], -abs(np.log(c / 2.0))))  # ties -> closest to old C=2
            row["val_roc_auc"][head] = {str(c): v for c, v in scores.items()}
        row["best_C"] = best_c
        # A best C at the end of the grid means the optimum may lie outside it: widen C_grid
        row["best_C_at_grid_edge"] = {h: c in (min(c_grid), max(c_grid)) for h, c in best_c.items()}
        row["score_mean_best_val_roc_auc"] = round(float(np.mean([row["val_roc_auc"][h][str(best_c[h])] for h in cols])), 4)
        table.append(row)
        if best is None or row["score_mean_best_val_roc_auc"] > best[2]["score_mean_best_val_roc_auc"]:
            if best is not None:
                best[0]._cache.clear()
            best = (emb, best_c, row)
        else:
            emb._cache.clear()  # free memory held by losing candidates
    for row in table:
        row["selected"] = row is best[2]
    return best[0], best[1], table


def train_dual_head_classifier(
    train_df,
    val_df,
    config: dict,
    embedder: Optional[SentenceEmbedder] = None,
    C: Optional[Dict[str, float]] = None,
) -> Tuple[SentenceEmbedder, BalancedLogisticRegressionHead, BalancedLogisticRegressionHead, Dict, Dict[str, np.ndarray]]:
    """
    Embed prompts and train independent `creative_head` and `safety_head` classifiers.
    `embedder` and per-head `C` normally come from `select_embedder_and_C`; without them the
    configured `embedding_model` and `C` are used. Also returns the train/val embeddings so callers
    can tune thresholds without re-embedding.
    """
    model_cfg = config.get("model", {})
    seed = int(config.get("seed", 42))
    if embedder is None:
        embedder = SentenceEmbedder(
            model_name=model_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=int(config["training"]["batch_size"]),
        )
    default_c = float(model_cfg.get("C", 2.0))
    C = {"creative": default_c, "safety": default_c, **(C or {})}

    X_train = embedder.transform(train_df["prompt"].tolist())
    X_val = embedder.transform(val_df["prompt"].tolist())

    # Each head trains (and is Platt-calibrated) only on rows that carry its label
    heads = {}
    for head, col in [("creative", "creative_label"), ("safety", "safety_label")]:
        m_tr, m_va = _labelled(train_df, col), _labelled(val_df, col)
        heads[head] = _make_head(model_cfg, C[head], seed).fit(
            X_train[m_tr], train_df[col].to_numpy(dtype=float)[m_tr], X_val[m_va], val_df[col].to_numpy(dtype=float)[m_va]
        )
    creative_head, safety_head = heads["creative"], heads["safety"]

    train_meta = {
        "embedding_backend": embedder.backend,
        "embedding_model": embedder.model_name,
        "embedding_prefix": embedder.prefix,
        "embedding_dim": int(X_train.shape[1]),
        "C": C,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "creative_head_train_samples": int(_labelled(train_df, "creative_label").sum()),
        "safety_head_train_samples": int(_labelled(train_df, "safety_label").sum()),
        "creative_platt_scale": [round(creative_head.platt_a_, 4), round(creative_head.platt_b_, 4)],
        "safety_platt_scale": [round(safety_head.platt_a_, 4), round(safety_head.platt_b_, 4)],
    }
    return embedder, creative_head, safety_head, train_meta, {"train": X_train, "val": X_val}
