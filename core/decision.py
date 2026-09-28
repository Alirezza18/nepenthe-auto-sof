"""
core.decision
-------------
Decision-support layer for Auto-SOF: everything between "here is a Pareto
front" and "here is the design we should build".

Four capabilities (all pure NumPy/pandas, no UI code, fully unit-testable):

1. Uncertainty quantification
   - `ensemble_std`      per-design std across surrogate family members
   - `gpr_std`           analytic posterior std for Gaussian-process models
   - `predict_with_uncertainty`  unified (mean, std) regardless of model type

2. Adaptive sampling
   - `suggest_next_simulations`  ranks candidate designs by an acquisition
     score (uncertainty + distance from known data) so the engineer knows
     which new points are worth running through the expensive simulator.

3. Inverse design
   - `inverse_design`    given desired target values, searches the surrogate
     input space for designs whose predicted outputs best match the targets
     (single-objective GA over the surrogate, reusing the Auto-SOF GA core).

4. Decision making on a Pareto front
   - `pareto_knee`       knee-point detection (max distance to the front's
     extreme-point chord, after min-max normalisation)
   - `topsis_rank`       full TOPSIS ranking with user objective weights
   - `recommend_designs` combined helper returning ranked recommendations

All functions are deterministic given their seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline

from core.optimizer import OptimizationResult, run_genetic_algorithm

# ---------------------------------------------------------------------------
# 1. Uncertainty quantification
# ---------------------------------------------------------------------------


def _objective_signs(directions: Sequence[str]) -> np.ndarray:
    """Sign vector mapping natural directions to minimisation space."""
    return np.array([1.0 if d == "min" else -1.0 for d in directions], dtype=float)


def ensemble_std(surrogate_payload: dict, x: pd.DataFrame, n_seeds: int = 5) -> np.ndarray:
    """Std of predictions across bootstrap refits of the surrogate family.

    Robust, model-agnostic uncertainty: refits the same architecture on
    bootstrap resamples of the training data and measures prediction spread.
    Uses the payload's stored estimators where possible; falls back to
    resampling the *training predictions* when refitting is impractical.
    """
    model = surrogate_payload.get("model")
    # Hybrid stacked ensembles: std across base estimator predictions.
    if isinstance(model, dict) and "base_estimators" in model:
        base_preds = np.hstack([
            np.asarray(est.predict(x), dtype=float)
            for est in model["base_estimators"]
        ])
        return base_preds.std(axis=1, ddof=1) if base_preds.shape[1] > 1 else np.zeros(len(x))

    # Gaussian processes: analytic posterior std.
    gpr_std = gpr_std_from_payload(surrogate_payload, x)
    if gpr_std is not None:
        return gpr_std

    # Generic sklearn estimators: std across bootstrap refits.
    import copy

    rng = np.random.default_rng(42)
    n = len(surrogate_payload.get("x_train", pd.DataFrame()))
    if n < 10:
        # Not enough provenance for bootstrap: fall back to test residual std.
        return np.full(len(x), _fallback_sigma(surrogate_payload))

    bootstrap_preds = []
    for _ in range(max(2, min(n_seeds, 10))):
        est = copy.deepcopy(model)
        idx = rng.integers(0, n, size=n)
        try:
            est.fit(surrogate_payload["x_train"].iloc[idx], surrogate_payload["y_train"].iloc[idx])
        except Exception:
            return np.full(len(x), _fallback_sigma(surrogate_payload))
        pred = np.asarray(est.predict(x), dtype=float)
        bootstrap_preds.append(pred.reshape(len(x), -1))
    return np.stack(bootstrap_preds).std(axis=0, ddof=1).mean(axis=1)


def gpr_std_from_payload(surrogate_payload: dict, x: pd.DataFrame) -> Optional[np.ndarray]:
    """Analytic GPR posterior std if the payload holds a (possibly pipelined) GP."""
    model = surrogate_payload.get("model")
    estimator = model
    scaler = None
    if isinstance(model, Pipeline):
        scaler = model.steps[0][1]
        estimator = model.steps[-1][1]
    if not isinstance(estimator, GaussianProcessRegressor):
        return None
    try:
        if scaler is not None:
            x_t = scaler.transform(x)
        else:
            x_t = x
        _, std = estimator.predict(x_t, return_std=True)
        return np.asarray(std, dtype=float).reshape(-1)
    except Exception:
        return None


def _fallback_sigma(surrogate_payload: dict) -> float:
    """Constant sigma from held-out residuals when bootstrap is impossible."""
    metrics = surrogate_payload.get("metrics", {})
    rmses = [m.get("rmse") for m in metrics.values() if m.get("rmse") is not None]
    return float(np.mean(rmses)) if rmses else 1.0


def predict_with_uncertainty(surrogate_payload: dict, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_prediction (n, k), std (n,)) for a trained surrogate."""
    from app import predict_surrogate  # local import to avoid a cycle

    mean = predict_surrogate(surrogate_payload, x)
    std = ensemble_std(surrogate_payload, x)
    return mean, std


# ---------------------------------------------------------------------------
# 2. Adaptive sampling
# ---------------------------------------------------------------------------


def suggest_next_simulations(
    surrogate_payload: dict,
    bounds: Sequence[tuple],
    directions: Sequence[str],
    n_candidates: int = 2000,
    n_suggest: int = 5,
    exploration_weight: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Rank random candidate designs by information gain for the next
    simulator campaign.

    Acquisition = normalised prediction spread (uncertainty) + normalised
    min-distance to known training points (space-filling). The suggested
    points are where the surrogate is most uncertain AND least supported:
    running the true simulation there improves the surrogate fastest.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    X_cand = lb + rng.random((n_candidates, d)) * (ub - lb)
    feature_names = surrogate_payload["feature_names"]
    cand_df = pd.DataFrame(X_cand, columns=feature_names)

    from app import predict_surrogate

    mean = predict_surrogate(surrogate_payload, cand_df)
    sigma = ensemble_std(surrogate_payload, cand_df)

    # Distance to known training data (space-filling term)
    x_train = surrogate_payload.get("x_train")
    if x_train is not None and len(x_train) > 0:
        span = np.where((ub - lb) > 0, ub - lb, 1.0)
        known = (np.asarray(x_train[feature_names], dtype=float) - lb) / span
        cand_n = (X_cand - lb) / span
        # chunked min-distance to keep memory bounded
        dists = np.empty(n_candidates)
        for i in range(n_candidates):
            dists[i] = np.min(np.linalg.norm(known - cand_n[i], axis=1))
        dist_norm = dists / (dists.max() if dists.max() > 0 else 1.0)
    else:
        dist_norm = np.ones(n_candidates)

    sig_norm = sigma / (sigma.max() if sigma.max() > 0 else 1.0)
    score = exploration_weight * sig_norm + (1.0 - exploration_weight) * dist_norm

    order = np.argsort(-score)[:n_suggest]
    out = cand_df.iloc[order].copy()
    out.insert(0, "acquisition_score", score[order])
    out.insert(1, "predicted_uncertainty", sigma[order])
    for t_i, t_name in enumerate(surrogate_payload["target_names"]):
        out[f"pred_{t_name}"] = mean[order, t_i]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Inverse design
# ---------------------------------------------------------------------------


def inverse_design(
    surrogate_payload: dict,
    predict_fn,
    bounds: Sequence[tuple],
    targets: dict,
    seed: int = 42,
    pop_size: int = 60,
    n_gen: int = 60,
) -> OptimizationResult:
    """Search surrogate input space for designs matching desired outputs.

    `targets` maps target name -> desired value (any subset of the
    surrogate's targets). The objective is the (relative) squared matching
    error; the returned result's F is the per-target relative error, so the
    "Pareto front" of the inverse problem is the set of designs that trade
    off matching accuracy across targets.

    `predict_fn(X: np.ndarray (n, d)) -> np.ndarray (n, k)` must return raw
    surrogate predictions; it is passed separately so app.py can reuse its
    own `predict_surrogate` wrapper.
    """
    feature_names = surrogate_payload["feature_names"]
    target_names = list(surrogate_payload["target_names"])
    tgt_vec = np.array([float(targets[name]) for name in target_names if name in targets], dtype=float)

    missing = [name for name in targets if name not in target_names]
    if missing:
        raise ValueError(f"Unknown target(s) for this surrogate: {missing}")

    span = np.abs(tgt_vec)
    span[span == 0] = 1.0

    def evaluate(X):
        preds = np.atleast_2d(np.asarray(predict_fn(X), dtype=float))
        err = (preds - tgt_vec) / span  # relative error per target
        return np.sqrt((err ** 2).mean(axis=1)).reshape(-1, 1)  # (n, 1) batch contract

    result = run_genetic_algorithm(
        evaluate,
        bounds,
        directions=["min"],
        n_weight_sets=1,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
    )
    return result


# ---------------------------------------------------------------------------
# 4. Decision making on a Pareto front
# ---------------------------------------------------------------------------


def _minmax(F: np.ndarray) -> np.ndarray:
    Fmin, Fmax = F.min(axis=0), F.max(axis=0)
    span = np.where(Fmax - Fmin > 0, Fmax - Fmin, 1.0)
    return (F - Fmin) / span


@dataclass
class RankedDesign:
    index: int
    score: float
    X: dict
    F: dict


def pareto_knee(F: np.ndarray) -> int:
    """Index of the knee point of a (min-space) front: the point with the
    maximum perpendicular distance from the chord joining the per-objective
    optima ("bend of the Pareto front"). Deterministic; F is (n, k)."""
    Fn = _minmax(F)
    if len(Fn) == 1:
        return 0
    # ideal = per-objective best (0 in normalised min space)
    ideal = Fn.min(axis=0)
    if Fn.shape[1] == 2:
        # classic 2-D knee: max perpendicular distance to the chord joining
        # the two axis-optimal extremes (explicit 2-D formula — np.cross on
        # 2-element vectors is deprecated/removed in NumPy 2.x)
        b = int(Fn[:, 0].argmin())
        e = int(Fn[:, 1].argmin())
        if b == e:
            return int(np.argmin(np.linalg.norm(Fn - ideal, axis=1)))
        p1, p2 = Fn[b], Fn[e]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        denom = float(np.hypot(dx, dy))
        if denom == 0:
            return int(np.argmin(np.linalg.norm(Fn - ideal, axis=1)))
        dist = np.abs(dy * (Fn[:, 0] - p1[0]) - dx * (Fn[:, 1] - p1[1])) / denom
        return int(dist.argmax())
    # k > 2: fall back to the point closest to the ideal point
    return int(np.argmin(np.linalg.norm(Fn - ideal, axis=1)))


def topsis_rank(
    F: np.ndarray,
    directions: Sequence[str],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """TOPSIS (Hwang & Yoon, 1981) closeness scores for each front member.

    Returns a score in [0, 1]; higher = better (closer to the ideal point,
    farther from the anti-ideal). Handles mixed min/max directions.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n, k = F.shape
    if weights is None:
        weights = np.ones(k) / k
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    Fmin, Fmax = F.min(axis=0), F.max(axis=0)
    span = np.where(Fmax - Fmin > 0, Fmax - Fmin, 1.0)
    Fn = (F - Fmin) / span

    # flip maximisation objectives so bigger-is-better becomes smaller-is-better
    signs = _objective_signs(directions)[:k]
    Fn_min_space = Fn * signs

    # vector normalisation (column Euclidean), then weighting
    norm = np.linalg.norm(Fn_min_space, axis=0)
    norm[norm == 0] = 1.0
    V = (Fn_min_space / norm) * weights

    ideal = V.min(axis=0)   # min-space: column minimum is the ideal value
    anti = V.max(axis=0)    # column maximum is the anti-ideal

    d_plus = np.linalg.norm(V - ideal, axis=1)
    d_minus = np.linalg.norm(V - anti, axis=1)
    closeness = d_minus / (d_plus + d_minus + 1e-12)
    return closeness


def recommend_designs(
    pareto_X: pd.DataFrame,
    pareto_F: pd.DataFrame,
    directions: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    n_top: int = 3,
) -> list[RankedDesign]:
    """Rank Pareto designs by TOPSIS and return the top-n as RankedDesigns."""
    F = pareto_F.to_numpy(dtype=float)
    closeness = topsis_rank(F, directions, weights)
    order = np.argsort(-closeness)[:n_top]
    feature_names = list(pareto_X.columns)
    target_names = list(pareto_F.columns)
    ranked = []
    for rank, idx in enumerate(order, start=1):
        ranked.append(RankedDesign(
            index=int(idx),
            score=float(closeness[idx]),
            X={name: float(pareto_X.iloc[idx][name]) for name in feature_names},
            F={name: float(pareto_F.iloc[idx][name]) for name in target_names},
        ))
    return ranked


__all__ = [
    "ensemble_std",
    "gpr_std_from_payload",
    "predict_with_uncertainty",
    "suggest_next_simulations",
    "inverse_design",
    "pareto_knee",
    "topsis_rank",
    "recommend_designs",
    "RankedDesign",
]
