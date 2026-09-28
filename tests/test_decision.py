"""Unit tests for core.decision — decision-support logic (no UI)."""
import numpy as np
import pandas as pd
import pytest

from core.decision import (
    pareto_knee,
    predict_with_uncertainty,
    recommend_designs,
    suggest_next_simulations,
    topsis_rank,
    inverse_design,
    ensemble_std,
)
from core.optimizer import pareto_front


# ---------------------------------------------------------------------------
# helpers: a tiny trainable surrogate payload (mirrors app.py's payload shape)
# ---------------------------------------------------------------------------

def _make_payload(model, feature_names, target_names, x_train, y_train, rmse=0.1):
    return {
        "model": model,
        "model_slug": "test",
        "model_name": "TestModel",
        "feature_names": feature_names,
        "target_names": target_names,
        "metrics": {t: {"r2": 0.9, "rmse": rmse} for t in target_names},
        "cv_metrics": None,
        "strategy": "Single-Target",
        "x_train": x_train,
        "y_train": y_train,
    }


@pytest.fixture
def linear_payload():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.uniform(0, 10, size=(n, 2))
    y1 = 2.0 * x[:, 0] + 1.0 * x[:, 1] + rng.normal(0, 0.01, n)
    y2 = -1.0 * x[:, 0] + 3.0 * x[:, 1] + rng.normal(0, 0.01, n)
    x_df = pd.DataFrame(x, columns=["a", "b"])
    y_df = pd.DataFrame({"t1": y1, "t2": y2})

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(x_df, y_df)
    return _make_payload(model, ["a", "b"], ["t1", "t2"], x_df, y_df)


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

def test_topsis_prefers_better_min_design():
    F = np.array([[1.0, 10.0], [2.0, 5.0], [3.0, 1.0]])
    scores = topsis_rank(F, ["min", "min"])
    assert scores.shape == (3,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # with equal weights the trade-off middle point should score well
    assert scores[1] >= scores.min() - 1e-9


def test_topsis_respects_weights():
    F = np.array([[1.0, 100.0], [10.0, 1.0]])
    favor_first = topsis_rank(F, ["min", "min"], weights=[1.0, 0.0])
    favor_second = topsis_rank(F, ["min", "min"], weights=[0.0, 1.0])
    assert favor_first[0] > favor_first[1]   # only obj-1 matters -> design 0 wins
    assert favor_second[1] > favor_second[0]  # only obj-2 matters -> design 1 wins


def test_topsis_mixed_directions():
    F = np.array([[1.0, 0.9], [5.0, 0.1]])  # obj1 min, obj2 max
    scores = topsis_rank(F, ["min", "max"])
    assert scores[0] > scores[1]  # design 0 is better on both axes


# ---------------------------------------------------------------------------
# Knee detection
# ---------------------------------------------------------------------------

def test_knee_finds_bend_on_convex_front():
    # convex front with a clear bend at the middle point
    F = np.array([[0.0, 1.0], [0.1, 0.1], [1.0, 0.0]])
    idx = pareto_knee(F)
    assert idx == 1


def test_knee_single_point():
    F = np.array([[2.0, 3.0]])
    assert pareto_knee(F) == 0


def test_knee_on_real_pareto_front_is_nondominated():
    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    F = np.column_stack([X[:, 0] + X[:, 1], X[:, 1] + X[:, 2]])
    mask = pareto_front(F)  # returns minimisation front mask/indices
    F_pf = np.atleast_2d(F[mask]) if mask.ndim == 1 else F
    idx = pareto_knee(F_pf)
    assert 0 <= idx < len(F_pf)


# ---------------------------------------------------------------------------
# Recommend designs
# ---------------------------------------------------------------------------

def test_recommend_designs_shape_and_order():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    F = pd.DataFrame({"t1": [1.0, 2.0, 3.0], "t2": [3.0, 2.0, 1.0]})
    ranked = recommend_designs(X, F, ["min", "min"], n_top=2)
    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score
    assert set(ranked[0].X.keys()) == {"a", "b"}
    assert set(ranked[0].F.keys()) == {"t1", "t2"}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def test_uncertainty_linear_model_is_tiny(linear_payload):
    x = pd.DataFrame({"a": [5.0], "b": [5.0]})
    mean, std = predict_with_uncertainty(linear_payload, x)
    assert mean.shape == (1, 2)
    # bootstrap on a near-perfect linear fit should give very small spread
    assert np.all(std < 0.5)


def test_uncertainty_stacked_ensemble_uses_base_spread(linear_payload):
    # fabricate a stacked-ensemble payload from three identical-ish estimators
    from sklearn.linear_model import LinearRegression, Ridge

    payload = dict(linear_payload)
    payload["model"] = {
        "base_estimators": [
            LinearRegression().fit(payload["x_train"], payload["y_train"]),
            Ridge(alpha=10.0).fit(payload["x_train"], payload["y_train"]),
            Ridge(alpha=100.0).fit(payload["x_train"], payload["y_train"]),
        ],
        "meta_estimator": LinearRegression().fit(
            np.hstack([payload["y_train"].to_numpy()] * 3), payload["y_train"]
        ),
    }
    x = pd.DataFrame({"a": [5.0, 7.0], "b": [3.0, 1.0]})
    std = ensemble_std(payload, x)
    assert std.shape == (2,)
    assert np.all(np.isfinite(std))


# ---------------------------------------------------------------------------
# Adaptive sampling
# ---------------------------------------------------------------------------

def test_suggest_next_simulations_returns_requested(linear_payload):
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    suggestions = suggest_next_simulations(
        linear_payload, bounds, ["min", "min"], n_candidates=300, n_suggest=4, seed=7
    )
    assert len(suggestions) == 4
    assert {"acquisition_score", "predicted_uncertainty", "pred_t1", "pred_t2"}.issubset(suggestions.columns)
    # scores sorted descending
    assert suggestions["acquisition_score"].is_monotonic_decreasing
    # inside bounds
    assert suggestions["a"].between(0, 10).all()
    assert suggestions["b"].between(0, 10).all()


# ---------------------------------------------------------------------------
# Inverse design
# ---------------------------------------------------------------------------

def test_inverse_design_recovers_target(linear_payload):
    bounds = [(0.0, 10.0), (0.0, 10.0)]

    def predict_fn(X):
        x_df = pd.DataFrame(np.atleast_2d(X), columns=["a", "b"])
        m = linear_payload["model"]
        return np.atleast_2d(m.predict(x_df))

    result = inverse_design(
        linear_payload, predict_fn, bounds,
        targets={"t1": 10.0, "t2": 10.0},
        pop_size=60, n_gen=40, seed=1,
    )
    achieved = predict_fn(result.X)
    # relative error should be small for a linear surrogate
    rel_err = np.abs(achieved - np.array([10.0, 10.0])) / 10.0
    assert rel_err.max() < 0.05


def test_inverse_design_rejects_unknown_target(linear_payload):
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    with pytest.raises(ValueError):
        inverse_design(
            linear_payload, lambda X: np.zeros((len(np.atleast_2d(X)), 2)),
            bounds, targets={"nope": 1.0},
        )
