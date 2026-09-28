"""UI smoke test via Streamlit's AppTest: STEP 4 must render its five tabs
(Design sliders, Inverse design, Uncertainty, Suggest simulations, Decision
helper) without exceptions when a trained surrogate + dataset are present."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _demo_state():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.uniform(0, 10, size=(n, 2))
    df = pd.DataFrame(x, columns=["a", "b"])
    df["t1"] = 2.0 * df["a"] + df["b"] + rng.normal(0, 0.01, n)
    df["t2"] = -df["a"] + 3.0 * df["b"] + rng.normal(0, 0.01, n)

    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(df[["a", "b"]], df[["t1", "t2"]])
    payload = {
        "model_slug": "linear",
        "model_name": "Linear Regression",
        "model": model,
        "feature_names": ["a", "b"],
        "target_names": ["t1", "t2"],
        "metrics": {"t1": {"r2": 0.99, "rmse": 0.01}, "t2": {"r2": 0.99, "rmse": 0.01}},
        "cv_metrics": None,
        "strategy": "Single-Target",
        "x_train": df[["a", "b"]],
        "y_train": df[["t1", "t2"]],
    }
    return df, [payload]


def test_step4_renders_all_tabs_without_exception():
    df, surrogates = _demo_state()
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["uploaded_df"] = df
    at.session_state["trained_surrogates"] = surrogates
    at.session_state["current_step"] = "STEP 4 — CONTROL ROOM"
    at.run()

    assert not at.exception, f"STEP 4 raised: {at.exception}"
    # all five tab labels are present in the rendered DOM
    joined = " ".join(str(t.label) for t in at.tabs)
    for expected in ("Design sliders", "Inverse design", "Uncertainty", "Suggest simulations", "Decision helper"):
        assert expected in joined, f"missing tab: {expected} (got {joined})"


def test_step4_decision_helper_needs_front():
    df, surrogates = _demo_state()
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.session_state["uploaded_df"] = df
    at.session_state["trained_surrogates"] = surrogates
    at.session_state["current_step"] = "STEP 4 — CONTROL ROOM"
    at.run()
    assert not at.exception
    # without optimization_results the decision tab shows an info message
    infos = " ".join(str(i.value) for i in at.info)
    assert "STEP 3" in infos or "Pareto" in infos or "optimization" in infos.lower()


def test_step4_inverse_search_produces_results():
    df, surrogates = _demo_state()
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["uploaded_df"] = df
    at.session_state["trained_surrogates"] = surrogates
    at.session_state["current_step"] = "STEP 4 — CONTROL ROOM"
    at.run()

    tabs = at.tabs
    if not tabs:
        pytest.skip("tabs API unavailable in this Streamlit version")
    # set target inputs inside the inverse-design tab and click Search
    inv_tab = tabs[1]
    numbers = inv_tab.number_input
    assert len(numbers) >= 2
    numbers[0].set_value(10.0).run()
    numbers[1].set_value(10.0).run()
    buttons = inv_tab.button
    search_btn = [b for b in buttons if "Search" in (b.label or "")]
    assert search_btn, "inverse-design search button missing"
    search_btn[0].click().run()
    assert not at.exception, f"inverse search raised: {at.exception}"
    # results table exists in session state
    assert "inverse_design_results" in at.session_state
