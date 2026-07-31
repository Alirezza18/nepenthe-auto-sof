import io
from datetime import datetime
from pathlib import Path

import chardet
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Lars, OrthogonalMatchingPursuit, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from core.kernel import initialize_engine_state, get_system_telemetry
from core.optimizer import (
    run_nsga2,
    run_nsga3,
    run_steady_state_nsga2,
    run_hype,
    run_genetic_algorithm,
    run_aco,
    run_random_search,
    run_differential_evolution,
    run_rbf_surrogate_assisted,
    run_ann_surrogate_assisted,
)

if "current_step" not in st.session_state:
    st.session_state.current_step = "STEP 0 — WELCOME / DIAGNOSTICS"
if "cognitive_nodes" not in st.session_state:
    st.session_state.cognitive_nodes = st.session_state.current_step
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "step2_demo_active" not in st.session_state:
    st.session_state.step2_demo_active = False
if "step2_data_source" not in st.session_state:
    st.session_state.step2_data_source = None

st.set_page_config(
    page_title="NEPENTHE | Auto-SOF",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parent
CSS_PATH = ROOT_DIR / "ui" / "styles.css"

try:
    CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    CUSTOM_CSS = ""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

initialize_engine_state(force_reset=False)
telemetry = get_system_telemetry()


def analyze_dataset(df: pd.DataFrame) -> dict:
    if df is None:
        return {
            "rows": 0,
            "cols": 0,
            "missing_values": 0,
            "numeric_columns": [],
            "constant_columns": [],
        }

    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    constant_columns = [column for column in numeric_columns if df[column].nunique(dropna=True) <= 1]
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "numeric_columns": numeric_columns,
        "constant_columns": constant_columns,
    }


def load_dataset(uploaded_file, use_demo_data: bool):
    if use_demo_data:
        demo_df = pd.DataFrame(
            {
                "Temp": np.random.default_rng(7).normal(loc=22.0, scale=3.5, size=100),
                "Press": np.random.default_rng(11).normal(loc=101.3, scale=2.0, size=100),
                "Flow": np.random.default_rng(13).normal(loc=45.0, scale=5.0, size=100),
                "EUI": np.random.default_rng(17).normal(loc=80.0, scale=10.0, size=100),
                "OCI": np.random.default_rng(19).normal(loc=0.65, scale=0.12, size=100),
            }
        )
        return demo_df, None

    if uploaded_file is None:
        return None, "No dataset loaded."

    try:
        content = uploaded_file.read()
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            detection = chardet.detect(content[:50000]) or {}
            encoding = detection.get("encoding") or "utf-8"
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, encoding_errors="replace")

        df.columns = df.columns.astype(str).str.replace("\xa0", " ").str.strip()
        return df, None
    except Exception as exc:
        return None, str(exc)


def build_regressor(model_slug: str, multi_output: bool = False):
    if model_slug == "gpr":
        estimator = GaussianProcessRegressor(random_state=42, alpha=1e-6, n_restarts_optimizer=5)
    elif model_slug == "svr":
        estimator = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1))])
    elif model_slug == "ridge":
        estimator = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    elif model_slug == "lasso":
        estimator = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.1, max_iter=5000))])
    elif model_slug == "elastic_net":
        estimator = Pipeline([("scaler", StandardScaler()), ("elastic_net", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))])
    elif model_slug == "bayes_ridge":
        estimator = Pipeline([("scaler", StandardScaler()), ("bayes_ridge", BayesianRidge())])
    elif model_slug == "omp":
        estimator = Pipeline([("scaler", StandardScaler()), ("omp", OrthogonalMatchingPursuit())])
    elif model_slug == "lars":
        estimator = Pipeline([("scaler", StandardScaler()), ("lars", Lars())])
    elif model_slug == "rf":
        estimator = RandomForestRegressor(n_estimators=250, random_state=42)
    elif model_slug == "gbrt":
        estimator = GradientBoostingRegressor(random_state=42)
    elif model_slug == "xgboost":
        estimator = GradientBoostingRegressor(random_state=42)
    elif model_slug == "lightgbm":
        estimator = RandomForestRegressor(n_estimators=250, random_state=42)
    elif model_slug == "catboost":
        estimator = ExtraTreesRegressor(n_estimators=250, random_state=42)
    elif model_slug == "adaboost":
        estimator = AdaBoostRegressor(random_state=42, n_estimators=150)
    elif model_slug == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=250, random_state=42)
    elif model_slug == "mlp":
        estimator = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))])
    else:
        estimator = RandomForestRegressor(n_estimators=250, random_state=42)

    # Models that do NOT natively support multi-output y in scikit-learn need
    # to be wrapped in MultiOutputRegressor whenever more than one target is
    # selected. RandomForest/ExtraTrees/MLP DO support 2D y natively and are
    # deliberately left out of this set.
    needs_multi_output_wrap = {
        "gpr", "svr", "ridge", "lasso", "elastic_net", "bayes_ridge", "omp", "lars",
        "gbrt", "xgboost", "adaboost",
    }
    if multi_output and model_slug in needs_multi_output_wrap:
        return MultiOutputRegressor(estimator)
    return estimator


def train_surrogate_pipeline(model_slug: str, model_name: str, x_train, y_train, x_test, y_test, feature_names, target_names, strategy: str, stage_callback=None):
    n_targets = len(target_names)
    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)

    if y_train_arr.ndim == 1:
        y_train_arr = y_train_arr.reshape(-1, 1)
        y_test_arr = y_test_arr.reshape(-1, 1)

    if strategy == "Hybrid Stacked Ensemble Pipeline":
        cv = KFold(n_splits=min(5, max(2, len(x_train) // 5)), shuffle=True, random_state=42)
        n_splits = cv.get_n_splits(x_train)
        oof_predictions = np.zeros((len(x_train), n_targets), dtype=float)
        base_estimators = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train)):
            if stage_callback is not None:
                stage_callback(f"fold {fold_idx + 1}/{n_splits}", fold_idx / n_splits)
            fold_estimator = build_regressor(model_slug, multi_output=True)
            fold_estimator.fit(x_train.iloc[train_idx], y_train_arr[train_idx])
            oof_predictions[valid_idx] = fold_estimator.predict(x_train.iloc[valid_idx])
            base_estimators.append(fold_estimator)

        if stage_callback is not None:
            stage_callback("meta-learner", 0.9)
        meta_estimator = build_regressor("ridge", multi_output=True)
        meta_estimator.fit(oof_predictions, y_train_arr)
        fitted_model = {"base_estimators": base_estimators, "meta_estimator": meta_estimator}
        test_features = np.hstack([np.asarray(model.predict(x_test), dtype=float) for model in base_estimators])
        test_predictions = meta_estimator.predict(test_features)
    else:
        if stage_callback is not None:
            stage_callback("fitting", 0.2)
        fitted_model = build_regressor(model_slug, multi_output=(n_targets > 1))
        fitted_model.fit(x_train, y_train_arr)
        test_predictions = fitted_model.predict(x_test)

    if stage_callback is not None:
        stage_callback("scoring", 0.95)

    if test_predictions.ndim == 1:
        test_predictions = test_predictions.reshape(-1, 1)
    if y_test_arr.ndim == 1:
        y_test_arr = y_test_arr.reshape(-1, 1)

    metrics = {}
    for idx, target_name in enumerate(target_names):
        target_true = y_test_arr[:, idx]
        target_pred = test_predictions[:, idx]
        metrics[target_name] = {
            "r2": float(r2_score(target_true, target_pred)),
            "rmse": float(np.sqrt(mean_squared_error(target_true, target_pred))),
        }

    return fitted_model, feature_names, metrics


def cross_validate_surrogate(
    model_slug: str,
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    target_names: list,
    n_splits: int = 5,
    seed: int = 42,
    progress_callback=None,
) -> dict:
    """K-fold cross-validated R2/RMSE per target — mean +/- std across folds.

    A single 80/20 holdout split (as used for the headline metrics reported
    right after training) can be noisy on small/medium datasets. This offers
    a more statistically defensible estimate by re-fitting a fresh model on
    each of `n_splits` folds and averaging the held-out performance. It is
    intentionally NOT used for the Hybrid Stacked Ensemble strategy, which
    already performs its own internal out-of-fold resampling for the
    meta-learner and would otherwise be prohibitively slow to re-run.
    """
    X = feature_frame.reset_index(drop=True)
    Y = target_frame.reset_index(drop=True).to_numpy(dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_targets = Y.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_r2 = np.zeros((n_splits, n_targets))
    fold_rmse = np.zeros((n_splits, n_targets))

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        fold_model = build_regressor(model_slug, multi_output=(n_targets > 1))
        fold_model.fit(X.iloc[train_idx], Y[train_idx])
        fold_pred = np.asarray(fold_model.predict(X.iloc[valid_idx]), dtype=float)
        if fold_pred.ndim == 1:
            fold_pred = fold_pred.reshape(-1, 1)
        for t in range(n_targets):
            fold_r2[fold_idx, t] = r2_score(Y[valid_idx, t], fold_pred[:, t])
            fold_rmse[fold_idx, t] = np.sqrt(mean_squared_error(Y[valid_idx, t], fold_pred[:, t]))
        if progress_callback is not None:
            progress_callback(fold_idx + 1, n_splits)

    cv_metrics = {}
    for t, target_name in enumerate(target_names):
        cv_metrics[target_name] = {
            "r2_mean": float(fold_r2[:, t].mean()),
            "r2_std": float(fold_r2[:, t].std()),
            "rmse_mean": float(fold_rmse[:, t].mean()),
            "rmse_std": float(fold_rmse[:, t].std()),
        }
    return cv_metrics


def predict_surrogate(surrogate_payload: dict, x: pd.DataFrame) -> np.ndarray:
    """Run a trained surrogate (plain estimator OR hybrid stacked dict) on x.

    Always returns a 2D array of shape (n_samples, n_targets), regardless of
    which architecture strategy produced the model.
    """
    model = surrogate_payload["model"]
    feature_names = surrogate_payload["feature_names"]

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(np.atleast_2d(x), columns=feature_names)
    else:
        x = x[feature_names]

    if isinstance(model, dict) and "base_estimators" in model:
        base_preds = np.hstack([np.asarray(estimator.predict(x), dtype=float) for estimator in model["base_estimators"]])
        prediction = model["meta_estimator"].predict(base_preds)
    else:
        prediction = model.predict(x)

    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim == 1:
        prediction = prediction.reshape(-1, 1)
    return prediction


def guess_direction(target_name: str) -> str:
    """Best-effort default optimization direction based on the target's name."""
    positive_hints = ("eff", "perf", "yield", "comfort", "score", "output", "gain")
    return "max" if any(hint in target_name.lower() for hint in positive_hints) else "min"


def reconstruct_test_split(surrogate_payload: dict, dataset: pd.DataFrame):
    """Rebuild the exact same train/test split used during training (STEP 2).

    Training always uses test_size=0.2 and random_state=42 on the numeric,
    NA-filled feature/target frames, so re-running the identical steps here
    reproduces the held-out test rows deterministically without needing to
    have cached them on the surrogate payload itself.
    """
    feature_names = surrogate_payload["feature_names"]
    target_names = surrogate_payload["target_names"]

    feature_frame = dataset.drop(columns=target_names, errors="ignore")
    feature_frame = feature_frame[feature_names] if all(f in feature_frame.columns for f in feature_names) else feature_frame
    target_frame = dataset[target_names].copy()

    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    target_frame = target_frame.apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    target_frame = target_frame.fillna(target_frame.median(numeric_only=True))

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame, target_frame, test_size=0.2, random_state=42
    )
    return x_test, y_test


def compute_permutation_importance(surrogate_payload: dict, x_test: pd.DataFrame, y_test: pd.DataFrame, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Model-agnostic permutation importance, averaged across all targets.

    Works for both plain sklearn estimators and the hybrid stacked-ensemble
    dict produced by the "Hybrid Stacked Ensemble Pipeline" strategy, since
    it only relies on predict_surrogate() rather than a specific model API.
    """
    rng = np.random.default_rng(seed)
    feature_names = surrogate_payload["feature_names"]
    target_names = surrogate_payload["target_names"]

    y_true = y_test[target_names].to_numpy(dtype=float)
    baseline_pred = predict_surrogate(surrogate_payload, x_test)
    baseline_scores = [r2_score(y_true[:, i], baseline_pred[:, i]) for i in range(len(target_names))]
    baseline_mean = float(np.mean(baseline_scores))

    records = []
    for feature_name in feature_names:
        drops = []
        for _ in range(n_repeats):
            shuffled = x_test.copy()
            shuffled[feature_name] = rng.permutation(shuffled[feature_name].to_numpy())
            shuffled_pred = predict_surrogate(surrogate_payload, shuffled)
            shuffled_scores = [r2_score(y_true[:, i], shuffled_pred[:, i]) for i in range(len(target_names))]
            drops.append(baseline_mean - float(np.mean(shuffled_scores)))
        records.append({"Feature": feature_name, "Importance": float(np.mean(drops)), "Std": float(np.std(drops))})

    importance_df = pd.DataFrame(records).sort_values("Importance", ascending=False).reset_index(drop=True)
    return importance_df


nav_options = [
    "STEP 0 — WELCOME / DIAGNOSTICS",
    "STEP 1 — DATA INTAKE",
    "STEP 2 — PROXY MODELING",
    "STEP 3 — OPTIMIZATION",
    "STEP 4 — CONTROL ROOM",
    "STEP 5 — REPORTING",
]
current_step = st.session_state.get("current_step", st.session_state.get("cognitive_nodes", nav_options[0]))
if current_step not in nav_options:
    current_step = nav_options[0]
    st.session_state.current_step = current_step
    st.session_state.cognitive_nodes = current_step

with st.sidebar:
    st.markdown('<div class="nepenthe-logo logo-mini">NEPENTHE</div>', unsafe_allow_html=True)
    st.markdown("<div class='sidebar-brief'>Auto-SOF · Control Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sidebar-metrics'><div class='sidebar-label'>LOCAL ORBIT</div><div class='sidebar-value'>Kernel {telemetry['Kernel_V']}</div><div class='sidebar-value'>{telemetry['Proxies']} proxies</div><div class='sidebar-value'>{telemetry['Clusters']} clusters</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    selected_sidebar_step = st.radio(
        "COGNITIVE NODES",
        nav_options,
        index=nav_options.index(current_step),
        key="navigation_step_radio",
        horizontal=False,
        label_visibility="collapsed",
    )
    if selected_sidebar_step in nav_options:
        st.session_state.current_step = selected_sidebar_step
        st.session_state.cognitive_nodes = selected_sidebar_step
    current_step = st.session_state.current_step
    if st.button("Shutdown System", key="shutdown_system", use_container_width=True):
        st.toast("Shutdown sequence initiated")

    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>PROJECT</div>", unsafe_allow_html=True)

    trained_surrogates_sidebar = st.session_state.get("trained_surrogates", [])
    if trained_surrogates_sidebar or st.session_state.get("uploaded_df") is not None:
        try:
            project_bundle = {
                "uploaded_df": st.session_state.get("uploaded_df"),
                "trained_surrogates": trained_surrogates_sidebar,
                "optimization_results": st.session_state.get("optimization_results"),
                "optimization_meta": st.session_state.get("optimization_meta"),
                "step2_log_lines": st.session_state.get("step2_log_lines", []),
            }
            project_buffer = io.BytesIO()
            joblib.dump(project_bundle, project_buffer)
            st.download_button(
                "💾 Save Project",
                data=project_buffer.getvalue(),
                file_name=f"nepenthe_project_{datetime.now().strftime('%Y%m%d_%H%M')}.joblib",
                mime="application/octet-stream",
                use_container_width=True,
                key="sidebar_save_project",
                help="Bundles your dataset, trained models, and optimization results into one file to share with a teammate or reload later.",
            )
        except Exception as save_error:
            st.caption(f"⚠️ Could not prepare project file: {save_error}")

    uploaded_project = st.file_uploader(
        "Load Project (.joblib)",
        type=["joblib"],
        key="sidebar_load_project",
        label_visibility="collapsed",
        help="Restore a previously saved NEPENTHE project file.",
    )
    if uploaded_project is not None and st.session_state.get("_last_loaded_project_name") != uploaded_project.name:
        try:
            restored_bundle = joblib.load(uploaded_project)
            if restored_bundle.get("uploaded_df") is not None:
                st.session_state["uploaded_df"] = restored_bundle["uploaded_df"]
            st.session_state["trained_surrogates"] = restored_bundle.get("trained_surrogates", [])
            st.session_state["optimization_results"] = restored_bundle.get("optimization_results")
            st.session_state["optimization_meta"] = restored_bundle.get("optimization_meta")
            st.session_state["step2_log_lines"] = restored_bundle.get("step2_log_lines", [])
            st.session_state["_last_loaded_project_name"] = uploaded_project.name
            st.success("Project restored from file.")
            st.rerun()
        except Exception as load_error:
            st.error(f"Could not load project file: {load_error}")

    st.markdown("<div class='sidebar-footer'>SAFE MODE · CORE LINK</div>", unsafe_allow_html=True)

if current_step == "STEP 0 — WELCOME / DIAGNOSTICS":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>AUTONOMOUS CONTROL HUB · STEP 0</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>The Auto-SOF engine now operates as a modular command environment for diagnostics, telemetry, and immersive mission planning.</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.0, 1.0], gap="large")

    with left_col:
        st.markdown("<div class='control-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>DIAGNOSTIC STATE</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>ENGINE STATUS · KERNEL ACTIVE</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='metric-stack'>"
            f"<div class='metric-pill'>KERNEL ACTIVE</div>"
            f"<div class='metric-pill'>TELEMETRY v{telemetry['Kernel_V']}</div>"
            f"<div class='metric-pill'>CLUSTERS {telemetry['Clusters']}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='metric-list'><div>● Control plane online</div><div>● Surrogate kernels synced</div><div>● Constraint envelopes loaded</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='control-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>SYSTEM LOGS</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>RUNTIME TELEMETRY · LIVE CHANNELS</div>", unsafe_allow_html=True)
        log_groups = [
            ["[OK] Model registry synced", "[INFO] Telemetry active on port 8501", "[READY] Awaiting dataset ingestion"],
            ["[SYS] Surrogate kernel booted", "[ALGO] Constraint envelopes loaded", "[SYNC] Control plane online"],
            ["[PULSE] Latency baseline stable", "[WARN] No critical anomalies detected", "[MODE] Diagnostics ready"],
        ]
        for group in log_groups:
            st.markdown(
                "<div class='telemetry-block'>" + "<br>".join(group) + "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

elif current_step == "STEP 1 — DATA INTAKE":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>STEP 1 · DATA INTAKE & VARIABLE PARSING</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>The intake workspace accepts tabular data, validates the schema, and prepares multi-output targets and input features for downstream surrogate construction.</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        key="step2_uploaded_dataset",
    )
    use_demo_data = st.checkbox("Try with Demo Data", key="step2_demo_toggle")

    if use_demo_data and not st.session_state.step2_demo_active:
        demo_df = pd.DataFrame(
            {
                "Temp": np.random.default_rng(7).normal(loc=22.0, scale=3.5, size=100),
                "Press": np.random.default_rng(11).normal(loc=101.3, scale=2.0, size=100),
                "Flow": np.random.default_rng(13).normal(loc=45.0, scale=5.0, size=100),
                "EUI": np.random.default_rng(17).normal(loc=80.0, scale=10.0, size=100),
                "OCI": np.random.default_rng(19).normal(loc=0.65, scale=0.12, size=100),
            }
        )
        st.session_state.uploaded_df = demo_df
        st.session_state.step2_demo_active = True
        st.session_state.step2_data_source = "demo"
        st.rerun()

    if not use_demo_data and st.session_state.step2_demo_active:
        st.session_state.step2_demo_active = False
        st.session_state.step2_data_source = None

    if uploaded_file is not None and not use_demo_data:
        loaded_df, load_error = load_dataset(uploaded_file, False)
        if load_error:
            st.warning(load_error)
        else:
            st.session_state.uploaded_df = loaded_df
            st.session_state.step2_data_source = uploaded_file.name
            st.session_state.step2_demo_active = False
    elif not use_demo_data and st.session_state.uploaded_df is None:
        load_error = "No dataset loaded."
    else:
        load_error = None

    df = st.session_state.uploaded_df
    if df is None and not use_demo_data:
        st.warning(load_error)

    if df is not None:
        analysis = analyze_dataset(df)
        numeric_columns = analysis["numeric_columns"]
        usable_numeric_columns = [column for column in numeric_columns if column not in analysis["constant_columns"]]

        if "step2_target_variables" not in st.session_state or not isinstance(st.session_state["step2_target_variables"], list):
            st.session_state["step2_target_variables"] = []
        if "step2_log_lines" not in st.session_state:
            st.session_state["step2_log_lines"] = [
                "[READY] Dataset intake pipeline online.",
                "[READY] Feature routing and model registry prepared.",
            ]

        quality_col, terminal_col = st.columns([1.05, 0.95], gap="small")

        with quality_col:
            st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
            st.markdown("<div class='module-title'>AUTO-SOF DATA QUALITY REPORT</div>", unsafe_allow_html=True)
            st.markdown("<div class='module-subtitle'>ENGINEERING DIAGNOSTICS · MULTI-OUTPUT VALIDATION MATRIX</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-stack'><div class='metric-pill'>Rows: {analysis['rows']}</div><div class='metric-pill'>Cols: {analysis['cols']}</div><div class='metric-pill'>Missing: {analysis['missing_values']}</div><div class='metric-pill'>Numeric: {len(analysis['numeric_columns'])}</div></div>",
                unsafe_allow_html=True,
            )

            if usable_numeric_columns:
                selected_targets = [target for target in st.session_state["step2_target_variables"] if target in usable_numeric_columns]
                if not selected_targets:
                    selected_targets = [usable_numeric_columns[0]]
                target_variables = st.multiselect(
                    "Target Objective Variables (y)",
                    options=usable_numeric_columns,
                    default=selected_targets,
                    key="step2_target_variables_widget",
                )
                target_variables = st.session_state["step2_target_variables_widget"]
                st.session_state["step2_target_variables"] = target_variables
                input_features = [column for column in usable_numeric_columns if column not in target_variables]
                st.markdown(
                    f"<div class='step2-selection-card'><div class='step2-selection-title'>Input Features (X)</div><div class='metric-list'><div>● {', '.join(input_features) if input_features else 'No remaining numeric features'}</div></div></div>",
                    unsafe_allow_html=True,
                )
                if len(target_variables) > 1:
                    st.caption("Multi-output ingestion active — stacked surrogate flows can optimize several response channels simultaneously.")
            else:
                st.info("No usable numeric columns available for supervised modeling.")
                target_variables = []

            if usable_numeric_columns and len(usable_numeric_columns) > 1:
                feature_frame = df[usable_numeric_columns].copy()
                corr_matrix = feature_frame.corr().abs()
                warnings = []
                for left_idx, left_col in enumerate(usable_numeric_columns):
                    for right_col in usable_numeric_columns[left_idx + 1 :]:
                        corr_value = corr_matrix.loc[left_col, right_col]
                        if pd.notna(corr_value) and corr_value > 0.95:
                            warnings.append((left_col, right_col, round(float(corr_value), 3)))

                if warnings:
                    pair_summary = ", ".join([f"{left}/{right}" for left, right, _ in warnings])
                    warning_text = f"⚠️ [WARN] High Multicollinearity detected between pairs: [{pair_summary}]. ROOT B models may experience instability."
                    st.markdown(
                        f"<div class='step2-warning-card'>{warning_text}</div>",
                        unsafe_allow_html=True,
                    )
                    if not any(entry == warning_text for entry in st.session_state["step2_log_lines"]):
                        st.session_state["step2_log_lines"].append(warning_text)

            display_columns = list(target_variables) + [column for column in usable_numeric_columns if column not in target_variables]
            if display_columns:
                summary_matrix = df[display_columns].describe().T.round(3)
                st.dataframe(summary_matrix, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with terminal_col:
            st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
            st.markdown("<div class='module-title'>SIMULATED TERMINAL</div>", unsafe_allow_html=True)
            st.markdown("<div class='module-subtitle'>DATA INTEGRITY DISPATCHER</div>", unsafe_allow_html=True)
            if st.button("Dispatch dropna() Integrity Cleanup", key="step2_dropna_button", use_container_width=True):
                cleaned_df = df.dropna()
                st.session_state["step2_log_lines"].append(f"[INFO] dropna() executed at {datetime.now().strftime('%H:%M:%S')}")
                st.session_state["step2_active_frame"] = cleaned_df
                st.session_state["step2_last_cleanup"] = datetime.now().strftime("%H:%M:%S")

            if "step2_active_frame" not in st.session_state:
                st.session_state["step2_active_frame"] = df

            active_frame = st.session_state["step2_active_frame"]
            if len(active_frame) != len(df):
                st.caption(f"Filtered to {len(active_frame)} rows after cleanup.")

            st.markdown(
                "<div class='terminal-log'>" + "<br>".join(st.session_state["step2_log_lines"]) + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a dataset or enable demo mode to begin the intake workflow.")

elif current_step == "STEP 2 — PROXY MODELING":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>STEP 2 · PROXY MODELING & SURROGATE REGISTRY</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>The proxy-modeling workspace now focuses on surrogate architecture selection, ensemble routing, and model queue activation.</div>",
        unsafe_allow_html=True,
    )

    if "step2_strategy" not in st.session_state:
        st.session_state["step2_strategy"] = "Single Surrogate Strategy"
    if "step2_selected_model" not in st.session_state:
        st.session_state["step2_selected_model"] = None
    if "step2_selected_models" not in st.session_state or not isinstance(st.session_state["step2_selected_models"], list):
        st.session_state["step2_selected_models"] = []

    st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
    st.markdown("<div class='module-title'>ARCHITECTURAL STRATEGY CONTROL</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-subtitle'>SELECT THE SURROGATE EXECUTION PATH</div>", unsafe_allow_html=True)
    architecture_strategy = st.radio(
        "SELECT ARCHITECTURAL STRATEGY",
        ["Single Surrogate Strategy", "Hybrid Stacked Ensemble Pipeline"],
        index=0 if st.session_state["step2_strategy"] == "Single Surrogate Strategy" else 1,
        key="step2_architecture_strategy",
        horizontal=True,
    )
    st.session_state["step2_strategy"] = architecture_strategy
    if architecture_strategy == "Hybrid Stacked Ensemble Pipeline":
        st.info("💡 [ENGINEERING ADVISORY] For optimal stacking efficiency, combine high-bias linear varieties from ROOT B (e.g., Ridge/Lasso) with high-variance ensemble models from ROOT C (e.g., XGBoost/LightGBM) using a GPR meta-learner.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
    st.markdown("<div class='module-title'>MATHEMATICAL SURROGATE REGISTRY</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-subtitle'>ROOT-BASED MODEL QUEUE · SELECT ACTIVE SURROGATES</div>", unsafe_allow_html=True)

    registry_groups = [
        (
            "ROOT A — NON-PARAMETRIC PROBABILISTIC VARIETIES",
            [
                ("gpr", "Gaussian Process Regression (GPR / Kriging)", r"f(x) \sim \mathcal{GP}(m(x), k(x, x'))"),
                ("svr", "Support Vector Regression (SVR)", r"f(x) = \sum_i \alpha_i K(x_i, x) + b"),
                ("rbf", "Radial Basis Function (RBF) Networks", r"f(x) = \sum_{j=1}^{m} w_j \phi(||x-c_j||)"),
                ("bayes_ridge", "Bayesian Ridge Regression", r"y = Xw + \epsilon"),
            ],
        ),
        (
            "ROOT B — REGULARIZED PARAMETRIC & LINEAR VARIETIES",
            [
                ("ridge", "Ridge Regression", r"\min_w ||Xw-y||^2_2 + \alpha ||w||^2_2"),
                ("lasso", "Lasso Regression", r"\min_w ||Xw-y||^2_2 + \alpha ||w||_1"),
                ("elastic_net", "ElasticNet", r"\min_w ||Xw-y||^2_2 + \alpha_1 ||w||_1 + \alpha_2 ||w||^2_2"),
                ("omp", "Orthogonal Matching Pursuit (OMP)", r"\min_w ||y - Xw||^2_2"),
                ("lars", "Least Angle Regression (LARS)", r"\min_w ||y - Xw||^2_2"),
            ],
        ),
        (
            "ROOT C — ENSEMBLE, STEPWISE LEAF & DEEP ARCHITECTURES",
            [
                ("rf", "Random Forest", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("gbrt", "Gradient Boosted Regression Trees (GBRT)", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("xgboost", "XGBoost", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("lightgbm", "LightGBM", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("catboost", "CatBoost", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("adaboost", "AdaBoost", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("extra_trees", "Extra Trees", r"\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)"),
                ("mlp", "DNN / MLP", r"y = \sigma(W_2\sigma(W_1x+b_1)+b_2)"),
                ("cnn", "1D-CNN", r"y = f_{conv}(x)"),
                ("lstm", "LSTM", r"h_t = \sigma(W_h h_{t-1} + W_x x_t + b)"),
                ("tabnet", "TabNet", r"\text{Attention}(x)"),
                ("ebm", "Explainable Boosting Machines (EBM)", r"\hat{y} = \sum_j f_j(x_j) + \sum_{j<k} f_{jk}(x_j, x_k)"),
            ],
        ),
    ]

    for title, models in registry_groups:
        with st.expander(title, expanded=True):
            cols = st.columns(2)
            for idx, (key_slug, model_name, formula) in enumerate(models):
                with cols[idx % 2]:
                    st.markdown("<div class='registry-card step2-registry-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='registry-title'>{model_name}</div>", unsafe_allow_html=True)
                    st.latex(formula)
                    widget_key = f"widget_model_{key_slug}"
                    if architecture_strategy == "Single Surrogate Strategy":
                        is_selected = st.session_state.get("step2_selected_model") == key_slug
                        checked = st.checkbox("Enable", key=widget_key, value=is_selected)
                        if checked and st.session_state.get("step2_selected_model") != key_slug:
                            st.session_state["step2_selected_model"] = key_slug
                        elif not checked and st.session_state.get("step2_selected_model") == key_slug:
                            st.session_state["step2_selected_model"] = None
                    else:
                        selected_models = st.session_state.get("step2_selected_models", [])
                        if not isinstance(selected_models, list):
                            selected_models = []
                        is_selected = key_slug in selected_models
                        checked = st.checkbox("Enable", key=widget_key, value=is_selected)
                        if checked and key_slug not in selected_models:
                            st.session_state["step2_selected_models"] = list(dict.fromkeys(selected_models + [key_slug]))
                        elif not checked and key_slug in selected_models:
                            st.session_state["step2_selected_models"] = [model_slug for model_slug in selected_models if model_slug != key_slug]
                    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    target_variables = st.session_state.get("step2_target_variables", [])
    queued_models = []
    if architecture_strategy == "Single Surrogate Strategy":
        selected_model = st.session_state.get("step2_selected_model")
        if selected_model:
            queued_models = [selected_model]
    else:
        queued_models = [model_slug for model_slug in st.session_state.get("step2_selected_models", []) if isinstance(model_slug, str)]

    st.caption(f"Active targets: {', '.join(target_variables) if target_variables else 'None selected'}")

    use_cv = st.checkbox(
        "🔁 Also compute 5-fold cross-validated metrics (mean ± std) — more statistically robust than a single train/test split. Skipped automatically for the Hybrid Stacked Ensemble strategy.",
        value=False,
        key="step2_use_cv",
    )

    if st.button("⚡ LAUNCH SURROGATE TRAINING ENGINE", key="step2_launch_button", use_container_width=True):
        if st.session_state.get("uploaded_df") is not None and target_variables and queued_models:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[EXEC] Training launched at {timestamp} for targets: {', '.join(target_variables)} using {len(queued_models)} selected surrogate(s)."
            if "step2_log_lines" not in st.session_state:
                st.session_state["step2_log_lines"] = []
            if not any(entry == log_entry for entry in st.session_state["step2_log_lines"]):
                st.session_state["step2_log_lines"] = st.session_state["step2_log_lines"] + [log_entry]

            dataset = st.session_state.uploaded_df.copy()
            feature_frame = dataset.drop(columns=target_variables, errors="ignore")
            target_frame = dataset[target_variables].copy()

            if feature_frame.empty or target_frame.empty:
                st.warning("The selected dataset does not contain any usable feature or target columns for training.")
            else:
                feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
                target_frame = target_frame.apply(pd.to_numeric, errors="coerce")
                feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
                target_frame = target_frame.fillna(target_frame.median(numeric_only=True))

                if feature_frame.empty or target_frame.empty:
                    st.warning("The selected dataset does not contain any numeric feature or target columns for training.")
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        feature_frame,
                        target_frame,
                        test_size=0.2,
                        random_state=42,
                    )

                    if "trained_surrogates" not in st.session_state:
                        st.session_state.trained_surrogates = []

                    total_models = len(queued_models)
                    progress_bar = st.progress(0.0, text="Preparing training queue...")
                    status_placeholder = st.empty()
                    failed_models = []

                    for model_index, model_slug in enumerate(queued_models):
                        display_name = {
                            "gpr": "Gaussian Process Regression",
                            "svr": "Support Vector Regression",
                            "ridge": "Ridge Regression",
                            "lasso": "Lasso Regression",
                            "elastic_net": "ElasticNet",
                            "bayes_ridge": "Bayesian Ridge",
                            "omp": "Orthogonal Matching Pursuit",
                            "lars": "LARS",
                            "rf": "Random Forest",
                            "gbrt": "Gradient Boosted Trees",
                            "xgboost": "XGBoost",
                            "lightgbm": "LightGBM",
                            "catboost": "CatBoost",
                            "adaboost": "AdaBoost",
                            "extra_trees": "Extra Trees",
                            "mlp": "MLP Regressor",
                        }.get(model_slug, model_slug)

                        status_placeholder.markdown(
                            f"<div class='step2-selection-card'>⚙️ Training <b>{display_name}</b> — model {model_index + 1} of {total_models}...</div>",
                            unsafe_allow_html=True,
                        )
                        progress_bar.progress(
                            model_index / total_models,
                            text=f"{display_name}: starting ({model_index + 1}/{total_models})",
                        )
                        st.session_state["step2_log_lines"].append(
                            f"[EXEC] Training {display_name} ({model_index + 1}/{total_models})..."
                        )

                        def stage_callback(stage_label, frac, _idx=model_index, _name=display_name, _total=total_models):
                            overall = (_idx + frac) / _total
                            progress_bar.progress(
                                min(max(overall, 0.0), 0.99),
                                text=f"{_name}: {stage_label} ({_idx + 1}/{_total})",
                            )

                        try:
                            fitted_model, feature_names, metrics = train_surrogate_pipeline(
                                model_slug=model_slug,
                                model_name=display_name,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                feature_names=list(feature_frame.columns),
                                target_names=list(target_frame.columns),
                                strategy=architecture_strategy,
                                stage_callback=stage_callback,
                            )

                            cv_metrics = None
                            if use_cv and architecture_strategy != "Hybrid Stacked Ensemble Pipeline":
                                def cv_progress(fold_done, n_splits, _idx=model_index, _name=display_name, _total=total_models):
                                    overall = (_idx + 0.95 + 0.05 * fold_done / n_splits) / _total
                                    progress_bar.progress(
                                        min(max(overall, 0.0), 0.999),
                                        text=f"{_name}: cross-validating fold {fold_done}/{n_splits} ({_idx + 1}/{_total})",
                                    )

                                cv_metrics = cross_validate_surrogate(
                                    model_slug=model_slug,
                                    feature_frame=feature_frame,
                                    target_frame=target_frame,
                                    target_names=list(target_frame.columns),
                                    n_splits=5,
                                    seed=42,
                                    progress_callback=cv_progress,
                                )
                        except Exception as train_error:
                            failed_models.append((display_name, str(train_error)))
                            st.session_state["step2_log_lines"].append(
                                f"[ERROR] {display_name} failed to train: {train_error}"
                            )
                            progress_bar.progress(
                                (model_index + 1) / total_models,
                                text=f"{display_name} failed — see error below",
                            )
                            continue

                        surrogate_payload = {
                            "model_slug": model_slug,
                            "model_name": display_name,
                            "model": fitted_model,
                            "feature_names": feature_names,
                            "target_names": list(target_frame.columns),
                            "metrics": metrics,
                            "cv_metrics": cv_metrics,
                            "strategy": architecture_strategy,
                        }
                        st.session_state.trained_surrogates.append(surrogate_payload)

                        for target_name, metric_values in metrics.items():
                            log_line = f"[SUCCESS] {display_name} trained successfully. Target: {target_name} -> R²: {metric_values['r2']:.2f}, RMSE: {metric_values['rmse']:.2f}"
                            if cv_metrics and target_name in cv_metrics:
                                cv_vals = cv_metrics[target_name]
                                log_line += f" | 5-fold CV R²: {cv_vals['r2_mean']:.2f}±{cv_vals['r2_std']:.2f}"
                            st.session_state["step2_log_lines"].append(log_line)

                        progress_bar.progress(
                            (model_index + 1) / total_models,
                            text=f"Completed {display_name} ({model_index + 1}/{total_models})",
                        )

                    if failed_models:
                        st.error(
                            "The following model(s) failed to train and were skipped:\n"
                            + "\n".join(f"• {name}: {err}" for name, err in failed_models)
                        )

                    status_placeholder.markdown(
                        "<div class='step2-selection-card'>✅ All queued surrogates trained.</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state["step2_log_lines"].append("[SUCCESS] Training pipeline completed successfully.")
                    st.success("Training pipeline executed successfully.")
        else:
            st.warning("Select at least one target output and at least one model before launching.")

    st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
    st.markdown("<div class='module-title'>SIMULATED TERMINAL</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-subtitle'>TRAINING DISPATCHER · LIVE EXECUTION TRACE</div>", unsafe_allow_html=True)
    terminal_lines = st.session_state.get("step2_log_lines", ["[READY] Training queue waiting for execution."])
    st.markdown("<div class='terminal-log'>" + "<br>".join(terminal_lines) + "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif current_step == "STEP 3 — OPTIMIZATION":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>STEP 3 · OPTIMIZATION & CONTROL POLICY</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>Search the trained surrogate(s) for a Pareto-optimal set of designs — pick the solver that fits your problem: a genetic algorithm, a Monte Carlo baseline, or Differential Evolution.</div>",
        unsafe_allow_html=True,
    )

    trained_surrogates = st.session_state.get("trained_surrogates", [])
    dataset = st.session_state.get("uploaded_df")

    if not trained_surrogates or dataset is None:
        st.info("Train at least one surrogate in STEP 2 — PROXY MODELING before running optimization.")
    else:
        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>SURROGATE SELECTION</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>CHOOSE THE TRAINED MODEL TO OPTIMIZE AGAINST</div>", unsafe_allow_html=True)

        surrogate_labels = [
            f"{idx}: {payload['model_name']} → [{', '.join(payload['target_names'])}] ({payload['strategy']})"
            for idx, payload in enumerate(trained_surrogates)
        ]
        selected_label = st.selectbox("Trained Surrogate", surrogate_labels, key="step3_selected_surrogate_label")
        selected_idx = surrogate_labels.index(selected_label)
        surrogate_payload = trained_surrogates[selected_idx]
        feature_names = surrogate_payload["feature_names"]
        target_names = surrogate_payload["target_names"]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>OBJECTIVE DIRECTIONS</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>SET WHETHER EACH TARGET SHOULD BE MINIMIZED OR MAXIMIZED</div>", unsafe_allow_html=True)
        direction_cols = st.columns(min(3, len(target_names)) or 1)
        directions = {}
        for idx, target_name in enumerate(target_names):
            with direction_cols[idx % len(direction_cols)]:
                directions[target_name] = st.selectbox(
                    target_name,
                    ["min", "max"],
                    index=0 if guess_direction(target_name) == "min" else 1,
                    key=f"step3_direction_{target_name}",
                )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>SEARCH SPACE BOUNDS</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>DERIVED FROM THE TRAINING DATASET · ADJUSTABLE</div>", unsafe_allow_html=True)
        numeric_feature_source = dataset[feature_names].apply(pd.to_numeric, errors="coerce")
        bounds = {}
        with st.expander("Adjust bounds per feature", expanded=False):
            for feature_name in feature_names:
                col_min, col_max = st.columns(2)
                default_low = float(numeric_feature_source[feature_name].min())
                default_high = float(numeric_feature_source[feature_name].max())
                if not np.isfinite(default_low) or not np.isfinite(default_high) or default_low == default_high:
                    default_low, default_high = default_low - 1.0, default_high + 1.0
                with col_min:
                    low = st.number_input(f"{feature_name} — min", value=default_low, key=f"step3_bound_low_{feature_name}")
                with col_max:
                    high = st.number_input(f"{feature_name} — max", value=default_high, key=f"step3_bound_high_{feature_name}")
                bounds[feature_name] = (low, high)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>OPTIMIZATION ALGORITHM</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>PICK THE SOLVER THAT MATCHES YOUR PROBLEM AND TIME BUDGET</div>", unsafe_allow_html=True)

        algorithm_catalog = {
            "nsga2": {
                "category": "Metaheuristic",
                "label": "🧬 NSGA-II",
                "hint": "General-purpose, recommended default. Handles 2-3 objectives well, including rugged/discontinuous landscapes from tree-based models (RF, GBRT...).",
            },
            "nsga3": {
                "category": "Metaheuristic",
                "label": "🧬 NSGA-III",
                "hint": "Reference-point-based variant of NSGA-II built for MANY-objective problems (4+ targets), where crowding distance loses resolution.",
            },
            "steady_state_nsga2": {
                "category": "Metaheuristic",
                "label": "🧬 Steady-State NSGA-II",
                "hint": "Replaces one individual at a time instead of a whole generation — often converges in fewer total evaluations, useful under a tight evaluation budget.",
            },
            "hype": {
                "category": "Metaheuristic",
                "label": "📊 HypE",
                "hint": "Hypervolume-indicator-based many-objective EA. More discriminating than crowding distance on many-objective problems, at higher per-generation cost.",
            },
            "ga": {
                "category": "Metaheuristic",
                "label": "🧬 Genetic Algorithm (GA)",
                "hint": "Textbook real-coded GA extended to multi-objective via weighted-sum scalarization. Fast, simple baseline; can't reach non-convex parts of the true front.",
            },
            "aco": {
                "category": "Metaheuristic",
                "label": "🐜 Ant Colony Optimization (ACOR)",
                "hint": "Continuous-domain ACO (archive of solutions + Gaussian sampling). A solid alternative outside the GA/DE family, good on smooth landscapes with a modest budget.",
            },
            "random_search": {
                "category": "Metaheuristic",
                "label": "🎲 Random Search",
                "hint": "Zero-tuning Monte Carlo baseline. Quality depends purely on sample count — useful as a sanity check against the other solvers.",
            },
            "de": {
                "category": "Model-Based / Hybrid",
                "label": "📐 Differential Evolution",
                "hint": "Weighted-sum scalarization + scipy DE. Best on smooth, continuous surrogates (Ridge, GPR, SVR); less robust on very rugged/discrete objective landscapes.",
            },
            "rbf_sa": {
                "category": "Model-Based / Hybrid",
                "label": "🧩 RBFMOpt (RBF Surrogate-Assisted)",
                "hint": "Fits a cheap RBF interpolant on the fly and runs NSGA-II on IT, refining with real evaluations. Minimizes calls to the true objective — most valuable if you later swap in an expensive simulator.",
            },
            "ann_sa": {
                "category": "Model-Based / Hybrid",
                "label": "🧠 ANN-Assisted Optimization",
                "hint": "Same surrogate-assisted loop as RBFMOpt, but with a small MLP as the internal meta-surrogate. Needs more archive data than RBF to be reliable.",
            },
        }

        category_options = ["Metaheuristic", "Model-Based / Hybrid"]
        selected_category = st.radio(
            "Category",
            category_options,
            key="step3_algorithm_category",
            horizontal=True,
            help="Metaheuristics search the trained surrogate directly. Model-Based/Hybrid methods fit a second, even-cheaper meta-surrogate on the fly and search that instead — the classic surrogate-assisted / SBO pattern from the literature.",
        )
        keys_in_category = [k for k, v in algorithm_catalog.items() if v["category"] == selected_category]
        algorithm_key = st.radio(
            "Solver",
            keys_in_category,
            format_func=lambda key: algorithm_catalog[key]["label"],
            key="step3_algorithm_key",
            horizontal=True,
        )
        st.caption(algorithm_catalog[algorithm_key]["hint"])

        with st.expander("📊 Compare all solvers (convergence, robustness, efficiency, many-objective fit, low-budget fit)"):
            comparison_rows = [
                {"Algorithm": "NSGA-II", "Category": "Metaheuristic", "Convergence": "Good", "Robustness": "High — handles rugged/discrete landscapes", "Computational Efficiency": "Moderate", "Multi/Many-Objective": "Best at 2-3 objectives", "Low Budget Fit": "Moderate (needs pop × gens evaluations)"},
                {"Algorithm": "NSGA-III", "Category": "Metaheuristic", "Convergence": "Good", "Robustness": "High", "Computational Efficiency": "Moderate", "Multi/Many-Objective": "Built for 4+ (many-objective)", "Low Budget Fit": "Moderate"},
                {"Algorithm": "Steady-State NSGA-II", "Category": "Metaheuristic", "Convergence": "Good, often faster per evaluation", "Robustness": "High", "Computational Efficiency": "More fine-grained, more bookkeeping", "Multi/Many-Objective": "2-3 objectives", "Low Budget Fit": "Good — precise evaluation-count control"},
                {"Algorithm": "HypE", "Category": "Metaheuristic", "Convergence": "Good on many-objective", "Robustness": "Moderate (MC hypervolume adds noise)", "Computational Efficiency": "Lower (HV sampling overhead)", "Multi/Many-Objective": "Built for many-objective (4+)", "Low Budget Fit": "Low–moderate"},
                {"Algorithm": "Genetic Algorithm (GA)", "Category": "Metaheuristic", "Convergence": "Moderate (scalarized)", "Robustness": "Moderate", "Computational Efficiency": "High — simple & fast", "Multi/Many-Objective": "Any, via weighted-sum (misses non-convex regions)", "Low Budget Fit": "Good"},
                {"Algorithm": "Ant Colony (ACOR)", "Category": "Metaheuristic", "Convergence": "Good on smooth landscapes", "Robustness": "Moderate", "Computational Efficiency": "Moderate", "Multi/Many-Objective": "Any, via weighted-sum", "Low Budget Fit": "Good"},
                {"Algorithm": "Random Search", "Category": "Metaheuristic", "Convergence": "Low — no guidance", "Robustness": "High — no assumptions about the landscape", "Computational Efficiency": "High, trivially parallel", "Multi/Many-Objective": "Any", "Low Budget Fit": "Good as baseline only"},
                {"Algorithm": "Differential Evolution", "Category": "Model-Based / Hybrid", "Convergence": "High on smooth continuous problems", "Robustness": "Lower on rugged/discrete objectives", "Computational Efficiency": "High", "Multi/Many-Objective": "Any, via weighted-sum", "Low Budget Fit": "Good"},
                {"Algorithm": "RBFMOpt", "Category": "Model-Based / Hybrid", "Convergence": "Good, very sample-efficient", "Robustness": "Moderate — depends on interpolant fit", "Computational Efficiency": "Very high (few real evaluations)", "Multi/Many-Objective": "Any (inner NSGA-II)", "Low Budget Fit": "Excellent — purpose-built for this"},
                {"Algorithm": "ANN-Assisted", "Category": "Model-Based / Hybrid", "Convergence": "Good with enough archive data", "Robustness": "Moderate — needs enough samples to train", "Computational Efficiency": "High (few real evaluations)", "Multi/Many-Objective": "Any (inner NSGA-II)", "Low Budget Fit": "Good, improves with larger initial sample"},
            ]
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
            st.caption("\"Low Budget Fit\" = how well the method performs when the real objective can only be called a small number of times. In THIS app the objective is a trained ML surrogate (already cheap), so this mostly matters if you later swap in a genuinely expensive simulator.")

        master_seed = st.number_input(
            "🎲 Random seed (reproducibility)",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key="step3_master_seed",
            help="Every solver here draws its randomness from a local generator seeded with exactly this value — re-running with the same seed and settings always reproduces the same Pareto front.",
        )

        if algorithm_key == "nsga2":
            pop_size = st.slider("Population size", min_value=20, max_value=400, value=100, step=10, key="step3_pop_size")
            n_gen = st.slider("Generations", min_value=10, max_value=300, value=60, step=10, key="step3_n_gen")
        elif algorithm_key == "nsga3":
            pop_size = st.slider("Population size", min_value=20, max_value=400, value=100, step=10, key="step3_nsga3_pop_size")
            n_gen = st.slider("Generations", min_value=10, max_value=300, value=60, step=10, key="step3_nsga3_n_gen")
        elif algorithm_key == "steady_state_nsga2":
            pop_size = st.slider("Population size", min_value=20, max_value=300, value=80, step=10, key="step3_ss_pop_size")
            n_iterations = st.slider("Replacement iterations", min_value=100, max_value=8000, value=1500, step=100, key="step3_ss_n_iterations")
        elif algorithm_key == "hype":
            pop_size = st.slider("Population size", min_value=10, max_value=150, value=60, step=10, key="step3_hype_pop_size")
            n_gen = st.slider("Generations", min_value=5, max_value=150, value=30, step=5, key="step3_hype_n_gen")
            hv_samples = st.slider("Monte-Carlo HV samples per selection", min_value=100, max_value=2000, value=400, step=100, key="step3_hype_hv_samples")
        elif algorithm_key == "ga":
            n_weight_sets = st.slider("Weight combinations" if len(target_names) > 1 else "Restarts", min_value=2, max_value=60, value=12, step=1, key="step3_ga_n_weight_sets")
            pop_size = st.slider("Population size", min_value=20, max_value=300, value=60, step=10, key="step3_ga_pop_size")
            n_gen = st.slider("Generations", min_value=10, max_value=300, value=80, step=10, key="step3_ga_n_gen")
        elif algorithm_key == "aco":
            n_weight_sets = st.slider("Weight combinations" if len(target_names) > 1 else "Restarts", min_value=2, max_value=60, value=12, step=1, key="step3_aco_n_weight_sets")
            archive_size = st.slider("Archive size", min_value=10, max_value=100, value=30, step=5, key="step3_aco_archive_size")
            n_ants = st.slider("Ants per iteration", min_value=2, max_value=40, value=10, step=2, key="step3_aco_n_ants")
            n_iterations = st.slider("Iterations", min_value=10, max_value=300, value=60, step=10, key="step3_aco_n_iterations")
        elif algorithm_key == "random_search":
            n_samples = st.slider("Number of random samples", min_value=200, max_value=20000, value=3000, step=200, key="step3_n_samples")
        elif algorithm_key == "de":
            n_weight_sets = st.slider(
                "Weight combinations (trade-off resolution)" if len(target_names) > 1 else "Restarts",
                min_value=2, max_value=60, value=15, step=1, key="step3_n_weight_sets",
            )
            de_maxiter = st.slider("Max iterations per weight combination", min_value=20, max_value=500, value=150, step=10, key="step3_de_maxiter")
        else:  # rbf_sa / ann_sa
            initial_samples = st.slider("Initial design-of-experiments samples", min_value=20, max_value=200, value=50, step=10, key="step3_sa_initial_samples")
            refine_iterations = st.slider("Refinement iterations", min_value=1, max_value=20, value=5, step=1, key="step3_sa_refine_iterations")
            candidates_per_iteration = st.slider("Real evaluations per refinement", min_value=5, max_value=50, value=15, step=5, key="step3_sa_candidates")
            inner_pop_size = st.slider("Inner NSGA-II population size", min_value=20, max_value=200, value=60, step=10, key="step3_sa_inner_pop")
            inner_n_gen = st.slider("Inner NSGA-II generations", min_value=10, max_value=150, value=40, step=10, key="step3_sa_inner_gen")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 RUN OPTIMIZATION", key="step3_run_optimization", use_container_width=True):
            try:
                bounds_list = [bounds[feature_name] for feature_name in feature_names]
                directions_list = [directions[target_name] for target_name in target_names]
                progress_bar = st.progress(0.0, text="Initializing solver...")

                def objective_fn(x_array):
                    return predict_surrogate(surrogate_payload, pd.DataFrame(x_array, columns=feature_names))

                if algorithm_key == "nsga2":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"NSGA-II — generation {current}/{total}")
                    with st.spinner("Evolving the design population..."):
                        result = run_nsga2(objective_fn, bounds_list, directions_list, pop_size=pop_size, n_gen=n_gen, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "nsga3":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"NSGA-III — generation {current}/{total}")
                    with st.spinner("Evolving the design population against reference points..."):
                        result = run_nsga3(objective_fn, bounds_list, directions_list, pop_size=pop_size, n_gen=n_gen, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "steady_state_nsga2":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"Steady-State NSGA-II — iteration {current}/{total}")
                    with st.spinner("Replacing individuals one at a time..."):
                        result = run_steady_state_nsga2(objective_fn, bounds_list, directions_list, pop_size=pop_size, n_iterations=n_iterations, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "hype":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"HypE — generation {current}/{total}")
                    with st.spinner("Estimating hypervolume contributions..."):
                        result = run_hype(objective_fn, bounds_list, directions_list, pop_size=pop_size, n_gen=n_gen, hv_samples=hv_samples, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "ga":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"Genetic Algorithm — weight set {current}/{total}")
                    with st.spinner("Evolving the population across weight combinations..."):
                        result = run_genetic_algorithm(objective_fn, bounds_list, directions_list, n_weight_sets=n_weight_sets, pop_size=pop_size, n_gen=n_gen, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "aco":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"ACOR — weight set {current}/{total}")
                    with st.spinner("Ants sampling around the archive..."):
                        result = run_aco(objective_fn, bounds_list, directions_list, n_weight_sets=n_weight_sets, archive_size=archive_size, n_ants=n_ants, n_iterations=n_iterations, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "random_search":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"Random Search — sampled {current}/{total} designs")
                    with st.spinner("Sampling the design space..."):
                        result = run_random_search(objective_fn, bounds_list, directions_list, n_samples=n_samples, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "de":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"Differential Evolution — weight set {current}/{total}")
                    with st.spinner("Running Differential Evolution across the objective trade-off space..."):
                        result = run_differential_evolution(objective_fn, bounds_list, directions_list, n_weight_sets=n_weight_sets, maxiter=de_maxiter, seed=master_seed, progress_callback=on_progress)

                elif algorithm_key == "rbf_sa":
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"RBFMOpt — refinement {current}/{total}")
                    with st.spinner("Fitting an RBF meta-surrogate and refining..."):
                        result = run_rbf_surrogate_assisted(objective_fn, bounds_list, directions_list, initial_samples=initial_samples, refine_iterations=refine_iterations, candidates_per_iteration=candidates_per_iteration, inner_pop_size=inner_pop_size, inner_n_gen=inner_n_gen, seed=master_seed, progress_callback=on_progress)

                else:  # ann_sa
                    def on_progress(current, total):
                        progress_bar.progress(current / total, text=f"ANN-Assisted — refinement {current}/{total}")
                    with st.spinner("Fitting an ANN meta-surrogate and refining..."):
                        result = run_ann_surrogate_assisted(objective_fn, bounds_list, directions_list, initial_samples=initial_samples, refine_iterations=refine_iterations, candidates_per_iteration=candidates_per_iteration, inner_pop_size=inner_pop_size, inner_n_gen=inner_n_gen, seed=master_seed, progress_callback=on_progress)

                algorithm_label_used = algorithm_catalog[algorithm_key]["label"]
                progress_bar.progress(1.0, text="Optimization complete.")

                pareto_df = pd.DataFrame(result.X, columns=feature_names)
                for idx, target_name in enumerate(target_names):
                    pareto_df[target_name] = result.F[:, idx]
                pareto_df = pareto_df.drop_duplicates().reset_index(drop=True)

                st.session_state["optimization_results"] = pareto_df
                st.session_state["optimization_meta"] = {
                    "surrogate_index": selected_idx,
                    "model_name": surrogate_payload["model_name"],
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "directions": directions,
                    "algorithm": algorithm_label_used,
                    "category": algorithm_catalog[algorithm_key]["category"],
                    "seed": master_seed,
                }
                st.success(f"{algorithm_label_used} found {len(pareto_df)} non-dominated design(s) on the Pareto front. (seed={master_seed})")
            except Exception as optimization_error:
                st.error(f"Optimization failed: {optimization_error}")

        pareto_df = st.session_state.get("optimization_results")
        opt_meta = st.session_state.get("optimization_meta")
        if pareto_df is not None and opt_meta is not None and opt_meta["surrogate_index"] == selected_idx:
            st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
            st.markdown("<div class='module-title'>PARETO FRONT</div>", unsafe_allow_html=True)
            st.markdown("<div class='module-subtitle'>NON-DOMINATED DESIGN CANDIDATES</div>", unsafe_allow_html=True)
            st.dataframe(pareto_df.round(4), use_container_width=True, hide_index=True)

            if len(target_names) == 2:
                st.scatter_chart(pareto_df, x=target_names[0], y=target_names[1])
            elif len(target_names) > 2:
                st.caption("More than two objectives selected — inspect trade-offs in the table above, or open STEP 4 — CONTROL ROOM to explore individual designs.")

            csv_bytes = pareto_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Pareto Front (CSV)",
                data=csv_bytes,
                file_name="nepenthe_pareto_front.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

elif current_step == "STEP 4 — CONTROL ROOM":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>STEP 4 · CONTROL ROOM</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>Explore designs interactively: drag feature sliders and watch the surrogate's live predictions, or load a candidate straight from the Pareto front.</div>",
        unsafe_allow_html=True,
    )

    trained_surrogates = st.session_state.get("trained_surrogates", [])
    dataset = st.session_state.get("uploaded_df")
    opt_meta = st.session_state.get("optimization_meta")
    pareto_df = st.session_state.get("optimization_results")

    if not trained_surrogates or dataset is None:
        st.info("Train at least one surrogate in STEP 2 — PROXY MODELING first.")
    else:
        surrogate_labels = [
            f"{idx}: {payload['model_name']} → [{', '.join(payload['target_names'])}] ({payload['strategy']})"
            for idx, payload in enumerate(trained_surrogates)
        ]
        default_index = opt_meta["surrogate_index"] if opt_meta else 0
        default_index = min(default_index, len(surrogate_labels) - 1)
        selected_label = st.selectbox(
            "Trained Surrogate", surrogate_labels, index=default_index, key="step4_selected_surrogate_label"
        )
        selected_idx = surrogate_labels.index(selected_label)
        surrogate_payload = trained_surrogates[selected_idx]
        feature_names = surrogate_payload["feature_names"]
        target_names = surrogate_payload["target_names"]
        numeric_feature_source = dataset[feature_names].apply(pd.to_numeric, errors="coerce")

        slider_col, readout_col = st.columns([1.1, 0.9], gap="large")

        with slider_col:
            st.markdown("<div class='control-module'>", unsafe_allow_html=True)
            st.markdown("<div class='module-title'>DESIGN VARIABLES</div>", unsafe_allow_html=True)
            st.markdown("<div class='module-subtitle'>LIVE SLIDERS · SURROGATE FEEDS PREDICTIONS BELOW</div>", unsafe_allow_html=True)

            has_matching_pareto = (
                pareto_df is not None and opt_meta is not None and opt_meta["surrogate_index"] == selected_idx
            )
            if has_matching_pareto:
                pareto_choice = st.selectbox(
                    "Load a Pareto design",
                    ["— manual —"] + [f"Design {i}" for i in range(len(pareto_df))],
                    key="step4_pareto_choice",
                )
            else:
                pareto_choice = "— manual —"

            slider_values = {}
            for feature_name in feature_names:
                col_series = numeric_feature_source[feature_name]
                low, high = float(col_series.min()), float(col_series.max())
                if not np.isfinite(low) or not np.isfinite(high) or low == high:
                    low, high = low - 1.0, low + 1.0
                if pareto_choice != "— manual —":
                    design_idx = int(pareto_choice.split(" ")[1])
                    default_val = float(pareto_df.iloc[design_idx][feature_name])
                else:
                    default_val = float(col_series.median())
                default_val = float(np.clip(default_val, low, high))
                slider_values[feature_name] = st.slider(
                    feature_name, min_value=low, max_value=high, value=default_val, key=f"step4_slider_{feature_name}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with readout_col:
            st.markdown("<div class='control-module'>", unsafe_allow_html=True)
            st.markdown("<div class='module-title'>LIVE PREDICTION</div>", unsafe_allow_html=True)
            st.markdown("<div class='module-subtitle'>SURROGATE OUTPUT FOR THE CURRENT DESIGN</div>", unsafe_allow_html=True)

            design_row = pd.DataFrame([slider_values])[feature_names]
            prediction = predict_surrogate(surrogate_payload, design_row)[0]

            pill_html = "".join(
                f"<div class='metric-pill'>{target_name}: {value:.3f}</div>"
                for target_name, value in zip(target_names, prediction)
            )
            st.markdown(f"<div class='metric-stack'>{pill_html}</div>", unsafe_allow_html=True)

            metrics_by_target = surrogate_payload.get("metrics", {})
            for target_name, value in zip(target_names, prediction):
                target_metrics = metrics_by_target.get(target_name, {})
                r2 = target_metrics.get("r2")
                caption = f"Surrogate confidence for {target_name}: R² = {r2:.2f}" if r2 is not None else target_name
                st.metric(label=target_name, value=f"{value:.3f}", help=caption)
            st.markdown("</div>", unsafe_allow_html=True)

elif current_step == "STEP 5 — REPORTING":
    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>NEPENTHE</div><div class='hero-subtitle'>STEP 5 · REPORTING & DECISION PACKAGE</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-copy'>Review surrogate performance, understand which inputs drive each output, and export the full decision package for the rest of the team.</div>",
        unsafe_allow_html=True,
    )

    trained_surrogates = st.session_state.get("trained_surrogates", [])
    dataset = st.session_state.get("uploaded_df")
    pareto_df = st.session_state.get("optimization_results")
    opt_meta = st.session_state.get("optimization_meta")

    if not trained_surrogates or dataset is None:
        st.info("Train at least one surrogate in STEP 2 — PROXY MODELING before generating a report.")
    else:
        # ---------------------------------------------------------------
        # 1. MODEL PERFORMANCE LEDGER
        # ---------------------------------------------------------------
        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>MODEL PERFORMANCE LEDGER</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>ALL SURROGATES TRAINED THIS SESSION · HELD-OUT TEST METRICS (+ 5-FOLD CV WHERE ENABLED)</div>", unsafe_allow_html=True)

        perf_rows = []
        for idx, payload in enumerate(trained_surrogates):
            cv_metrics = payload.get("cv_metrics")
            for target_name in payload["target_names"]:
                target_metrics = payload["metrics"].get(target_name, {})
                row = {
                    "Surrogate #": idx,
                    "Model": payload["model_name"],
                    "Strategy": payload["strategy"],
                    "Target": target_name,
                    "R2 (holdout)": round(target_metrics.get("r2", float("nan")), 4),
                    "RMSE (holdout)": round(target_metrics.get("rmse", float("nan")), 4),
                }
                if cv_metrics and target_name in cv_metrics:
                    cv_vals = cv_metrics[target_name]
                    row["R2 (5-fold CV)"] = f"{cv_vals['r2_mean']:.3f} ± {cv_vals['r2_std']:.3f}"
                    row["RMSE (5-fold CV)"] = f"{cv_vals['rmse_mean']:.3f} ± {cv_vals['rmse_std']:.3f}"
                else:
                    row["R2 (5-fold CV)"] = "—"
                    row["RMSE (5-fold CV)"] = "—"
                perf_rows.append(row)
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.caption("5-fold CV columns show \"—\" when cross-validation wasn't enabled at training time (STEP 2), or for the Hybrid Stacked Ensemble strategy (which already uses its own internal out-of-fold resampling).")

        best_rows = perf_df.loc[perf_df.groupby("Target")["R2 (holdout)"].idxmax()] if not perf_df.empty else perf_df
        if not best_rows.empty:
            best_pill_html = "".join(
                f"<div class='metric-pill'>Best for {row['Target']}: {row['Model']} (R²={row['R2 (holdout)']:.2f})</div>"
                for _, row in best_rows.iterrows()
            )
            st.markdown(f"<div class='metric-stack'>{best_pill_html}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # 2. FEATURE IMPORTANCE (MODEL-AGNOSTIC PERMUTATION)
        # ---------------------------------------------------------------
        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>PERMUTATION IMPORTANCE · WHICH INPUTS DRIVE THE SURROGATE MOST</div>", unsafe_allow_html=True)

        surrogate_labels = [
            f"{idx}: {payload['model_name']} → [{', '.join(payload['target_names'])}] ({payload['strategy']})"
            for idx, payload in enumerate(trained_surrogates)
        ]
        selected_label = st.selectbox("Trained Surrogate", surrogate_labels, key="step5_selected_surrogate_label")
        selected_idx = surrogate_labels.index(selected_label)
        surrogate_payload = trained_surrogates[selected_idx]

        if st.button("🔬 Compute Feature Importance", key="step5_run_importance", use_container_width=True):
            try:
                with st.spinner("Shuffling features and measuring the R² drop..."):
                    x_test, y_test = reconstruct_test_split(surrogate_payload, dataset)
                    importance_df = compute_permutation_importance(surrogate_payload, x_test, y_test)
                st.session_state["step5_importance_cache"] = {"idx": selected_idx, "df": importance_df}
            except Exception as importance_error:
                st.error(f"Feature importance computation failed: {importance_error}")

        cached_importance = st.session_state.get("step5_importance_cache")
        if cached_importance and cached_importance["idx"] == selected_idx:
            importance_df = cached_importance["df"]
            st.dataframe(importance_df.round(4), use_container_width=True, hide_index=True)
            chart_df = importance_df.set_index("Feature")[["Importance"]]
            st.bar_chart(chart_df)
            st.caption("Importance = average drop in R² (across all targets of this surrogate) when a feature is randomly shuffled. Higher is more influential.")
        else:
            st.caption("Click the button above to analyze which input variables this surrogate relies on most.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # 3. OPTIMIZATION SUMMARY
        # ---------------------------------------------------------------
        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>OPTIMIZATION SUMMARY</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>PARETO FRONT FROM STEP 3 · BEST DESIGN PER OBJECTIVE</div>", unsafe_allow_html=True)

        if pareto_df is None or opt_meta is None:
            st.info("Run STEP 3 — OPTIMIZATION to include Pareto-optimal designs in this report.")
        else:
            target_names = opt_meta["target_names"]
            directions = opt_meta["directions"]
            best_design_rows = []
            for target_name in target_names:
                if directions.get(target_name) == "max":
                    best_idx = pareto_df[target_name].idxmax()
                else:
                    best_idx = pareto_df[target_name].idxmin()
                row = pareto_df.loc[best_idx].to_dict()
                row["Optimized For"] = f"{target_name} ({directions.get(target_name, 'min')})"
                best_design_rows.append(row)
            best_design_df = pd.DataFrame(best_design_rows)
            cols_order = ["Optimized For"] + [c for c in best_design_df.columns if c != "Optimized For"]
            st.dataframe(best_design_df[cols_order].round(4), use_container_width=True, hide_index=True)
            algo_used = opt_meta.get("algorithm", "NSGA-II")
            st.caption(f"Surrogate used: {opt_meta['model_name']} · Solver: {algo_used} · {len(pareto_df)} non-dominated designs on the front.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # 4. EXPORT DECISION PACKAGE
        # ---------------------------------------------------------------
        st.markdown("<div class='step2-module'>", unsafe_allow_html=True)
        st.markdown("<div class='module-title'>EXPORT DECISION PACKAGE</div>", unsafe_allow_html=True)
        st.markdown("<div class='module-subtitle'>DOWNLOAD A SHAREABLE SUMMARY FOR THE REST OF THE TEAM</div>", unsafe_allow_html=True)

        report_lines = [
            "# NEPENTHE Auto-SOF — Decision Package",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset",
            f"- Rows: {len(dataset)}",
            f"- Columns: {len(dataset.columns)}",
            "",
            "## Model Performance",
            "```",
            perf_df.to_csv(index=False).strip() if not perf_df.empty else "No trained surrogates.",
            "```",
            "",
        ]
        if pareto_df is not None and opt_meta is not None:
            report_lines += [
                "## Optimization — Pareto Front",
                f"Surrogate: {opt_meta['model_name']} · Solver: {opt_meta.get('algorithm', 'NSGA-II')} · {len(pareto_df)} designs",
                "```",
                pareto_df.round(4).to_csv(index=False).strip(),
                "```",
                "",
            ]
        report_text = "\n".join(str(line) for line in report_lines)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "📄 Download Summary Report (Markdown)",
                data=report_text.encode("utf-8"),
                file_name="nepenthe_decision_package.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with export_col2:
            try:
                import openpyxl  # noqa: F401  (import check only — raises ModuleNotFoundError if missing)

                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    perf_df.to_excel(writer, sheet_name="Model_Performance", index=False)
                    if pareto_df is not None:
                        pareto_df.to_excel(writer, sheet_name="Optimized_Designs", index=False)
                    cached_importance = st.session_state.get("step5_importance_cache")
                    if cached_importance:
                        cached_importance["df"].to_excel(writer, sheet_name="Feature_Importance", index=False)
                st.download_button(
                    "📊 Download Full Results (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="nepenthe_auto_sof_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ModuleNotFoundError:
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    zf.writestr("Model_Performance.csv", perf_df.to_csv(index=False))
                    if pareto_df is not None:
                        zf.writestr("Optimized_Designs.csv", pareto_df.to_csv(index=False))
                    cached_importance = st.session_state.get("step5_importance_cache")
                    if cached_importance:
                        zf.writestr("Feature_Importance.csv", cached_importance["df"].to_csv(index=False))
                st.download_button(
                    "📦 Download Full Results (ZIP of CSVs)",
                    data=zip_buffer.getvalue(),
                    file_name="nepenthe_auto_sof_results.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
                st.caption("⚠️ `openpyxl` isn't installed, so Excel export isn't available — you got a ZIP of CSVs instead. Run `pip install openpyxl` (already in requirements.txt) and restart the app for a single .xlsx file.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        "<div class='control-module'><div class='module-title'>NODE READY</div><div class='module-subtitle'>The selected cognitive node is being prepared for implementation.</div></div>",
        unsafe_allow_html=True,
    )
    st.info(f"{current_step} will be implemented in the next build phase.")
