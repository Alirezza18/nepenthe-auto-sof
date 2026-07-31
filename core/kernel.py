"""
core.kernel
-----------
Lightweight "engine" bookkeeping for the NEPENTHE Auto-SOF platform.

app.py imports two things from here:
    - initialize_engine_state(force_reset: bool)
    - get_system_telemetry() -> dict

This module intentionally owns only the *engine-level* bits of session
state (kernel version, a place for future model-registry bookkeeping).
All UI/workflow state (uploaded_df, step2_*, trained_surrogates, ...)
stays owned by app.py itself so this module can be re-run safely on
every Streamlit rerun without clobbering user work.
"""
from __future__ import annotations

import streamlit as st

KERNEL_VERSION = "2.4.0"

# Keys that belong to the "engine" and are safe to wipe on a force reset.
_ENGINE_RESET_KEYS = (
    "uploaded_df",
    "trained_surrogates",
    "step2_log_lines",
    "step2_target_variables",
    "step2_selected_model",
    "step2_selected_models",
    "optimization_results",
    "optimization_meta",
)


def initialize_engine_state(force_reset: bool = False) -> None:
    """Ensure the engine-level session_state keys exist.

    Idempotent by design: Streamlit reruns the whole script top-to-bottom
    on every widget interaction, so this gets called constantly. It must
    never overwrite state that already exists unless force_reset=True.
    """
    if force_reset:
        for key in _ENGINE_RESET_KEYS:
            st.session_state.pop(key, None)

    st.session_state.setdefault("kernel_version", KERNEL_VERSION)
    st.session_state.setdefault("trained_surrogates", [])
    st.session_state.setdefault("optimization_results", None)
    st.session_state.setdefault("optimization_meta", None)


def get_system_telemetry() -> dict:
    """Small dict of cosmetic/real numbers used in the sidebar + Step 0 panels."""
    trained = st.session_state.get("trained_surrogates", []) or []
    strategies = {payload.get("strategy") for payload in trained if isinstance(payload, dict)}
    has_optimization = st.session_state.get("optimization_results") is not None

    return {
        "Kernel_V": KERNEL_VERSION,
        "Proxies": len(trained),
        "Clusters": len(strategies) if strategies else (1 if trained else 0),
        "Optimized": "YES" if has_optimization else "NO",
    }
