# NEPENTHE | Auto-SOF

A Streamlit web app version of your `App.ipynb` notebook — an automated
surrogate-based optimization framework (Auto-SOF) that lets a team:

1. **Upload data** (or use demo data) and validate it.
2. **Train surrogate models** (18 regressors across 3 "roots": probabilistic,
   linear/regularized, and ensemble/deep) as a single model or a stacked
   hybrid ensemble, for one or several targets at once — with a live
   progress bar and status readout so you can watch each model (and each
   cross-validation fold, for the hybrid strategy) train in real time.
3. **Optimize** the trained surrogate with a solver you pick to match your
   problem: NSGA-II (general-purpose genetic algorithm), Random Search
   (zero-tuning Monte Carlo baseline), or Differential Evolution (weighted
   scalarization, best for smooth continuous surrogates) — to find
   Pareto-optimal designs.
4. **Explore designs live** in a control room with sliders feeding the
   surrogate in real time.
5. **Report & export** — performance ledger (with optional 5-fold
   cross-validated metrics), permutation feature importance, Pareto summary,
   and one-click Markdown / Excel (or CSV-zip fallback) export.

## Reliability & professional-workflow features

- **Cross-validation**: tick "Also compute 5-fold cross-validated metrics" in
  STEP 2 to get mean ± std R²/RMSE per target, not just a single noisy
  80/20 holdout score. Skipped for the Hybrid Stacked Ensemble strategy
  (which already does its own internal out-of-fold resampling).
- **Project save/load**: the sidebar lets you download your whole session
  (dataset + every trained surrogate + optimization results) as one
  `.joblib` file, and reload it later or hand it to a teammate — no more
  losing everything on a page refresh.
- **Graceful degradation**: if `openpyxl` isn't installed, the Excel export
  button automatically falls back to a ZIP of CSVs instead of crashing, with
  an on-screen hint to `pip install openpyxl`.
- **Error containment**: training, optimization, and feature-importance runs
  are wrapped so one failing model or run shows a clear message instead of
  taking down the whole app; other queued models keep training.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Share it with your team

- **Easiest**: push this folder to a GitHub repo and deploy for free on
  [Streamlit Community Cloud](https://share.streamlit.io) — pick the repo,
  set `app.py` as the entry point, and you get a public URL.
- **Internal/company use**: run it on any server/VM with
  `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`, or
  containerize it (a minimal `Dockerfile` would just `pip install -r
  requirements.txt` then `CMD ["streamlit","run","app.py"]`).

## Roadmap — recommended next steps

Not implemented yet, but worth prioritizing for a more "production" data
science tool:
- Hyperparameter tuning (`RandomizedSearchCV` / Optuna) instead of fixed defaults
- Real SHAP-based explainability + partial-dependence/ICE plots
- Constraint handling in optimization (most engineering problems need more
  than simple box bounds on each variable)
- Sensitivity analysis (Sobol indices), as sketched in the original notebook
- A "batch scoring" tab: upload new candidate designs, get predictions from
  the best surrogate without re-running the whole pipeline
- `st.cache_data`/`st.cache_resource` so navigating between steps doesn't
  recompute things unnecessarily
- Categorical feature support (currently numeric-only; non-numeric columns
  are coerced and NaN-filled with the median)
- Swap the placeholder registry entries (RBF, CNN, LSTM, TabNet, EBM
  currently fall back to a generic ensemble under the hood) for their real
  implementations if you need them to behave as labeled

## Project structure

```
app.py                # main Streamlit app (all 6 workflow steps)
core/kernel.py         # session-state / telemetry bookkeeping
core/optimizer.py       # dependency-free NSGA-II implementation (numpy only)
ui/styles.css           # dark "control room" visual theme
requirements.txt
```

## Notes on what changed vs. the notebook

- Colab-only bits (`google.colab.files`, `ipywidgets`) were replaced with
  native Streamlit widgets and `st.download_button` so it runs anywhere.
- SHAP was swapped for a lightweight, dependency-free permutation
  importance (shuffle-and-measure-R²-drop) so the app doesn't need the
  heavier `shap` package and works uniformly for both single models and the
  hybrid stacked ensemble.
- The optimization engine offers three solvers in `core/optimizer.py`,
  rather than `pymoo`/`deap`, to keep the dependency footprint small:
  - `run_nsga2` — a compact from-scratch NSGA-II (numpy only).
  - `run_random_search` — Monte Carlo sampling + Pareto filtering.
  - `run_differential_evolution` — `scipy.optimize.differential_evolution`
    run across a spread of weighted-sum scalarizations, then filtered to
    the non-dominated union.
  Swap in `pymoo` later if you want NSGA-III, MOPSO, TPE, etc. like the
  notebook sketched.
