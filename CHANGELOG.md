# Changelog

All notable changes to Auto-SOF are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: SemVer.

## [Unreleased]

### Changed
- Citation block upgraded to APA 7 / IEEE / version-pinned BibTeX; GitHub "Cite this repository" pointer added
- `CITATION.cff` enriched (affiliation, license, keywords, abstract) for richer citation exports
- `.zenodo.json` added so the future Zenodo DOI record mints with complete metadata
- Placeholder DOI badge/field removed; author title corrected (PhD in Architecture, 2026)

## [1.0.0] — 2026-09-25

### Added
- Six-step Streamlit workflow: Welcome & Diagnostics → Data Intake → Proxy Modeling → Optimization → Control Room → Reporting
- 21-entry surrogate registry across probabilistic, regularized-linear, and ensemble/deep families
- Hybrid stacked ensembling with out-of-fold meta-learning and per-fold live progress
- Validation: held-out 80/20 metrics plus optional 5-fold cross-validation (mean ± std R², RMSE)
- Ten optimizers, including from-scratch NSGA-II, NSGA-III, Steady-State NSGA-II, HypE, and two surrogate-assisted loops
- Model-agnostic permutation feature importance
- Exports: Pareto table + plot, CSV/Markdown/Excel (CSV-zip fallback)
- Session save/restore as a single `.joblib` project file
- Docker image (`ghcr.io/alirezza18/auto-sof`) and CI workflow
