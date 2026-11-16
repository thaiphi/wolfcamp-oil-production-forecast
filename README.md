# Wolfcamp oil production forecasting

An end-to-end **ML Zoomcamp midterm** project predicting oil production during months 7–12 from measurements available after a well's first six months. This supervised regression project tests whether machine learning improves future-oil estimates over simple forecasting benchmarks. Add observed six-month oil to the predicted increment to obtain cumulative first-year oil. Units are oil barrels (BBL).

Start with the **[current executed submission notebook](notebooks/wolfcamp_midterm_submission.ipynb)**. It follows the complete workflow: problem → data exploration → preprocessing → training and tuning → evaluation → saved pipeline → API → Docker. Section 6 shows separate executable tuning sections and trial outputs for all five model families. `notebook.ipynb` is a synchronized copy of this same notebook.

## Data and problem

The included [CSV](data/raw/wolfcamp_wells.csv) contains 2,105 completion records and 252 fields from a historical Wolfcamp/Delaware Basin well export. The owner confirmed permission to share the data; reviewers need no credentials. [Data provenance](data/raw/README.md).

Target: `First12MonthOil_BBL - First6MonthOil_BBL`. Future production fields are excluded from predictors. One valid record per well is retained; the study requires TVD 7,000–15,000 ft, lateral 4,000–16,000 ft and at least twelve producing months. TVD is a proxy for producing depth. Completion covariates with missing dates or dates after month six are masked.

Chronological cohorts: **1,249 training wells (2014–18), 428 validation wells (2019), 264 historical holdout wells (2020–21)**. Training labels are purged if they would not mature by the earliest forecast date in the scoring cohort. Calendar maturity approximates actual producing/reporting availability.

## Modeling workflow

The pipeline accepts 25 raw predictors and constructs eleven month-six production and completion-intensity features. Numeric fields receive training-fitted median imputation, missingness indicators and standard scaling. Categorical fields receive training-mode imputation and one-hot encoding, grouping categories with fewer than ten training observations. The notebook shows the actual fills and encoded outputs. All learned preprocessing is fitted inside the training folds and reused at inference.

Preprocessing visuals adapt the author's original missingness charts, histogram grids and correlation heatmaps. They are regenerated from the current training cohort: missingness patterns and bins, observed-value correlations, before/after median-fill distributions, before/after scaling histograms and a one-hot encoding matrix. Captions distinguish observed measurements from imputed values and explain why scaling preserves shape. These diagnostic figures do not claim to measure unknown-value imputation accuracy.

Train **ordinary linear regression, Ridge, Decision Tree, Random Forest and XGBoost** with the same predictors, preprocessing, purged folds and clipped RMSE. Ridge, Decision Tree, Random Forest and XGBoost each receive twelve deterministic tuning configurations; ordinary least squares is an untuned reference. The search contains **49 trials and 147 fold fits**. [Exact selection protocol](reports/MODEL_SELECTION_PROTOCOL.md) and [all parameters/fold scores](reports/tuning.json).

| Model family | Trials | Mean training-CV RMSE, BBL | 2019 validation RMSE, BBL |
|---|---:|---:|---:|
| Ordinary linear regression | 1 | 28,897.63 | 22,229.64 |
| Ridge | 12 | 26,061.19 | 21,358.70 |
| Decision Tree | 12 | 28,112.04 | 23,895.85 |
| **Random Forest** | 12 | **25,623.00** | 21,089.47 |
| XGBoost | 12 | 25,981.49 | 21,370.06 |

**Selection is based only on mean training-CV RMSE**, so Random Forest wins this declared search. These are the reference **Docker/Linux** results. Windows selected the same forest and reproduced its metrics/predictions, but XGBoost's scores differed (Windows CV 26,028.29; validation 20,449.73 BBL). [Both observed runs and differences](reports/platform_reproducibility.json) are preserved. Equal trial counts are not exhaustive optimization or equal compute. No model is claimed universally best.

The selected forest uses **240 trees, minimum leaf size 5, unrestricted depth, max_features 0.75 and full-size bootstrap samples**, with seed 42. It is refit on 1,500 label-mature 2014–19 wells. Only that selected specification receives a historical holdout score in this run.

| Historical holdout metric | Random Forest | Recent-rate benchmark | Training-median benchmark |
|---|---:|---:|---:|
| RMSE, BBL | 19,158.43 | 20,538.31 | 34,519.09 |
| MAE, BBL | 13,273.65 | 13,255.89 | 25,292.31 |
| R² | 0.6202 | 0.5635 | -0.2329 |

The forest reduces RMSE by **6.72% versus the recent-rate benchmark** and **44.50% versus the training-median benchmark**. Its MAE is 17.76 BBL higher than the recent-rate benchmark, so the benefit is in squared error rather than a consistent advantage on every metric. Its largest impurity importances are recent months 4–6 oil and total six-month oil; these describe predictive use, not causal effects.

**The historical holdout was previously inspected during development; it is not fresh independent prospective evidence.** Only 17 of the 264 holdout wells are from 2021. Regional/domain restrictions, 90.6% training missingness in FracStages, approximate calendar maturity and unknown source revisions limit generalization. The service gives point forecasts without calibrated uncertainty intervals. The notebook's conclusion, limitations and next steps explain how those limits affect interpretation and set out later-cohort validation, data-quality checks, subgroup error analysis and selection-stability work. Callers must enforce the month-six snapshot and study domain.

## Run the project

Use Python 3.12. In PowerShell from this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
.\.venv\Scripts\python.exe verify_project.py
.\.venv\Scripts\python.exe train.py --output reports/reproduction
.\.venv\Scripts\python.exe verify_project.py --rerun reports/reproduction --allow-platform-differences
.\.venv\Scripts\python.exe -m jupyter lab notebooks/wolfcamp_midterm_submission.ipynb
```

On Linux/macOS create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python`. Activation is optional. Training exports `model.joblib`, `metrics.json`, `tuning.json` and `tuning.csv` into the chosen output folder. Separate run folders preserve packaged results. The supplied deployable artifact is `models/model.joblib`.

To activate the environment, run `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux/macOS. The explicit interpreter commands above work without activation. Training also exports `selection.json` and `model_selection.csv` so the selected family and the other candidates' diagnostics are inspectable.

The notebook and published tuning reports were executed in Docker/Linux. There, `verify_project.py --rerun reports/reproduction` checks the complete search strictly. The Windows command above explicitly records nonselected-family numerical differences; it still requires the same winner, parameters, rounded final metrics and saved predictions.

On Windows, extract the project to a short directory such as `C:\projects\wolfcamp` before creating `.venv`. Very deeply nested paths can prevent compiled scikit-learn modules from importing even when installation succeeds.

For local serving without Docker:

```text
python -m uvicorn predict:app --host 127.0.0.1 --port 8000
```

## Docker and prediction API

```text
docker build -t wolfcamp-midterm-submission .
docker run --rm -p 127.0.0.1:8000:8000 wolfcamp-midterm-submission
```

Open `http://127.0.0.1:8000/docs`. In a second terminal:

Reviewer evidence: [actual Docker API request and HTTP 200 prediction](docs/docker-prediction.png).

![Docker API prediction](docs/docker-prediction.png)

```text
python examples/call_api.py
python examples/smoke_api.py --url http://127.0.0.1:8000
python examples/verify_api_parity.py --url http://127.0.0.1:8000
```

`GET /health` checks the model, `GET /features` describes allowed inputs, and `POST /predict` returns incremental and cumulative oil forecasts. A minimal request is:

```json
{"features": {"First6MonthOil_BBL": 61280}}
```

A full request is in [examples/request.json](examples/request.json). Optional fields may be omitted/null and use fitted imputation. Future/unknown fields, negative production and invalid types are rejected. Supply only month-six values and omit completion attributes whose dates are unknown or later than month six; the API cannot verify dates or vendor revisions. `MODEL_PATH` can point to a retrained artifact.

## Project files

```text
notebooks/wolfcamp_midterm_submission.ipynb  Executed end-to-end walkthrough
src/wolfcamp/            Data, features, preprocessing and training
train.py                 Tune, refit, evaluate and export
predict.py               FastAPI prediction service
models/model.joblib      Complete fitted pipeline
reports/                 Metrics, tuning and verification evidence
tests/                   Automated data, training and API checks
Dockerfile               Reproducible runtime image
verify_project.py        Artifact and retraining verification
```

The project uses problem framing, regression, categorical processing, validation, serialization, serving and tree-ensemble concepts from Modules 1–6. Its primary task is regression; MAE/RMSE/R² are the appropriate metrics. AI assisted implementation, documentation and review; the author remains responsible for the decisions and results.

See [VERIFICATION.md](VERIFICATION.md) for executed checks and [SUBMISSION.md](SUBMISSION.md) for GitHub submission steps.
