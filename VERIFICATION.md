# Verification

This project has its own executable training workflow, dataset, saved pipeline, notebook and Docker service. Commands below run from the project root with Python 3.12 and `requirements-dev.txt` installed. Evidence is in `reports/verification/`.

## Training and saved results

```text
python train.py --output reports/reproduction
python verify_project.py --rerun reports/reproduction
```

The training run evaluates ordinary linear regression and twelve configurations each of Ridge, Decision Tree, Random Forest and XGBoost: 49 trials / 147 purged chronological fold fits. Family and parameters are selected by training CV before 2019 diagnostics. Only the selected family is refit and scored on the historical holdout. Selected Random Forest metrics are RMSE 19,158.43 BBL, MAE 13,273.65 BBL, R² 0.6202. Eleven artifact checksums protect the dataset, model, synchronized notebooks, selection protocol, platform results and reports.

## Notebook and automated tests

```text
python -m pytest -q --basetemp .pytest-tmp
python scripts/verify_notebook.py
```

The current notebook is `notebooks/wolfcamp_midterm_submission.ipynb` (with an identical `notebook.ipynb` copy). It executes twenty-three code cells, with each model's constructor, parameter configurations, common fit/scoring loop and all trial tables visible inline in Sections 6.1–6.7. It also executes preprocessing, serialization parity and an API request. Nine rendered figures include five preprocessing figures adapted from the author's original notebooks: missingness matrix/bins, correlation map, median-fill distributions, scaling distributions and encoding matrix. The figures were visually inspected, and data assertions verify the fills, finite scaled values and indicator membership. Original-file hashes and unchanged core artifact checks are recorded in `reports/verification/preprocessing_visuals.json`. Automated tests cover cohort validity, label purging, fold-local preprocessing, unseen/missing inputs, saved metrics, API parity, input rejection, unavailable-model health behavior and the presence of executed model-specific tuning sections.

## Docker checks

```text
docker build -t wolfcamp-midterm-submission .
docker run --rm -p 127.0.0.1:8000:8000 wolfcamp-midterm-submission
python examples/smoke_api.py --url http://127.0.0.1:8000
python examples/verify_api_parity.py --url http://127.0.0.1:8000
```

The smoke test checks fourteen HTTP cases. Five additional inputs compare actual HTTP predictions with direct calls to the packaged pipeline. Runtime predictions were also checked from inside the container. The runtime image contains the selected pipeline and has no dependency on a parent project.

For Linux tests, first build the runtime image, then add development requirements with `docker build -f scripts/Dockerfile.tests -t wolfcamp-submission-tests .`. Mount a clean extracted project as `/work` and run tests/training with that image. Test dependencies are separate from the runtime image.

## Clean-copy verification

The upload ZIP is extracted into a separate directory. A fresh virtual environment installs the supplied requirements; tests, artifact verification and notebook execution run there. Docker is also built from that extracted directory. Concrete final outcomes are recorded in `reports/verification/summary.json`; packaging records checksums in `submission_files.json` and a ZIP SHA-256 sidecar.

Final multi-model outcomes are recorded in `reports/verification/summary.json` after the updated checks. Earlier RF-only evidence is preserved separately from this package. Verification distinguishes exact selected-pipeline parity from numerical differences in nonselected families. All twelve XGBoost trials differed across Windows/Linux; the largest individual fold difference was 1,597.32 BBL. Best XGBoost mean CV RMSE was 26,028.29 BBL on Windows versus 25,981.49 on Linux. The forest won CV on both platforms, with matching parameters, rounded final metrics and saved predictions. Full Windows/Linux details are in `reports/platform_reproducibility.json`.

The reference notebook and tuning reports are executed in Docker/Linux, where the complete 49-trial search was reproduced strictly (all 147 fold scores within 0.01 BBL). A strict cross-platform trial-score comparison initially failed on XGBoost; that failure is preserved in `clean_training_strict_failure.log`. Windows verification uses `--allow-platform-differences` to enumerate those nonselected differences and returns `PASS_WITH_PLATFORM_DIFFERENCES`; it does not relax winner or deployed-artifact checks. Do not describe all family scores as identical across operating systems.

Final clean checks are summarized in `reports/verification/summary.json`. The release Docker image passed fourteen HTTP cases plus five direct-artifact prediction-parity cases. [A real Swagger request and HTTP 200 response](docs/docker-prediction.png) is included for reviewer-visible deployment evidence. The clean release is available locally at `http://127.0.0.1:8009/docs` while container `wolfcamp-submission-release-api` is running.

The first Windows clean-copy environment was nested deeply enough that a compiled scikit-learn module reached a 260-character path and could not import. That initial attempt failed. A newly created environment at the shorter project-level `.verify-env` path installed all 116 requirements and passed the clean-copy checks. README recommends extracting to a short Windows directory. Neither environment is included in the upload package.

Non-failing environment warnings may include Starlette/AnyIO's deprecation and Windows Jupyter's selector-thread fallback. These do not change the saved metrics.

## Interpretation limits

The historical holdout was inspected during development. Calendar-label maturity is approximate; source revision history and exact reporting availability are not established. Caller-supplied snapshots must obey the month-six availability rules and study domain. Local Docker verification is established; prospective validation and cloud deployment are not claimed. The submission repository is `https://github.com/thaiphi/wolfcamp-oil-production-forecast`; submit its reviewed commit through the course portal. Portal submission has not been performed by this preparation task.
