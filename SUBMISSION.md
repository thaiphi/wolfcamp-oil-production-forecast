# Submission checklist

- Dataset and provenance are included with permission to share.
- Open `notebooks/wolfcamp_midterm_submission.ipynb`: the current executed notebook includes all five models' tuning code and outputs inline. `notebook.ipynb` is an identical copy.
- Five regression families, tuned model selection, saved pipeline, metrics and checksum manifest are included.
- Prediction service, pinned dependencies, Dockerfile and automated tests are included.
- A real Docker API request/response screenshot is included in `docs/docker-prediction.png`.
- Verification results and reproduction commands are in `VERIFICATION.md` and `README.md`.
- Optional Render Free deployment configuration and instructions are included. Cloud deployment requires an actual live URL and remote checks; local Render-style Docker checks are recorded separately.

Run `python prepare_submission.py` to create the upload folder and ZIP. Use the **contents** of `dist/wolfcamp-midterm-submission/` as the GitHub repository root, including dotfiles. Environments, caches and duplicate training runs are excluded.

Repository: [thaiphi/wolfcamp-oil-production-forecast](https://github.com/thaiphi/wolfcamp-oil-production-forecast).

Submission steps:

1. Confirm the prepared project files are published in the repository above.
2. Open it while signed out and check that the README, executed notebook, CSV and model are accessible.
3. Submit the repository URL and evaluated commit identifier through the ML Zoomcamp portal before its displayed deadline.
4. Complete the portal's peer-review requirements.

Use the exact published commit you have reviewed when submitting. Portal submission has not been performed by this preparation task.
