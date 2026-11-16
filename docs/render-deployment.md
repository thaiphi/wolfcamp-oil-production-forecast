# Deploy the prediction API to Render

This deploys the packaged Random Forest preprocessing/model pipeline using the
same Dockerfile as local serving. It does not retrain the model. The Blueprint
explicitly selects the Free compute plan, one Uvicorn worker, port 10000 and the
model-loading `/health` check. Local Docker still defaults to port 8000.

## Deployment status

**Live:** https://wolfcamp-oil-production-forecast.onrender.com/docs

The service was created from the public Git repository on Render's Free plan
with the settings below. Deployed source commit:
`a6a6899439da8f92cc9fd40c4c4bcdcfa2562a2a`. Automatic deploys are off; later
documentation/evidence commits do not change the running model service.
No payment method was added. All 14 remote HTTP cases and five saved-model
prediction parity checks passed. Remote evidence is in
`reports/verification/render_deployment.json`, `render_http.json` and
`render_parity.json`. See [the actual HTTP 200 screenshot](render-prediction.png).
It uses the minimal six-month-oil input, while the full-feature example in
`examples/request.json` produces a different prediction as expected.
Local resource-limit verification is in `reports/verification/render_local.json`.

## Create the service

1. Sign in to [Render](https://dashboard.render.com/).
2. Choose **New > Blueprint**, connect
   `thaiphi/wolfcamp-oil-production-forecast`, and select the default branch.
3. Render reads the root `render.yaml`. Confirm that the service compute plan
   says **Free** before creating it. No database, disk, payment method or paid
   service is needed by this project.
4. Deploy and wait for the service to become live. Record the actual assigned
   HTTPS URL and deployed GitHub commit from the dashboard; do not assume that
   the service name guarantees a particular URL.
5. Open `<actual-service-url>/health`, then `<actual-service-url>/docs`.
   In Swagger, expand `POST /predict`, choose **Try it out**, and use the
   contents of `examples/request.json` as the body. Execute and confirm HTTP 200.

Alternatively, choose **New > Web Service > Public Git Repository** and paste
`https://github.com/thaiphi/wolfcamp-oil-production-forecast`. Select Docker,
the repository root, Dockerfile `./Dockerfile`, Free compute and health path
`/health`. Add the four environment variables listed in `render.yaml`. Leave
the Docker command override empty so the Dockerfile command is used. Disable
automatic deploys to keep the reviewed submission version stable.

## Verify the public endpoint

The first request after inactivity may take about a minute. Open `/health` in
the browser and wait for `{"status":"ok"}` before running the checks. From the
project root in PowerShell, replace the example URL with the actual assigned URL:

```powershell
$wolfcampUrl = 'https://wolfcamp-oil-production-forecast.onrender.com'
.\.venv\Scripts\python.exe examples/smoke_api.py --url $wolfcampUrl --output reports/verification/render_http.json
.\.venv\Scripts\python.exe examples/verify_api_parity.py --url $wolfcampUrl --output reports/verification/render_parity.json
```

On Linux/macOS use `.venv/bin/python` and the actual URL directly. The first
command checks 14 HTTP cases, including invalid/future inputs. The second checks
five predictions against the packaged model to within 0.01 BBL. Both must pass.
Record the actual URL, deployed commit, check results and an HTTP 200 prediction
screenshot in the README/verification record before claiming cloud deployment.
Repackage with `prepare_submission.py` after adding evidence.

## Free-plan limits

Render Free services sleep after 15 idle minutes and take about a minute to wake.
Filesystem changes are ephemeral; the model is built into the image and no
persistent disk is required. Available free hours, bandwidth and build minutes
are limited. Do not upgrade to a paid plan for this submission. A failed memory
or build check must be reported rather than replaced with an assumed result.

Official references: [Docker](https://render.com/docs/docker),
[Blueprint fields](https://render.com/docs/blueprint-spec),
[Free-plan limits](https://render.com/docs/free).
