"""Serve the persisted full preprocessing + regression pipeline using FastAPI."""
import os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.wolfcamp.data import FEATURES, NUMERIC_FEATURES, CUMULATIVE_6

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "models" / "model.joblib")))
app = FastAPI(title="Wolfcamp Oil Production Forecast", version="1.0.0")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: dict[str, float | int | str | None] = Field(description="Month-6 snapshot; see /features")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}; restore the packaged model or set MODEL_PATH to a trained artifact")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(503, "Model unavailable") from exc
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"required": [CUMULATIVE_6], "allowed": FEATURES,
            "prediction_point": "after 6 months of production", "units": "oil barrels (BBL)",
            "input_contract": "Caller must provide a month-six snapshot. Omit completion attributes when their date is unknown or later than month six. Dates and vendor revisions cannot be verified by this API.",
            "study_domain": "TVD 7000-15000 ft; lateral 4000-16000 ft. Partial snapshots are supported but applicability is caller-verified."}


@app.post("/predict")
def predict(request: PredictionRequest):
    provided = request.features
    extra = sorted(set(provided) - set(FEATURES))
    if extra:
        raise HTTPException(422, f"Future or unknown features not allowed: {extra}")
    six = provided.get(CUMULATIVE_6)
    if not isinstance(six, (float, int)) or not np.isfinite(six) or six < 0:
        raise HTTPException(422, "First6MonthOil_BBL is required as a non-negative number")
    row = {}
    for field in FEATURES:
        value = provided.get(field)
        if field in NUMERIC_FEATURES:
            if value is not None and (not isinstance(value, (float, int)) or not np.isfinite(value)):
                raise HTTPException(422, f"{field} must be a finite number or null")
            if field.startswith("First") and "Month" in field and field != "FirstProdYear" and value is not None and value < 0:
                raise HTTPException(422, f"{field} cannot be negative")
        elif value is not None and not isinstance(value, str):
            raise HTTPException(422, f"{field} must be a string or null")
        # CSV training uses NaN. None is a separate categorical value to sklearn,
        # so normalize JSON nulls/omissions before the fitted imputer sees them.
        row[field] = np.nan if value is None else value
    try:
        estimate = float(load_model().predict(pd.DataFrame([row], columns=FEATURES))[0])
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    increment = max(0.0, estimate)
    return {"predicted_months_7_to_12_oil_bbl": round(increment, 2),
            "predicted_12_month_cumulative_oil_bbl": round(float(six) + increment, 2),
            "actual_first_6_month_oil_bbl": float(six)}
