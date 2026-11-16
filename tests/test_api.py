from fastapi.testclient import TestClient
from predict import app, load_model
from src.wolfcamp.data import load_data, split_by_year, FEATURES
import pytest

client = TestClient(app)


def test_api_documentation_and_health():
    assert client.get("/health").json()["status"] == "ok"
    assert "First6MonthOil_BBL" in client.get("/features").json()["required"]


def test_prediction_is_finite_and_cumulative():
    full = load_data()
    _, _, test = split_by_year(full)
    vals = test.iloc[0][FEATURES].where(test.iloc[0][FEATURES].notna(), None).to_dict()
    # Cast numpy scalar types to JSON-friendly Python scalars.
    import pandas as pd
    vals = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k,v in vals.items()}
    reply = client.post("/predict", json={"features": vals})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["predicted_12_month_cumulative_oil_bbl"] >= body["actual_first_6_month_oil_bbl"]


def test_future_data_not_accepted():
    response = client.post("/predict", json={"features": {"First6MonthOil_BBL": 1000, "First12MonthOil_BBL": 2000}})
    assert response.status_code == 422


def test_required_production_not_negative():
    response = client.post("/predict", json={"features": {"First6MonthOil_BBL": -1}})
    assert response.status_code == 422


def test_saved_pipeline_matches_api_prediction():
    import numpy as np
    import pandas as pd
    from src.wolfcamp.data import get_xy
    _, _, test = split_by_year(load_data())
    features = test.iloc[1][FEATURES].to_dict()
    serializable = {k: (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v) for k, v in features.items()}
    response = client.post('/predict', json={'features': serializable})
    assert response.status_code == 200, response.text
    raw_prediction = float(load_model().predict(pd.DataFrame([serializable], columns=FEATURES))[0])
    assert np.isclose(response.json()['predicted_months_7_to_12_oil_bbl'],
                      round(max(0, raw_prediction), 2), atol=0.01)


@pytest.mark.parametrize("optional", [{}, {"County": None, "First3MonthOil_BBL": None}])
def test_partial_snapshot_uses_training_imputation(optional):
    import pandas as pd
    import numpy as np
    supplied = {"First6MonthOil_BBL": 1000, **optional}
    response = client.post("/predict", json={"features": supplied})
    assert response.status_code == 200, response.text
    row = {field: np.nan if supplied.get(field) is None else supplied[field] for field in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    expected = round(max(0, float(load_model().predict(frame)[0])), 2)
    assert response.json()["predicted_months_7_to_12_oil_bbl"] == expected


@pytest.mark.parametrize("field", ["First6MonthOil_BBL", "First3MonthGas_MCF", "FracStages"])
def test_boolean_is_not_a_measurement(field):
    response = client.post("/predict", json={"features": {"First6MonthOil_BBL": 1000, field: True}})
    assert response.status_code == 422


def test_health_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("predict.MODEL_PATH", tmp_path / "missing.joblib")
    load_model.cache_clear()
    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "Model unavailable"
    finally:
        load_model.cache_clear()
