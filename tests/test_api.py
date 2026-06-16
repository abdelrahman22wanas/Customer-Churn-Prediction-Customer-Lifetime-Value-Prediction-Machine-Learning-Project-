from fastapi.testclient import TestClient
from app.main import app

VALID_PAYLOAD = {
    "age": 45,
    "monthly_charges": 85.50,
    "total_charges": 1200.00,
    "tenure_months": 24,
    "monthly_usage_gb": 25.0,
    "customer_satisfaction": 7,
    "number_of_services": 3,
    "gender": "male",
    "contract_type": "month-to-month",
    "payment_method": "electronic check",
    "internet_service": "fiber optic",
    "phone_service": "yes",
    "streaming_tv": "yes",
    "streaming_movies": "no",
}


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "classifier" in data
    assert "regressor" in data


def test_predict_churn():
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in ("yes", "no")
    assert "probability" in data
    assert 0 <= data["probability"] <= 1


def test_predict_clv():
    with TestClient(app) as client:
        resp = client.post("/predict/clv", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
    assert "predicted_lifetime_value" in data
    assert isinstance(data["predicted_lifetime_value"], float)


def test_invalid_age_too_young():
    payload = {**VALID_PAYLOAD, "age": 5}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422


def test_invalid_age_too_old():
    payload = {**VALID_PAYLOAD, "age": 200}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422


def test_invalid_enum():
    payload = {**VALID_PAYLOAD, "gender": "alien"}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422


def test_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422


def test_negative_charges():
    payload = {**VALID_PAYLOAD, "monthly_charges": -10}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422


def test_satisfaction_out_of_range():
    payload = {**VALID_PAYLOAD, "customer_satisfaction": 15}
    with TestClient(app) as client:
        resp = client.post("/predict/churn", json=payload)
    assert resp.status_code == 422
