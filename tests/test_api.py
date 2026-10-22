from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app


class FakeModel:
    def predict(self, data):
        return [250000.0]


def test_health_endpoint():
    with patch("app.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "house price predictin api is running.!!"