from fastapi import FastAPI
from fastapi.testclient import TestClient
from .app import app


def test_keep_alive():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message":"I'm alive"}

def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", test_app.py
                            json={  "text": "American Airlines is amazing"})
        assert response.status_code == 200
        assert response.json() == {"prediction":[0]}