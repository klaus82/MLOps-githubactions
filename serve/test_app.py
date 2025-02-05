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
                            json={  "concavity_mean": 0,
                                    "concave_points_mean": 0,
                                    "perimeter_se":0,
                                    "area_se":0,
                                    "texture_worst":0,
                                    "area_worst":0})
        assert response.status_code == 200
        assert response.json() == {"prediction":[0]}