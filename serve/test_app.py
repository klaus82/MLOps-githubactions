# from fastapi import FastAPI
# from fastapi.testclient import TestClient
# from .app import app


# def test_keep_alive():
#     with TestClient(app) as client:
#         response = client.get("/")
#         assert response.status_code == 200
#         assert response.json() == {"message":"I'm alive"}

# def test_predict():
#     with TestClient(app) as client:
#         response = client.post("/predict", test_app.py
#                             json={  "text": "American Airlines is amazing"})
#         assert response.status_code == 200
#         assert response.json() == {"prediction":[0]}

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from .app import app, predict_sentiment, clean_text

test_client = TestClient(app)

@pytest.fixture
def mock_model_and_vectorizer():
    with patch("app.ml_models", {"model": MagicMock(), "vectorizer": MagicMock()}) as mock_ml_models:
        mock_ml_models["vectorizer"].transform.return_value = [[0.1, 0.2, 0.3]]
        mock_ml_models["model"].predict.return_value = [1]
        yield mock_ml_models

def test_root():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "I'm alive"}

def test_clean_text():
    raw_text = "Hello!!! Check this out: https://example.com @user #hashtag"
    cleaned = clean_text(raw_text)
    assert cleaned == "hello check this out   hashtag"

# def test_predict_sentiment(mock_model_and_vectorizer):
#     sentiment = predict_sentiment("This is a great product!")
#     assert sentiment == "positive"

# def test_predict_endpoint(mock_model_and_vectorizer):
#     response = test_client.post("/predict", json={"text": "This product is amazing!"})
#     assert response.status_code == 200
#     assert response.json() == {"text": "This product is amazing!", "predicted_sentiment": "positive"}

# def test_predict_invalid_request():
#     response = test_client.post("/predict", json={})  # Missing "text" field
#     assert response.status_code == 422