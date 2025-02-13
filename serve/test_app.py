import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from .app import app, predict_sentiment, clean_text

test_client = TestClient(app)


def test_root():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "I'm alive"}

def test_clean_text():
    raw_text = "Hello!!! Check this out: https://example.com @user #hashtag"
    cleaned = clean_text(raw_text)
    assert cleaned == "hello check this out   hashtag"
