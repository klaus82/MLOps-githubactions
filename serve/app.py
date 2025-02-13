from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import asynccontextmanager
import os

# Download stopwords
nltk.download('stopwords')

ml_models = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
    model_path = os.environ.get("MODEL_PATH", "./sentiment_model.pkl")
    vectorizer_path = os.environ.get("VECTORIZER_PATH", "./tfidf_vectorizer.pkl")
    ml_models["model"] = joblib.load(model_path)
    ml_models["vectorizer"] = joblib.load(vectorizer_path)
    yield

    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(title="Sentiment ML API", description="API to predict sentiment", version="1.0",lifespan=lifespan)

# Function to clean text (same as before)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()
    return text

# Request model for input validation
class TextInput(BaseModel):
    text: str

# Define a sentiment prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    transformed_text = ml_models["vectorizer"].transform([cleaned_text])  # Convert to TF-IDF
    prediction = ml_models["model"].predict(transformed_text)[0]  # Make prediction
    sentiment_mapping_reverse = {1: "positive", 0: "neutral", -1: "negative"}
    return sentiment_mapping_reverse[prediction]

# Define FastAPI endpoint
@app.post("/predict")
async def predict(input_data: TextInput):
    sentiment = predict_sentiment(input_data.text)
    return {"text": input_data.text, "predicted_sentiment": sentiment}

@app.get("/")
async def root():
    return {"message":"I'm alive"}
