from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and vectorizer (Save them after training)
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

clf = joblib.load(MODEL_PATH)  # Load the trained model
vectorizer = joblib.load(VECTORIZER_PATH)  # Load the vectorizer

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
    transformed_text = vectorizer.transform([cleaned_text])  # Convert to TF-IDF
    prediction = clf.predict(transformed_text)[0]  # Make prediction
    sentiment_mapping_reverse = {1: "positive", 0: "neutral", -1: "negative"}
    return sentiment_mapping_reverse[prediction]

# Define FastAPI endpoint
@app.post("/predict")
def predict(input_data: TextInput):
    sentiment = predict_sentiment(input_data.text)
    return {"text": input_data.text, "predicted_sentiment": sentiment}
