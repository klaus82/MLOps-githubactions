# import pandas as pd
# import numpy as np
# import re
# import string
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# # Load dataset
# df = pd.read_csv("Tweets.csv")

# # Select relevant columns
# df = df[['text', 'airline_sentiment']]

# # Preprocessing function
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     return text

# # Apply preprocessing
# df['clean_text'] = df['text'].apply(preprocess_text)

# # Encode target labels
# df['label'] = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# print(X_test.head())

# # Text vectorization
# vectorizer = TfidfVectorizer()
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train classifier
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# # Predictions
# y_pred = model.predict(X_test_tfidf)

# # Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.4f}')
# print(classification_report(y_test, y_pred))


import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset

df = pd.read_csv("Tweets.csv")

# Select relevant columns
df = df[['text', 'airline_sentiment']]

# Text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove whitespace
    return text

# Apply text cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Convert labels to numerical values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment'] = df['airline_sentiment'].map(sentiment_mapping)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "sentiment_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

