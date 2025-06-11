FROM python:3.10.14-slim as trainer

WORKDIR data

COPY requirements.txt train.py Tweets.csv ./

RUN python -m pip install -r requirements.txt && python train.py

FROM scratch

COPY --from=trainer /data/sentiment_model.pkl sentiment_model.pkl
COPY --from=trainer /data/tfidf_vectorizer.pkl tfidf_vectorizer.pkl
