FROM python:3.10.14-slim

ENV MODEL_PATH /model/model.dat.gz

WORKDIR app

COPY requirements.txt app.py ./

RUN python -m pip install -r requirements.txt

ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]