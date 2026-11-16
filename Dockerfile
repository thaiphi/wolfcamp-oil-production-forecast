FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY predict.py .
COPY models/model.joblib ./models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
