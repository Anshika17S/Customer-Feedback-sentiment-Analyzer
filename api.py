import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from preprocess import clean_text

app = FastAPI()

# Load model once at startup
model = joblib.load("model/sentiment_model.joblib")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    cleaned = clean_text(req.text)
    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = round(float(proba.max()), 4)

    return {
        "text": req.text,
        "sentiment": str(pred),
        "confidence": confidence
    }

@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    if "review" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'review' column")

    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="File has more than 5000 rows")

    cleaned_reviews = df["review"].fillna("").apply(clean_text)
    preds = model.predict(cleaned_reviews)
    probas = model.predict_proba(cleaned_reviews)
    confidences = probas.max(axis=1).round(4)

    has_date = "date" in df.columns
    results = []
    for i in range(len(df)):
        row = {
            "review": str(df["review"].iloc[i]),
            "sentiment": str(preds[i]),
            "confidence": float(confidences[i])
        }
        if has_date:
            row["date"] = str(df["date"].iloc[i])
        results.append(row)

    return results
