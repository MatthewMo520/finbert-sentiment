from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# 1. Load model on startup
app = FastAPI()

model_path = "./model/finbert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# 2. Request schema
class TextInput(BaseModel):
    text: str

# 3. Predict endpoint
@app.post("/predict")
def predict(input: TextInput):
    tokens = tokenizer(
        input.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**tokens)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    predicted_class = int(np.argmax(probs))

    return {
        "text": input.text,
        "sentiment": label_map[predicted_class],
        "confidence": round(float(probs[predicted_class]), 4),
        "scores": {
            "Bearish": round(float(probs[0]), 4),
            "Bullish": round(float(probs[1]), 4),
            "Neutral": round(float(probs[2]), 4)
        }
    }

# 4. Health check
@app.get("/health")
def health():
    return {"status": "ok"}