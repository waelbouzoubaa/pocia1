from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Chargement du modèle zero-shot
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Thèmes possibles
LABELS = ["service", "nourriture", "prix", "propreté"]

class AvisInput(BaseModel):
    texte: str

@app.post("/predict-themes")
def predict_themes(input: AvisInput):
    result = classifier(input.texte, LABELS, multi_label=True)
    themes = [
        {"label": label, "score": round(score, 3)}
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.3
    ]
    return {"texte": input.texte, "themes_detectés": themes}
