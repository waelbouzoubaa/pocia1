from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Modèle 1 : Zero-shot multilingue
zero_shot_classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
LABELS = ["service", "nourriture", "prix", "propreté"]

# Modèle 2 : Sentiment (inchangé pour l’instant)
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class AvisInput(BaseModel):
    texte: str

@app.post("/predict-themes")
def predict_themes(input: AvisInput):
    result = zero_shot_classifier(
        input.texte,
        LABELS,
        multi_label=True,
        hypothesis_template="Ce commentaire concerne {}."
    )
    themes = [
        {"label": label, "score": round(score, 3)}
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.8
    ]
    return {"texte": input.texte, "themes_detectés": themes}

@app.post("/predict-sentiment")
def predict_sentiment(input: AvisInput):
    result = sentiment_classifier(input.texte)
    return {"texte": input.texte, "sentiment": result}
