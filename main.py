from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Modèle 1 : Détection de thèmes (zero-shot multilingue)
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)
LABELS = ["service", "nourriture", "prix", "propreté"]

# Modèle 2 : Sentiment par thème (ABSA)
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(absa_model_name),
    tokenizer=AutoTokenizer.from_pretrained(absa_model_name)
)

class AvisInput(BaseModel):
    texte: str

class AvisThemeInput(BaseModel):
    texte: str
    theme: str

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

@app.post("/predict-sentiment-theme")
def predict_sentiment_theme(input: AvisThemeInput):
    input_text = f"{input.texte} [SEP] aspect: {input.theme}"
    result = sentiment_classifier(input_text)
    return {
        "texte": input.texte,
        "theme": input.theme,
        "sentiment": result
    }
