from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional
from langdetect import detect

app = FastAPI()

# =========================
# Configuration sécurité
# =========================
SECRET_KEY = "U9MWmcqYqSdj3X7n0lRY8675ThJuua9TzQvEnVi4EF57cPokF"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "themes": "Accès à la détection de thèmes",
        "sentiment": "Accès à l’analyse de sentiment"
    }
)

# Simule une base utilisateurs (POC)
TEST_users_db = {
    "admin":   {"username": "admin",   "password": "1234",           "scopes": ["themes", "sentiment"]},
    "lecteur": {"username": "lecteur", "password": "test",           "scopes": ["themes"]},
    "wael":    {"username": "wael",    "password": "1@rt1f1c13ll3",  "scopes": ["themes", "sentiment"]}
}

# =========================
# Fonctions sécurité
# =========================
def authenticate_user(username: str, password: str):
    user = TEST_users_db.get(username)
    if user and user["password"] == password:
        return user
    return None

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    token = create_access_token({"sub": user["username"], "scopes": user["scopes"]},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        token_scopes = payload.get("scopes", [])
        if username is None:
            raise HTTPException(status_code=401, detail="Token invalide")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token incorrect")

    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(status_code=403, detail=f"Permission '{scope}' requise")
    return username

# =========================
# Modèles NLP
# =========================

# Zero-shot multilingue (ok pour détecter les thèmes sans traduire)
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)
LABELS = ["service", "nourriture", "prix", "propreté"]

# ABSA EN (sentiment par thème) – on garde ton setup
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(absa_model_name),
    tokenizer=AutoTokenizer.from_pretrained(absa_model_name)
)

# ===== Traduction mBART-50 (many-to-many) =====
MBART_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"
mbart_tokenizer = AutoTokenizer.from_pretrained(MBART_MODEL_ID)
mbart_model = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL_ID)

# map codes langdetect -> codes mBART
MBART_LANG_MAP = {
    "en": "en_XX", "fr": "fr_XX", "es": "es_XX", "ar": "ar_AR",
    "de": "de_DE", "it": "it_IT", "pt": "pt_XX", "nl": "nl_XX"
    # ajoute d'autres si besoin
}

def translate_to_en(text: str, src_lang: str) -> str:
    src_code = MBART_LANG_MAP.get(src_lang, "en_XX")
    tgt_code = MBART_LANG_MAP["en"]
    mbart_tokenizer.src_lang = src_code
    encoded = mbart_tokenizer(text, return_tensors="pt")
    forced_bos_token_id = mbart_tokenizer.convert_tokens_to_ids(tgt_code)
    generated_tokens = mbart_model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=512
    )
    return mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# =========================
# Schémas Pydantic
# =========================
class AvisInput(BaseModel):
    texte: str

class AvisThemeInput(BaseModel):
    texte: str
    theme: str
    # "auto" = détecte la langue; ou force "fr", "en", "ar", ...
    lang: str = Field(default="auto", description='Langue du texte: "auto" | "fr" | "en" | ...')

# =========================
# Endpoints
# =========================

# 1) Détection de thèmes (zero-shot multilingue)
@app.post("/predict-themes")
def predict_themes(input: AvisInput, user: str = Security(get_current_user, scopes=["themes"])):
    result = zero_shot_classifier(
        input.texte,
        LABELS,
        multi_label=True,
        hypothesis_template="Ce commentaire concerne {}."
    )
    themes = [
        {"label": label, "score": round(score, 3)}
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.9
    ]
    return {"utilisateur": user, "texte": input.texte, "themes_detectés": themes}

# 2) Sentiment par thème (ABSA) avec traduction auto -> EN si nécessaire
@app.post("/predict-sentiment-theme")
def predict_sentiment_theme(input: AvisThemeInput, user: str = Security(get_current_user, scopes=["sentiment"])):
    # a) détection/choix de langue
    lang = input.lang
    if lang == "auto":
        try:
            lang = detect(input.texte)
        except Exception:
            lang = "en"  # fallback

    # b) traduction si pas anglais
    text_for_absa = input.texte
    if lang != "en":
        try:
            text_for_absa = translate_to_en(input.texte, lang)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Échec de traduction: {e}")

    # c) format attendu par le modèle ABSA EN
    input_text = f"{text_for_absa} [SEP] aspect: {input.theme}"
    result = sentiment_classifier(input_text)

    return {
        "utilisateur": user,
        "texte": input.texte,
        "lang_infered": lang,
        "theme": input.theme,
        "sentiment": result
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "zero_shot_multilingual": True,
            "absa_en": True,
            "translator_mbart50": True
        }
    }