# main.py
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import re

from langdetect import detect
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

app = FastAPI(title="Reviews NLP API", version="2.0")

# =========================
# Sécurité / Auth
# =========================
SECRET_KEY = "U9MWmcqYqSdj3X7n0lRY8675ThJuua9TzQvEnVi4EF57cPokF"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "themes": "Accès à la détection de thèmes",
        "sentiment": "Accès à l’analyse de sentiment",
    },
)

# Base utilisateurs de test (POC)
TEST_users_db = {
    "admin":   {"username": "admin",   "password": "1234",          "scopes": ["themes", "sentiment"]},
    "lecteur": {"username": "lecteur", "password": "test",          "scopes": ["themes"]},
    "wael":    {"username": "wael",    "password": "1@rt1f1c13ll3", "scopes": ["themes", "sentiment"]},
}

def authenticate_user(username: str, password: str):
    user = TEST_users_db.get(username)
    if user and user["password"] == password:
        return user
    return None

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    token = create_access_token(
        {"sub": user["username"], "scopes": user["scopes"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
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
# Modèles & Pipelines NLP
# =========================

# 1) Zero-shot (EN) — plus précis en anglais
zero_shot_en = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Labels + synonymes (meilleure couverture)
LABELS_EN_AUG = [
    "food", "taste", "meal",
    "service", "staff", "friendliness",
    "price", "value for money", "cost",
    "cleanliness", "hygiene",
    "wait time", "speed", "delay",
]
PRIMARY_LABEL_FOR: Dict[str, str] = {
    "food": "food", "taste": "food", "meal": "food",
    "service": "service", "staff": "service", "friendliness": "service",
    "price": "price", "value for money": "price", "cost": "price",
    "cleanliness": "cleanliness", "hygiene": "cleanliness",
    "wait time": "wait_time", "speed": "wait_time", "delay": "wait_time",
}

# 2) ABSA (sentiment par thème) — modèle anglais
ABSA_MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_ID),
    tokenizer=AutoTokenizer.from_pretrained(ABSA_MODEL_ID),
)

# 3) Traduction légère vers EN (Marian), chargée à la demande et cachée
MARIAN_MODELS: Dict[str, str] = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
}
_marian_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}

def _load_marian(model_id: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    if model_id not in _marian_cache:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _marian_cache[model_id] = (tok, mdl)
    return _marian_cache[model_id]

def translate_to_en(text: str, src_lang: str) -> str:
    if not text or src_lang == "en":
        return text
    model_id = MARIAN_MODELS.get(src_lang)
    if not model_id:
        # langue non gérée : renvoyer tel quel (fallback)
        return text
    tok, mdl = _load_marian(model_id)
    enc = tok(text, return_tensors="pt")
    out = mdl.generate(**enc, max_new_tokens=512)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

# =========================
# Helpers (fiabilité)
# =========================
def _clean_text(s: str) -> str:
    if not s:
        return s
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedupe_primary(selected: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Déduplique en gardant le meilleur score par label principal."""
    best: Dict[str, Dict[str, float]] = {}
    for it in selected:
        l = it["label"]
        if l not in best or it["score"] > best[l]["score"]:
            best[l] = it
    return list(best.values())

# =========================
# Schémas d'entrée
# =========================
class AvisInput(BaseModel):
    texte: str
    # si tu veux forcer la langue au lieu d'auto:
    lang: str = Field(default="auto", description='Langue du texte: "auto" | "fr" | "en" | ...')

class AvisThemeInput(BaseModel):
    texte: str
    theme: str
    lang: str = Field(default="auto", description='Langue du texte: "auto" | "fr" | "en" | ...')

class TranslateIn(BaseModel):
    texte: str
    lang: str = Field(default="auto", description='Langue du texte: "auto" | "fr" | "en" | ...')

# =========================
# Endpoints
# =========================

@app.post("/predict-themes")
def predict_themes(input: AvisInput, user: str = Security(get_current_user, scopes=["themes"])):
    # 1) langue
    lang = input.lang
    text = _clean_text(input.texte)
    if lang == "auto":
        try:
            lang = detect(text)
        except Exception:
            lang = "en"

    # 2) standardiser en EN
    text_en = translate_to_en(text, lang) if lang != "en" else text

    # 3) zero-shot en anglais
    out = zero_shot_en(
        text_en,
        candidate_labels=LABELS_EN_AUG,   # synonymes inclus
        multi_label=True,
        hypothesis_template="This review is about {}."
    )

    # 4) sélection simple et fiable : seuil 0.7 + map vers label principal
    selected = []
    for lbl, sc in zip(out["labels"], out["scores"]):
        if sc >= 0.7:
            primary = PRIMARY_LABEL_FOR.get(lbl, lbl)
            selected.append({"label": primary, "score": round(float(sc), 3)})
    themes = _dedupe_primary(selected)

    return {
        "utilisateur": user,
        "texte": input.texte,
        "lang_detectee": lang,
        "themes": themes
    }

# Mapping FR->EN pour l'aspect (au cas où le client envoie un thème FR)
THEME_FR_EN = {
    "nourriture": "food",
    "service": "service",
    "prix": "price",
    "propreté": "cleanliness",
    "attente": "wait_time",
}

def to_en_aspect(theme: str) -> str:
    t = (theme or "").strip().lower()
    return THEME_FR_EN.get(t, t)

@app.post("/predict-sentiment-theme")
def predict_sentiment_theme(input: AvisThemeInput, user: str = Security(get_current_user, scopes=["sentiment"])):
    # 1) langue
    lang = input.lang
    text = _clean_text(input.texte)
    if lang == "auto":
        try:
            lang = detect(text)
        except Exception:
            lang = "en"

    # 2) traduction du texte -> EN si nécessaire
    text_en = translate_to_en(text, lang) if lang != "en" else text

    # 3) aspect en anglais pour l'ABSA
    aspect_en = to_en_aspect(input.theme)

    # 4) format attendu par le modèle ABSA
    input_text = f"{text_en} [SEP] aspect: {aspect_en}"
    result = sentiment_classifier(input_text)

    return {
        "utilisateur": user,
        "texte": input.texte,
        "lang_detectee": lang,
        "theme": aspect_en,
        "sentiment": result
    }

@app.post("/translate")
def translate(payload: TranslateIn, user: str = Security(get_current_user, scopes=["sentiment"])):
    lang = payload.lang
    txt = _clean_text(payload.texte)
    if lang == "auto":
        try:
            lang = detect(txt)
        except Exception:
            lang = "en"
    translated = translate_to_en(txt, lang)
    return {"utilisateur": user, "lang_detectee": lang, "texte_original": txt, "texte_en": translated}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "zero_shot_en": True,
            "absa_en": True,
            "translator_marian": True
        }
    }
