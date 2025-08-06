from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List

app = FastAPI()

# 🔐 Configuration sécurité
SECRET_KEY = "5MWWmcTYqSdj3X7n0lRYOafbsvnscvWTzQvEnVi4EF57cPQkg"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "themes": "Accès à la détection de thèmes",
        "sentiment": "Accès à l’analyse de sentiment"
    }
)

# 🧑‍💻 Simule une base utilisateurs
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "1234",  # version simplifiée
        "scopes": ["themes", "sentiment"]
    },
    "lecteur": {
        "username": "lecteur",
        "password": "test",
        "scopes": ["themes"]
    }
}

# 🔑 Fonctions de sécurité
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
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
    token = create_access_token({
        "sub": user["username"],
        "scopes": user["scopes"]
    })
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

# 📊 Chargement des modèles
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)
LABELS = ["service", "nourriture", "prix", "propreté"]

absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(absa_model_name),
    tokenizer=AutoTokenizer.from_pretrained(absa_model_name)
)

# 📥 Schémas Pydantic
class AvisInput(BaseModel):
    texte: str

class AvisThemeInput(BaseModel):
    texte: str
    theme: str

# 🔒 Endpoint 1 : Détection de thèmes
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
        if score > 0.8
    ]
    return {"utilisateur": user, "texte": input.texte, "themes_detectés": themes}

# 🔒 Endpoint 2 : Analyse de sentiment par thème
@app.post("/predict-sentiment-theme")
def predict_sentiment_theme(input: AvisThemeInput, user: str = Security(get_current_user, scopes=["sentiment"])):
    input_text = f"{input.texte} [SEP] aspect: {input.theme}"
    result = sentiment_classifier(input_text)
    return {
        "utilisateur": user,
        "texte": input.texte,
        "theme": input.theme,
        "sentiment": result
    }
