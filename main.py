import os
import traceback
from typing import Optional, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# -------------------
# Config
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://jyotai-v2-git-main-jyoti-ais-projects.vercel.app")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# -------------------
# CORS
# -------------------
origins = [
    FRONTEND_URL,
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # ✅ FIXED: Was empty string
    allow_headers=["*"],  # ✅ FIXED: Was empty string
)

# -------------------
# Schema
# -------------------
class PredictionRequest(BaseModel):
    question: str
    dob: Optional[str] = None
    tob: Optional[str] = None
    pob: Optional[str] = None
    gender: Optional[str] = None
    plan: Literal["standard", "premium"] = "standard"

# -------------------
# Routes
# -------------------
@app.get("/")
def health():
    return {"status": "Brahmin GPT is awake and ready."}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY"})

    try:
        system_prompt = (
            "You are Brahmin GPT, a divine sage and Vedic astrologer. "
            "You provide profound, wise, and comforting answers based on ancient wisdom. "
            "Be specific, compassionate, and practical. If DOB/TOB/POB are missing, still respond gracefully."
        )

        # 💡 Extend context later with DOB, TOB, POB, etc.
        user_context = (
            f"Question: {req.question}\n"
            f"DOB: {req.dob or 'N/A'}\n"
            f"TOB: {req.tob or 'N/A'}\n"
            f"POB: {req.pob or 'N/A'}\n"
            f"Gender: {req.gender or 'N/A'}\n"
            f"Plan: {req.plan}\n"
        )

        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
        )

        prediction = chat.choices[0].message.content
        print("🔮 Prediction:", prediction)
        return {"prediction": prediction}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "The divine oracle is resting. Please try again."})
