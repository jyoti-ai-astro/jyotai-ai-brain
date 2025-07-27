import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import openai

# OpenAI key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI()

# ✅ Correct CORS placement BEFORE routes
origins = [
    "https://jyotai-v2-git-main-jyoti-ais-projects.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],        # ✅ Must include OPTIONS
    allow_headers=["*"],        # ✅ Needed for JSON requests
)

# ✅ Prediction request model
class PredictionRequest(BaseModel):
    question: str

# ✅ Prediction route
@app.post("/api/predict")
async def get_prediction(request: PredictionRequest):
    if not openai.api_key:
        return JSONResponse(status_code=500, content={"error": "Missing OpenAI API key"})

    try:
        print("📩 Incoming question:", request.question)

        chat_completion = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are Brahmin GPT, a divine Vedic oracle. Answer with spiritual wisdom.",
                },
                {
                    "role": "user",
                    "content": request.question,
                },
            ]
        )

        prediction = chat_completion.choices[0].message.content
        print("🔮 Prediction:", prediction)
        return {"prediction": prediction}

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse(status_code=500, content={"error": "Prediction failed."})

# ✅ Health check
@app.get("/")
def read_root():
    return {"status": "Brahmin GPT is awake."}
