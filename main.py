# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai

# --- Configuration ---
# NOTE: We are NOT putting the key here. We will set it in Render's environment.
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# This defines the structure of the incoming request
class PredictionRequest(BaseModel):
    question: str
    # We will add more fields like DOB, name, etc., later.

# This is our main prediction endpoint
@app.post("/api/predict")
async def get_prediction(request: PredictionRequest):
    if not openai.api_key:
        return {"error": "OpenAI API key is not configured."}

    try:
        chat_completion = await openai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Brahmin GPT, a divine sage and Vedic astrologer. You provide profound, wise, and comforting answers based on ancient wisdom. Your tone is respectful and full of spiritual insight.",
                },
                {
                    "role": "user",
                    "content": request.question,
                },
            ],
            model="gpt-4o",
        )
        
        prediction = chat_completion.choices[0].message.content
        return {"prediction": prediction}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "Failed to get a prediction from the divine oracle."}

# A simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "Brahmin GPT is awake."}