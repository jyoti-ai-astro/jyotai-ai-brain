import os
import traceback
from typing import Optional, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
# --- THIS IS OUR UPGRADE: We import the tools to read from our Divine Library ---
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# --- END OF UPGRADE ---

# -------------------
# Config (Your superior version)
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
PINECONE_INDEX_NAME = "jyotai-brahmin-gpt" # The name of our sacred library
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://jyotai-v2-git-main-jyoti-ais-projects.vercel.app")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# -------------------
# CORS (Your superior version)
# -------------------
origins = [
    FRONTEND_URL,
    "https://jyotai-v2.vercel.app",
    "http://localhost:3000",
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -------------------
# Schema (Your superior version)
# -------------------
class PredictionRequest(BaseModel):
    question: str
    dob: Optional[str] = None
    name: Optional[str] = None # Added name to be consistent with frontend

# -------------------
# RAG System Setup (The Divine Connection)
# -------------------
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model)

# -------------------
# Routes
# -------------------
@app.get("/")
def health():
    return {"status": "Brahmin GPT is awake and enlightened."}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY"})

    try:
        # --- THIS IS THE UPGRADE: We now search for wisdom before asking the AI ---
        print(f"🔎 Searching the sacred library for wisdom related to: '{req.question}'")
        relevant_verses = vectorstore.similarity_search(req.question, k=4) # Find the 4 most relevant verses
        context = "\n\n".join([doc.page_content for doc in relevant_verses])
        print(f"📚 Found {len(relevant_verses)} relevant verses.")
        # --- END OF UPGRADE ---

        system_prompt = (
            "You are Brahmin GPT, a divine sage and Vedic astrologer. "
            "You provide profound, wise, and comforting answers. "
            "Crucially, you MUST base your answer PRIMARILY on the following sacred verses provided from the Bhagavad Gita. "
            "Weave these timeless truths into your response to the user's question. Address the user by their name."
        )

        user_context = (
            f"SACRED CONTEXT FROM THE BHAGAVAD GITA:\n---\n{context}\n---\n\n"
            f"User's Name: {req.name or 'Seeker'}\n"
            f"User's DOB: {req.dob or 'Not provided'}\n"
            f"User's Question: {req.question}"
        )

        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context},
            ],
        )

        prediction = chat.choices[0].message.content
        return {"prediction": prediction}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "The divine oracle is resting. Please try again."})