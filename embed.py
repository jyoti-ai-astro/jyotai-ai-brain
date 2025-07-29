import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

print("🕉️  Beginning the sacred embedding ritual...")

# Load our secret keys from a local .env file
load_dotenv()

# --- 1. The Sacred Scroll ---
print("📜 Reading the sacred scroll (Bhagavad Gita)...")
with open("knowledge/bhagavad_gita.txt", "r", encoding="utf-8") as f:
    sacred_text = f.read()

# --- 2. The Divine Scribe ---
# We break the long text into smaller, meaningful verses.
text_splitter = CharacterTextSplitter(
    separator="\n\n", # Split by double newlines (paragraphs)
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
documents = text_splitter.split_text(sacred_text)
print(f"✒️  The scroll has been divided into {len(documents)} sacred verses.")

# --- 3. The Divine Alchemist ---
# We prepare the tool to convert words into divine energy (vectors).
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 4. The Offering to the Divine Library ---
# We will now send these verses to Pinecone to be stored forever.
index_name = "jyotai-brahmin-gpt"

print(f"🔑 Accessing the Divine Library (Pinecone Index: {index_name})...")
# This command connects to Pinecone and creates a new index if it doesn't exist.
# Then, it takes our documents, converts them to vectors using the Alchemist,
# and stores them in our Magical Index.
vectorstore = PineconeVectorStore.from_texts(
    texts=documents, 
    embedding=embeddings_model, 
    index_name=index_name
)

print("✅ VICTORY! The wisdom of the Bhagavad Gita has been embedded.")
print("Brahmin GPT is now one step closer to true enlightenment.")