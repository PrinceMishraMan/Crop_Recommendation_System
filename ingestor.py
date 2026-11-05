import json
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file.")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model='gemini-embedding-2',
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

def ingest():
    client = chromadb.PersistentClient(path="./kisan_rag_db")
    gemini_ef = GeminiEmbeddingFunction(api_key=gemini_api_key)
    collection = client.get_or_create_collection(
        name="agri_treatments",
        embedding_function=gemini_ef,
        metadata={"hnsw:space": "cosine"}
    )
    file_path = r"tnau_knowledge.json"
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    ids = []
    docs = []
    metas = []
    for i, item in enumerate(records):
        ids.append(f"doc_{item['crop'].lower()}_{i}")
        docs.append(
            f"Crop: {item['crop']}\n"
            f"Disease: {item['disease']}\n"
            f"Symptoms: {item['symptoms']}\n"
            f"Control Measures and Dosage: {item['management']}"
        )
        metas.append({
            "crop": item["crop"],
            "disease": item["disease"],
            "source": item["source"]
        })
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"[✓] Successfully embedded and stored {collection.count()} records in ChromaDB using Gemini Embedding 2.")

if __name__ == "__main__":
    ingest()
