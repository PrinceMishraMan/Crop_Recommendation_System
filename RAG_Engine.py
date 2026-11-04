# rag_engine.py
import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# 1. Initialize persistent ChromaDB vector store
DB_PATH = "./kisan_rag_db"
client = chromadb.PersistentClient(path=DB_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="agri_treatments",
    embedding_function=embedding_fn
)

# 2. Initialize LLM Client
# Export your key in terminal: export GROQ_API_KEY="your_api_key_here"
# rag_engine.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    # If .env still fails, print out exact search path for debugging
    raise ValueError(f"GROQ_API_KEY is None. Searched in: {BASE_DIR / '.env'}")
groq_client = Groq(api_key=groq_api_key)


def retrieve_disease_context(crop: str, disease: str, top_k: int = 2) -> str:
    """Retrieve authentic TNAU treatment guidelines from ChromaDB."""
    search_query = f"{crop} {disease} fungicide chemical dosage organic control measures"
    
    try:
        results = collection.query(
            query_texts=[search_query],
            n_results=top_k,
            where={"crop": crop} if crop else None,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = results.get("documents", [[]])[0]
        if not docs:
            # Fallback without crop filter if nothing returned
            fallback_results = collection.query(
                query_texts=[search_query],
                n_results=top_k
            )
            docs = fallback_results.get("documents", [[]])[0]
            
        return "\n\n---\n\n".join(docs) if docs else "No specific university advisory found."
    except Exception as e:
        print(f"[-] Vector retrieval error: {e}")
        return "Standard agricultural practices apply."


def generate_prescription(
    crop: str, 
    disease: str, 
    confidence: float, 
    weather: dict, 
    language: str = "English"
) -> str:
    """Combines retrieved context + live weather into a structured farmer prescription."""
    
    context_docs = retrieve_disease_context(crop, disease)

    system_instruction = (
        "You are an expert Chief Agronomist for Kisan Mitra. "
        "Your task is to provide clear, actionable, and safe remedies for farmers. "
        "Strictly adhere to the chemical dosages and remedies present in the Context Documents. "
        "Always tailor your application timing advice to the provided weather conditions."
    )

    user_prompt = f"""
    ### DIAGNOSIS:
    - Crop: {crop}
    - Identified Disease: {disease} (Confidence: {confidence * 100:.1f}%)

    ### LIVE WEATHER AT FARM:
    - Temperature: {weather.get('temp', 'N/A')}°C
    - Humidity: {weather.get('humidity', 'N/A')}%
    - Condition / Forecast: {weather.get('condition', 'Clear')}

    ### RETRIEVED UNIVERSITY GUIDELINES (TNAU):
    \"\"\"
    {context_docs}
    \"\"\"

    ### OUTPUT INSTRUCTIONS:
    Provide an agronomic prescription in {language} following this exact structure:
    1. **Immediate Sanitation** (Physical/cultural actions to prevent spread)
    2. **Chemical Control** (Exact chemical names, concentrations, and dosage per liter from context)
    3. **Organic / Biological Alternative** (Neem formulation, biocontrol agent, or compost extract)
    4. **Weather-Specific Application Warning** (Advice on when to spray based on humidity/rain/temperature)
    """

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        model="openai/gpt-oss-120b",
        temperature=0.15  # Low temperature prevents hallucination
    )

    return response.choices[0].message.content