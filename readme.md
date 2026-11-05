# Kisan Mitra 🌾

An AI-driven agricultural advisory platform built to help farmers make data-driven decisions. Kisan Mitra integrates crop recommendation models, computer vision for disease detection, and a RAG pipeline to generate localized, weather-aware treatment prescriptions based on university guidelines.

## 🚀 Key Features
* **Smart Crop Recommendation:** Predicts the best crop to plant based on soil NPK levels, pH, and live weather data via the Open-Meteo API.
* **Plant Disease Detection:** Identifies crop diseases from uploaded leaf images using a custom-trained CNN (deployed via Google's `ai-edge-litert` for maximum efficiency).
* **AI Agronomist (RAG Pipeline):** Retrieves authentic Tamil Nadu Agricultural University (TNAU) treatment guidelines using ChromaDB and Gemini embeddings, then generates safe, weather-specific chemical and organic prescriptions using the Groq LLM.
* **Nearby Agricultural Hubs:** Locates nearby agricultural markets and resources using browser geolocation and the Overpass OpenStreetMap API.

## 🛠️ Tech Stack
* **Backend:** Flask, Python, Gunicorn
* **Machine Learning:** Scikit-Learn, LiteRT (formerly TFLite)
* **Generative AI & RAG:** Groq API (LLM), Google Gemini API (Embeddings), ChromaDB
* **External APIs:** Open-Meteo API, Overpass API
* **Deployment:** Render (Optimized with lazy-loading to operate strictly under the 512MB RAM free-tier limit)

## ⚙️ Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kisan-mitra.git
   cd kisan-mitra
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file (refer to `.env.example`):
   ```env
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   ```
4. Run the Flask development server:
   ```bash
   python app.py
   ```

## 🧠 Architecture Notes for Deployment
To deploy this heavy ML/RAG pipeline on Render's 512MB free tier, this app utilizes a strict **lazy-loading architecture**. Heavy libraries like `chromadb`, `ai_edge_litert`, and `google-genai` are instantiated inside route functions rather than at the global level. This reduces boot-time memory to ~120MB, completely bypassing Gunicorn port timeout failures.
