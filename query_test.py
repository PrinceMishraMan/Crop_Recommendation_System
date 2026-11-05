# test_rag_retrieval.py
import chromadb
from chromadb.utils import embedding_functions
DB_PATH = "./kisan_rag_db"
client = chromadb.PersistentClient(path=DB_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection(
    name="agri_treatments",
    embedding_function=embedding_fn
)

def test_query(crop: str, query_text: str, top_k: int = 2):
    print(f"\n{'='*70}")
    print(f"Target Crop: {crop}")
    print(f"Query: \"{query_text}\"")
    print(f"{'='*70}")
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where={"crop": crop} if crop else None,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        print("[-] No matching records found.")
        return

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        similarity_score = 1.0 - dist
        print(f"\n--- Match #{idx} [Cosine Similarity: {similarity_score:.4f}] ---")
        print(f"Disease Name: {meta.get('disease')}")
        print(f"Source: {meta.get('source')}")
        print("Content Excerpt:")
        print(doc[:350] + ("..." if len(doc) > 350 else ""))

if __name__ == "__main__":
    print(f"Total documents indexed in collection: {collection.count()}")

    test_query(
        crop="Tomato",
        query_text="Tomato leaf with dark concentric spots, need chemical fungicide and dosage"
    )

    test_query(
        crop="Rice",
        query_text="Spindle-shaped lesions on leaves, organic and chemical control for blast"
    )

    # Test Case 3: Potato Late Blight query
    test_query(
        crop="Potato",
        query_text="Late blight fungal infection high humidity treatment"
    )
