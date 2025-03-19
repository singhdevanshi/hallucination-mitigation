# retrieve_external_knowledge.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Sentence-BERT for knowledge retrieval
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# Load FAISS knowledge index
INDEX_PATH = "/workspace/data/knowledge_index.faiss"
knowledge_index = faiss.read_index(INDEX_PATH)

# Load stored knowledge texts
knowledge_texts = []
with open("/workspace/data/knowledge_corpus.txt", "r") as f:
    knowledge_texts = f.readlines()

def retrieve_knowledge(query, top_k=3):
    """Retrieve top-k relevant knowledge for a given query."""
    query_embedding = embedder.encode(query)
    query_embedding = np.array([query_embedding]).astype("float32")
    
    _, indices = knowledge_index.search(query_embedding, top_k)
    retrieved_knowledge = [knowledge_texts[i] for i in indices[0]]
    
    return retrieved_knowledge

if __name__ == "__main__":
    sample_query = "When was the Eiffel Tower constructed?"
    retrieved_docs = retrieve_knowledge(sample_query)
    print(f"Retrieved Knowledge:\n{retrieved_docs}")
