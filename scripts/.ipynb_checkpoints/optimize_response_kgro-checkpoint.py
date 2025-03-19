# optimize_response_kgro.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrieve_external_knowledge import retrieve_knowledge
from sentence_transformers import SentenceTransformer

# Load fine-tuned model and tokenizer
MODEL_PATH = "/workspace/models/mistral7b_kgro_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).cuda()

# Load Sentence-BERT for embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

def optimize_response(query, max_length=100, top_k=3):
    """Optimize response by integrating retrieved knowledge."""
    retrieved_docs = retrieve_knowledge(query, top_k)
    knowledge_context = " ".join(retrieved_docs)

    # Add knowledge to the query
    prompt = f"{query} [KNOWLEDGE]: {knowledge_context}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output_ids = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    sample_query = "Who invented the telephone?"
    optimized_response = optimize_response(sample_query)
    print(f"Optimized Response: {optimized_response}")
