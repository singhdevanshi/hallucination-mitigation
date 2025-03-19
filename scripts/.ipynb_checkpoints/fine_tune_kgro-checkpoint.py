# fine_tune_kgro.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrieve_external_knowledge import retrieve_knowledge

# Load model and tokenizer
MODEL_NAME = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

# Define KGRO Loss with Dynamic Weighting
class KGROLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(KGROLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output_logits, target_ids, knowledge_embeddings, response_embeddings):
        base_loss = self.ce_loss(output_logits, target_ids)
        knowledge_consistency = torch.cosine_similarity(knowledge_embeddings, response_embeddings).mean()
        kgro_loss = self.alpha * base_loss - (1 - self.alpha) * knowledge_consistency
        return kgro_loss

def fine_tune_with_kgro(train_texts, knowledge_texts, epochs=3):
    """Fine-tune Mistral 7B with KGRO optimization."""
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    kgro_loss = KGROLoss()

    for epoch in range(epochs):
        for i, text in enumerate(train_texts):
            # Retrieve relevant knowledge
            retrieved_docs = retrieve_knowledge(text)
            knowledge_embedding = embedder.encode(retrieved_docs).mean(axis=0)

            # Prepare input and target
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            target_ids = inputs["input_ids"]

            # Generate model response and compute loss
            outputs = model(**inputs, labels=target_ids)
            logits = outputs.logits

            response_embedding = embedder.encode(text).mean(axis=0)
            loss = kgro_loss(logits, target_ids, knowledge_embedding, response_embedding)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained("/workspace/models/mistral7b_kgro_finetuned")
    tokenizer.save_pretrained("/workspace/models/mistral7b_kgro_finetuned")
    print("KGRO Fine-Tuning Complete!")

if __name__ == "__main__":
    train_texts = [
        "The Eiffel Tower was constructed in 1889.",
        "Isaac Newton discovered gravity in 1687.",
    ]
    fine_tune_with_kgro(train_texts, knowledge_texts)
