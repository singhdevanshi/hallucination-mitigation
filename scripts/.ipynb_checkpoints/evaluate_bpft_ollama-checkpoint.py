import sys
import logging
import ollama
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)

def load_model(model_name):
    try:
        # Ensure the model is pulled and available
        ollama.pull(model_name)
        return model_name
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)

def evaluate_model(model, dataset):
    predictions = []
    true_labels = []

    for data in dataset:
        try:
            prompt = data['input']
            label = data['label']
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            prediction = response['text']
            predictions.append(prediction)
            true_labels.append(label)
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")

    return predictions, true_labels

def compute_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

if __name__ == "__main__":
    model_name = "mistral:7b"
    dataset = [{'input': 'What is AI?', 'label': 'Definition of AI'}]  # Replace with your actual dataset

    model = load_model(model_name)
    predictions, true_labels = evaluate_model(model, dataset)
    accuracy, f1 = compute_metrics(predictions, true_labels)

    logging.info(f"Evaluation Results - Accuracy: {accuracy}, F1: {f1}")
