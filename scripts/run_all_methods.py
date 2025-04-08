# Combined Script to Run All Hallucination Mitigation Methods
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run hallucination mitigation techniques for Llama-3.1-8B-Instruct')
parser.add_argument('--method', type=str, choices=['bpft', 'isc', 'kgro', 'all'], default='all',
                   help='Which hallucination mitigation method to run (default: all)')
parser.add_argument('--train', action='store_true', help='Run training (default: False, only inference)')
parser.add_argument('--prompt', type=str, default="What is the capital of France?",
                   help='Test prompt for model inference')
args = parser.parse_args()

# Check for HF_TOKEN
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN environment variable not set. You may encounter access issues.")
    print("Please set your Hugging Face token: export HF_TOKEN=your_token_here")

# Run BPFT
def run_bpft(train=False, prompt="What is the capital of France?"):
    print("\n" + "="*50)
    print("Running Belief Propagation Fine-Tuning (BPFT)")
    print("="*50)
    
    if train:
        print("Starting BPFT training...")
        # Import and run bpft script
        import bpft_implementation
        # Assuming the script is saved and contains the training code
        
    else:
        # Run inference only
        print("Running BPFT inference...")
        model_path = "./llama-8b-bpft/final"
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please run with --train first or download a pre-trained model")
            return
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

# Run ISC
def run_isc(train=False, prompt="What is the capital of France?"):
    print("\n" + "="*50)
    print("Running Internal State Calibration (ISC)")
    print("="*50)
    
    if train:
        print("Starting ISC training...")
        # Import and run isc script
        import isc_implementation
        # Assuming the script is saved and contains the training code
        
    else:
        # Run inference only
        print("Running ISC inference...")
        model_path = "./llama-8b-isc/final"
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please run with --train first or download a pre-trained model")
            return
        
        # For ISC we need to load our hallucination detector as well
        from isc_implementation import ISCModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create ISC model wrapper
        isc_model = ISCModel(base_model)
        
        # Load hallucination detector weights
        detector_path = f"{model_path}/hallucination_detector.pt"
        if os.path.exists(detector_path):
            isc_model.hallucination_detector.load_state_dict(
                torch.load(detector_path, map_location=base_model.device)
            )
        else:
            print(f"Warning: Hallucination detector weights not found at {detector_path}")
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        outputs = isc_model.model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

# Run KGRO
def run_kgro(train=False, prompt="What is the capital of France?"):
    print("\n" + "="*50)
    print("Running Knowledge-Guided Response Optimization (KGRO)")
    print("="*50)
    
    if train:
        print("Starting KGRO training...")
        # Import and run kgro script
        import kgro_implementation
        # Assuming the script is saved and contains the training code
        
    else:
        # Run inference only
        print("Running KGRO inference...")
        model_path = "./llama-8b-kgro/final"
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please run with --train first or download a pre-trained model")
            return
        
        # For KGRO we need our knowledge retriever and predictor
        from kgro_implementation import KGROModel, KnowledgeRetriever
        
        # Initialize knowledge retriever
        knowledge_retriever = KnowledgeRetriever()
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create KGRO model wrapper
        kgro_model = KGROModel(base_model, knowledge_retriever)
        
        # Load retrieval predictor weights
        predictor_path = f"{model_path}/retrieval_predictor.pt"
        if os.path.exists(predictor_path):
            kgro_model.retrieval_predictor.load_state_dict(
                torch.load(predictor_path, map_location=base_model.device)
            )
        else:
            print(f"Warning: Retrieval predictor weights not found at {predictor_path}")
        
        # Retrieve knowledge for the prompt
        retrieved_knowledge = knowledge_retriever.retrieve(prompt)
        print(f"Retrieved knowledge: {retrieved_knowledge}")
        
        # Format input with knowledge
        knowledge_text = " ".join(retrieved_knowledge)
        full_prompt = f"[KNOWLEDGE] {knowledge_text} [/KNOWLEDGE]\n{prompt}"
        
        # Generate text with KGRO
        inputs = tokenizer(full_prompt, return_tensors="pt").to(base_model.device)
        outputs = kgro_model.model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode and return only the part after the prompt
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = full_output[len(full_prompt):]
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")

# Run the selected method(s)
if args.method == 'bpft' or args.method == 'all':
    run_bpft(train=args.train, prompt=args.prompt)

if args.method == 'isc' or args.method == 'all':
    run_isc(train=args.train, prompt=args.prompt)
    
if args.method == 'kgro' or args.method == 'all':
    run_kgro(train=args.train, prompt=args.prompt)

print("\nAll specified hallucination mitigation methods completed.")