import os
import argparse
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/workspace/logs/bpft_pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BPFT_Pipeline")

def run_command(command, description):
    """Run a shell command and log output"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream and log output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        
        # Check for errors
        if process.returncode != 0:
            for line in process.stderr:
                logger.error(line.strip())
            logger.error(f"Failed: {description}")
            return False
        else:
            elapsed_time = time.time() - start_time
            logger.info(f"Completed: {description} in {elapsed_time:.2f} seconds")
            return True
            
    except Exception as e:
        logger.error(f"Exception during {description}: {str(e)}")
        return False

def prepare_directories():
    """Prepare necessary directories"""
    directories = [
        "/workspace/data/bpft",
        "/workspace/models/mistral-7b-bpft",
        "/workspace/results",
        "/workspace/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_pipeline(skip_training=False):
    """Run the entire BPFT pipeline"""
    logger.info("Starting BPFT pipeline")
    
    # Prepare directories
    prepare_directories()
    
    # Step 1: Prepare dataset
    if not run_command("python /workspace/scripts/prepare_bpft_dataset.py", 
                     "Preparing BPFT dataset"):
        logger.error("Dataset preparation failed. Stopping pipeline.")
        return False
    
    # Step 2: Train model (can be skipped)
    if not skip_training:
        if not run_command("python /workspace/scripts/implement_bpft.py",
                         "Training BPFT model"):
            logger.error("BPFT training failed. Evaluation may use base model only.")
    else:
        logger.info("Skipping training as requested.")
    
    # Step 3: Evaluate model
    if not run_command("python /workspace/scripts/evaluate_bpft.py",
                     "Evaluating BPFT model"):
        logger.error("BPFT evaluation failed.")
        return False
    
    logger.info("BPFT pipeline completed successfully")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BPFT pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    args = parser.parse_args()
    
    run_pipeline(skip_training=args.skip_training)