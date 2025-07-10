import os
import torch

class Config:
    """
    A central configuration class for the VLM project.
    It stores model identifiers, hyperparameters, and file paths.
    """
    # --- Model & Tokenizer Identifiers ---
    LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    VISION_ID = "google/siglip-so400m-patch14-384"

    # --- Model & Preprocessing Hyperparameters ---
    VISION_TOWER_R = 9          # Number of adjacent tokens to concatenate in the vision tower
    MAX_SEQ_LENGTH = 128        # Max sequence length for text tokenization
    BATCH_SIZE = 16             # Batch size for the DataLoader
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 0  # For DataLoader
    JSONL_FILE_PATH = os.getenv('JSONL_FILE_PATH', '<enter the path to the jsonl file that was created by the previous script>')

    # Path to save or load fine-tuned model weights.
    # The model.py script will look for weights here if a path is provided.
    MODEL_WEIGHTS_PATH = os.getenv('MODEL_WEIGHTS_PATH', "models/fine_tuned_vlm.pth")  # Please do enter the path when it is saved after training
    INPUT_PROMPT_IMAGE = "PATH"
    INPUT_PROMPT_TEXT = "TEXT"