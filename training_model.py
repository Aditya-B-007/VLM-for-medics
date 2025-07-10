import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset  # Using Hugging Face's Dataset for its powerful .map() method
from concurrent.futures import ThreadPoolExecutor
from config import Config
from model import VLMModel
from transformers import Trainer,TrainingArguments
from torch.utils.data import DataLoader
import functools
from torch import nn


# Initialize the model to get access to the correctly configured tokenizer
model = VLMModel()
vlm: nn.Module = model.model
tokenizer = model.tokenizer

# --- 1. Function to load data from a .jsonl file ---
def load_dataset_from_jsonl(jsonl_path):
    image_paths = []
    captions = []
    print(f"Reading data from: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                # Ensure the required keys exist
                if 'image_path' in row and 'text_output' in row:
                    image_paths.append(row['image_path'])
                    captions.append(row['text_output'])
                else:
                    print(f"Warning: Skipping line due to missing 'image_path' or 'text_output': {line.strip()}")
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")

    # Create a dictionary for the Hugging Face Dataset
    data = {'image_path': image_paths, 'caption': captions}
    
    # Convert to a Hugging Face Dataset
    hf_dataset = HFDataset.from_dict(data)
    print(f"Successfully loaded {len(hf_dataset)} records.")
    return hf_dataset

# --- 2. Function to preprocess text data (Your original `preprocess` function) ---
def preprocess_text(samples):
    """
    Tokenizes captions and prepares labels for language model training.

    Args:
        samples (dict): A batch of samples from the Hugging Face dataset.

    Returns:
        A dictionary containing tokenized inputs and labels.
    """
    text = samples['caption']  # We now use the 'caption' column from our dataset
    inputs = tokenizer(text, padding='max_length', max_length=Config.MAX_SEQ_LENGTH, truncation=True)
    
    # Keep the image path for the collate function to use later
    inputs['image_path'] = samples['image_path']

    # Create labels and ignore pad tokens (-100) during loss calculation
    labels = inputs['input_ids']
    labels_t = torch.tensor(labels)
    labels_t.masked_fill_(labels_t == tokenizer.pad_token_id, -100)
    inputs['labels'] = labels_t.tolist()

    return inputs

# --- 3. Custom collate function to load images and batch data ---
def custom_collate_fn(batch):
    """
    Collator for loading images from local paths at runtime and batching data.
    """
    # Convert list of dicts to dict of lists
    inputs = {k: [d[k] for d in batch] for k in batch[0].keys()}

    def load_img(path):
        """Loads a single image from a local file path."""
        try:
            img = Image.open(path).convert("RGB")
            # Basic validation for the loaded image
            if not hasattr(img, 'size') or img.size[:2] <= (1, 1):
                return None
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    # Load images concurrently from the provided paths
    image_paths = inputs['image_path']
    with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        imgs = list(executor.map(load_img, image_paths))

    # Filter out any data points where image loading failed
    valid_indices = [i for i, img in enumerate(imgs) if img is not None]
    
    if not valid_indices:
        return None  # Return None if no images in the batch could be loaded

    # Create the final batch with only the valid items
    input_ids = [inputs['input_ids'][i] for i in valid_indices]
    attention_mask = [inputs['attention_mask'][i] for i in valid_indices]
    labels = [inputs['labels'][i] for i in valid_indices]
    final_imgs = [imgs[i] for i in valid_indices]

    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels),
        'images': final_imgs  # This remains a list of PIL Images
    }

# --- 4. Standard PyTorch Custom Dataset Wrapper ---
class ImageCaptionDataset(Dataset):
    """A simple PyTorch Dataset wrapper for the processed data."""
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# --- Main execution block to run the pipeline ---
if __name__ == '__main__':
    # Use the path from the central config file
    jsonl_file_path = Config.JSONL_FILE_PATH

    if not os.path.exists(jsonl_file_path):
        print(f"Error: The data file was not found at '{jsonl_file_path}'.")
        print("Please update the 'JSONL_FILE_PATH' in 'config.py' to the correct location.")
        exit()

    raw_dataset = load_dataset_from_jsonl(jsonl_file_path)

    # Preprocess the text data using .map(), which calls `preprocess_text`**
    processed_dataset = raw_dataset.map(
        preprocess_text,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    # Wrap the processed data in a PyTorch Dataset**
    train_dataset = ImageCaptionDataset(processed_dataset)
    # This will load images and create batches on the fly.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS
    )

    print("\n--- Starting DataLoader Iteration ---")
    # Iterate through the DataLoader to get batches ready for the model**
    for i, batch in enumerate(train_dataloader):
        if batch is None:
            print(f"Skipping empty batch {i+1} (all images failed to load).")
            continue
        print(f"Batch {i+1} successfully created:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Number of images: {len(batch['images'])}")
    print("\n Integration complete. The DataLoader is ready.")

#----------- Below is the methodology for training the model-------------
    class CustomTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
          self.train_dataset,
          batch_size=self.args.train_batch_size,
          collate_fn=custom_collate_fn,
          shuffle=True,
          drop_last=self.args.dataloader_drop_last,
          num_workers=self.args.dataloader_num_workers,
      )
    training_args = TrainingArguments(
        output_dir=Config.MODEL_WEIGHTS_PATH,#Only for this part I have to define the correct path.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        max_grad_norm=1.0,
        max_steps=1100,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 5e-5},
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=25,
        seed=42,
        dataloader_num_workers=4,
        label_names=["labels"],
        report_to="none",
        dataloader_pin_memory=True,
        fp16=True,
    # half_precision_backend="auto",
)
#Freezing all model parameters except those in the connector
    for p in vlm.parameters():
        p.requires_grad = False
    for p in vlm.connector.parameters():
        p.requires_grad = True
    params=sum((p.numel() for p in model.parameters()))
    trainable=sum((p.numel() for p in model.parameters() if p.requires_grad))
    trainer=CustomTrainer(
        model=vlm,
        tokenizer=model.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
    )

    #---Lastly let us run the trainer command for training the model and saving the model weights------
    trainer.train()
    trainer.save_model(Config.MODEL_WEIGHTS_PATH)