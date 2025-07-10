import os
import sys
import torch
from PIL import Image
from config import Config
from model import VLMModel

def load_model_and_tokenizer(weights_path: str):
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at '{weights_path}'")
        sys.exit(1)

    print("Loading model...")
    vlm_wrapper = VLMModel(weights_path=weights_path)
    model = vlm_wrapper.model
    tokenizer = vlm_wrapper.tokenizer
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer

def run_inference(model, tokenizer, image_path: str, prompt: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return
    except Exception as e:
        print(f"Error opening image file: {e}")
        return
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    inputs['images'] = [image,]
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)
    print("\nGenerating response...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, temperature=0.2, max_new_tokens=64, do_sample=True)
    input_len = inputs['input_ids'].shape[1]
    response_ids = generated_ids[0][input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    print(f"\nModel Response: {response_text}")

def main():
    weights_path = Config.MODEL_WEIGHTS_PATH
    model, tokenizer = load_model_and_tokenizer(weights_path)

    print("\n--- Interactive Real-Time Prediction ---")
    print('Type "quit" or "exit" to stop.')
    while True:
        image_path = input("Enter the path to your image: ").strip()
        if image_path.lower() in ["quit", "exit"]:
            break

        prompt = input("Enter your prompt/question: ").strip()
        if prompt.lower() in ["quit", "exit"]:
            break

        run_inference(model, tokenizer, image_path, prompt)
        print("-" * 40)

if __name__ == "__main__":
    main()