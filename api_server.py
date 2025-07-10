import io
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from contextlib import asynccontextmanager
from .config import Config
from .model import VLMModel

# This dictionary will hold the loaded model and tokenizer.
# Using a dictionary is a common pattern for sharing state in FastAPI.
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Loading model and tokenizer ---")
    try:
        vlm_wrapper = VLMModel(weights_path=Config.MODEL_WEIGHTS_PATH)
        ml_models["model"] = vlm_wrapper.model
        ml_models["tokenizer"] = vlm_wrapper.tokenizer
        ml_models["model"].eval()  # Set the model to evaluation mode
        print("--- Model and tokenizer loaded successfully ---")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        # In a real-world scenario, you might want the app to fail fast
        # if the model can't be loaded.
    
    yield
    
    # Clean up on shutdown
    print("--- Clearing model and tokenizer ---")
    ml_models.clear()

app = FastAPI(
    title="Vision for Medics API",
    description="An API to get predictions from a fine-tuned Vision-Language Model for medical imaging.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "Vision for Medics API is running."}

@app.post("/predict/")
async def predict(
    prompt: str = Form(..., description="The text prompt or question for the model."),
    image_file: UploadFile = File(..., description="The medical image to be analyzed.")
):
    """
    Receives an image and a text prompt, and returns the model's generated response.
    """
    if "model" not in ml_models or "tokenizer" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file provided: {e}")

    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    inputs['images'] = [image,]
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, temperature=0.2, max_new_tokens=128, do_sample=True)
    
    input_len = inputs['input_ids'].shape[1]
    response_ids = generated_ids[0][input_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return {"prompt": prompt, "response": response_text}