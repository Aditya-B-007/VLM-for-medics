# 1. Base Image
# Use an official PyTorch image with CUDA support for GPU acceleration.
# Check https://hub.docker.com/r/pytorch/pytorch/tags for the latest tags.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 2. Set up Environment
WORKDIR /app

# Set environment variables to manage Hugging Face cache inside the container
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED 1

# 3. Install Dependencies
# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Code
# Copy all python files and any other necessary assets.
COPY *.py ./

# 5. Create directory for model weights
# The fine-tuned model will be saved here during training
# and loaded from here during inference.
RUN mkdir -p /app/models

# 6. Set the default command to run the inference script
CMD ["python", "realt_time_prediction.py"]
## 6. Expose the port the API will run on and set the default command. USE ONLY FOR TESTING PURPOSE, FOR NO  OTHER REASON.
#EXPOSE 8000
#CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]