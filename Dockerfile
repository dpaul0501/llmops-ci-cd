# Use the official PyTorch image, which already includes PyTorch and CUDA (if needed)
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set working directory
WORKDIR /app

# Install the remaining dependencies
RUN pip install --upgrade pip \
    && pip install transformers[torch] accelerate mlflow flask datasets scikit-learn evaluate

# Install supervisord
RUN apt-get update && apt-get install -y supervisor

# Copy your fine-tuning and Flask scripts
COPY finetune.py /app/finetune.py
COPY app.py /app/app.py

# Expose Flask port
EXPOSE 5001

# Run the fine-tuning script first, then start Flask
CMD python3 finetune.py && python3 app.py