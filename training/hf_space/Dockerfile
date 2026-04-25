FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /app

# Clone the repo
RUN git clone https://github.com/sparshagra/cascadedebug.git /app

# Install training dependencies
RUN pip install --no-cache-dir \
    torch \
    unsloth \
    trl>=0.12.0 \
    datasets \
    matplotlib \
    numpy \
    huggingface_hub \
    pydantic \
    fastapi \
    uvicorn

# Copy training script (will be overwritten by Space files)
COPY train_entrypoint.sh /app/train_entrypoint.sh
RUN chmod +x /app/train_entrypoint.sh

# Port for status page
EXPOSE 7860

CMD ["/app/train_entrypoint.sh"]
