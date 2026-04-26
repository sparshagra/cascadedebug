# CascadeDebug GRPO Training — Hugging Face Space (Docker SDK)
# Python 3.11 + CUDA 12.1 + Unsloth 2025.11.4[cu121-torch251]
# Avoids Gradio SDK’s Python 3.13 stack (Triton/bitsandbytes/Unsloth GRPO issues).
#
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1 wheels — matches base image)
RUN python3.11 -m pip install \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Unsloth + matched xformers / bitsandbytes / triton for this torch build
RUN python3.11 -m pip install "unsloth[cu121-torch251]==2025.11.4"

# Gradio UI + plots (training stack does not depend on Gradio at import time)
RUN python3.11 -m pip install \
    "gradio>=5.25.0" \
    matplotlib \
    scipy \
    huggingface_hub

WORKDIR /app
COPY . /app

EXPOSE 7860

CMD ["python3.11", "app.py"]
