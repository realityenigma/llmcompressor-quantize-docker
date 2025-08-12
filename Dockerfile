## CUDA + cuDNN developer image for GPU-accelerated workloads
# Using CUDA 12.8.1 developer image with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
		LANG=C.UTF-8 \
		LC_ALL=C.UTF-8 \
		PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	HF_HOME=/root/.cache/huggingface \
	HF_TOKEN="" \
	HF_MODEL="" \
	HF_MODEL_TOKENIZER="" \
	QUANT_MODEL_NAME=""

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		software-properties-common \
	gnupg \
		build-essential \
		git \
		curl \
		ca-certificates \
		libgl1 \
		libglib2.0-0 \
		&& rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
		apt-get update && apt-get install -y --no-install-recommends \
			python3.12 python3.12-venv python3.12-dev \
		&& rm -rf /var/lib/apt/lists/*

# Ensure 'python' points to Python 3.12 and upgrade pip via ensurepip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 2 && \
	/usr/bin/python3.12 -m ensurepip --upgrade && \
	python -m pip install --upgrade pip

WORKDIR /opt/app

# Install Python dependencies (cached separately)
COPY requirements.txt ./
RUN if [ -s requirements.txt ]; then \
			python -m pip install -r requirements.txt; \
		else \
			echo "requirements.txt empty, skipping install"; \
		fi

# Copy source after deps for better caching
COPY src/ ./src/

# Entrypoint script: RunPod-compatible wrapper at /workspace/src/app.py
CMD ["python", "-u", "src/run.py"]

# Notes:
# - Use `--gpus all` when running the container and ensure NVIDIA Container Toolkit is installed.
# - Add your Python dependencies to requirements.txt (e.g., transformers, torch, etc.).
# - For PyTorch CUDA wheels, select the appropriate cu12x extra index (e.g., cu124 or cu121) from:
#   https://download.pytorch.org/whl/torch_stable.html
