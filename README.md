# Quantize Docker Container (CUDA + Python 3.12)

Container for GPU-accelerated LLM quantization work using NVIDIA CUDA 12.8 + cuDNN with Python 3.12 + llmcompressor from vLLM

## Prerequisites

- NVIDIA GPU and recent NVIDIA driver on the host
- NVIDIA Container Toolkit for Docker

## Project layout

- `Dockerfile` — CUDA 12.8 base, installs Python 3.12, installs `requirements.txt`, copies `src/`
- `requirements.txt` — your Python dependencies
- `src/run.py` — quantization script executed on container start
- `.dockerignore` — keeps builds lean

## Adding dependencies

Edit `requirements.txt`. For PyTorch with CUDA 12.x, choose the matching wheel index. Example for cu121 (adjust to your desired cu12x tag):

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121
transformers==4.43.*
accelerate==0.33.*
```

## Build

```bash
docker build -t <YOUR IMAGE:TAG> .
```

## Run (with GPU)

```bash
docker run --rm -it \
	--gpus all \
	--ipc=host \
	-e HF_TOKEN="$HF_TOKEN" \
	-e HF_MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct" \
	-e HF_MODEL_TOKENIZER="Qwen/Qwen2.5-Coder-0.5B-Instruct" \
	-e QUANT_MODEL_NAME="Qwen2.5-Coder-0.5B-Instruct-INT8" \
	realityenigma/llmcompressor-quantize:cuda-w8a8
```

Container entrypoint

This image runs `python -u src/app.py` on start (RunPod-compatible). `app.py` simply imports `run.py`, which contains the quantization logic and then exits.

```
SAVE_DIR = "Qwen3-Coder-30B-A3B-Instruct-INT8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Provide the required environment variables so `run.py` can load and quantize your model, then save and push it.

## Environment variables

- `HF_TOKEN` — Hugging Face access token (required for private/gated repos). Used by `huggingface_hub.login()`.
- `HF_MODEL` — Model repo ID to load (e.g., `Qwen/Qwen2.5-Coder-0.5B-Instruct`).
- `HF_MODEL_TOKENIZER` — Tokenizer repo ID to load (often same as `HF_MODEL`).
- `QUANT_MODEL_NAME` — Output folder and Hub repo name for the quantized model (e.g., `Qwen2.5-Coder-0.5B-Instruct-INT8`).
- `HF_HOME` — Optional cache dir for Hugging Face data (default set in image).

## Using a .env file

Copy `.env.example` to `.env` and fill in the values, then export and run:

```bash
set -a; source .env; set +a
docker run --rm -it \
	--gpus all \
	--ipc=host \
	-e HF_TOKEN="$HF_TOKEN" \
	-e HF_MODEL="$HF_MODEL" \
	-e HF_MODEL_TOKENIZER="$HF_MODEL_TOKENIZER" \
	-e QUANT_MODEL_NAME="$QUANT_MODEL_NAME" \
	-v "$PWD":/workspace -w /workspace \
	quantize-cuda
```

## Notes

- Base image: CUDA 12.8 developer. Change the tag if you need a different CUDA.
- Ensure your host driver is compatible with the CUDA version in the image.
- For PyTorch versions and matching cu12x wheels, refer to: https://download.pytorch.org/whl/torch_stable.html
