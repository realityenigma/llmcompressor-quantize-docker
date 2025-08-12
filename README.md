# Quantize Docker Container (CUDA + Python 3.12)

Container for GPU-accelerated LLM work using NVIDIA CUDA 12.8 + cuDNN with Python 3.12 and dependencies managed via `requirements.txt`.

## Prerequisites

- NVIDIA GPU and recent NVIDIA driver on the host
- NVIDIA Container Toolkit for Docker

## Project layout

- `Dockerfile` — CUDA 12.8 base, installs Python 3.12, installs `requirements.txt`, copies `src/`
- `requirements.txt` — your Python dependencies
- `src/run.py` — single entry script executed on container start
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
docker build -t quantize-cuda .
```

## Run (with GPU)

```bash
docker run --rm -it \
	--gpus all \
	--ipc=host \
	-e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
	-v "$PWD":/workspace \
	-w /workspace \
	quantize-cuda
```

Inside the container, test PyTorch GPU (if installed):

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Devices:', torch.cuda.device_count())
print('Device 0 name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```

Container entrypoint

This image runs `python -u src/run.py` on start, then exits. The script contains:

```
SAVE_DIR = "Qwen3-Coder-30B-A3B-Instruct-INT8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Provide `model` and `tokenizer` in your environment or code (e.g., via mounted volumes or dependencies) before running the container so the script can save them.

## Hugging Face token

- Set `HUGGINGFACE_HUB_TOKEN` (and optionally `HF_TOKEN`) when running the container:

```bash
export HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
docker run --rm -it --gpus all --ipc=host \
	-e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
	-v "$PWD":/workspace -w /workspace quantize-cuda
```

- Or use a `.env` file (copy `.env.example` to `.env` and fill in):

```bash
set -a; source .env; set +a
docker run --rm -it --gpus all --ipc=host \
	-e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
	-v "$PWD":/workspace -w /workspace quantize-cuda
```

## Notes

- Base image: CUDA 12.8 developer. Change the tag if you need a different CUDA.
- Ensure your host driver is compatible with the CUDA version in the image.
- For PyTorch versions and matching cu12x wheels, refer to: https://download.pytorch.org/whl/torch_stable.html
