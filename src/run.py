from huggingface_hub import login
import os

hf_token = os.environ.get("HF_TOKEN")
hf_model = os.environ.get("HF_MODEL")
hf_model_tokenizer = os.environ.get("HF_MODEL_TOKENIZER")
quant_model_name = os.environ.get("QUANT_MODEL_NAME")

login(token = hf_token)

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = hf_model
tokenizer_id = hf_model_tokenizer

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset="open_platypus",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

SAVE_DIR = quant_model_name
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

model.push_to_hub(quant_model_name, use_temp_dir=False)
tokenizer.push_to_hub(quant_model_name, use_temp_dir=False)
