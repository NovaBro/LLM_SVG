import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ---------- 1. Model & Tokenizer ----------
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Good balance of size and capability

# QLoRA: load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True,         # Double quantization
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# Prepare model for k-bit training (freezes base, enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

print(f"Model loaded: {MODEL_ID}")
print(f"Model dtype: {model.dtype}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ---------- 2. LoRA Configuration ----------
lora_config = LoraConfig(
    r=16,                     # Rank — start with 16, increase if underfitting
    lora_alpha=32,            # Scaling: effective lr multiplier = alpha/r = 2.0
    lora_dropout=0.05,        # Light regularization
    target_modules=[          # Which linear layers to add LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP (feed-forward)
    ],
    bias="none",              # Don't train biases
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable vs. total parameters
model.print_trainable_parameters()
# Output example: "trainable params: 6,815,744 || all params: 1,549,507,584 || trainable%: 0.4399"

# ---------- 3. Inspecting LoRA Layers ----------
# Let's see what LoRA actually added to the model

print("Layers with LoRA adapters:\n")
for name, module in model.named_modules():
    if "lora" in name.lower() and hasattr(module, 'weight'):
        print(f"  {name}: {module.weight.shape}")

# ---------- 4. Dataset Preparation ----------
# For SVG fine-tuning, each sample is a (prompt, SVG_code) pair formatted for instruction tuning

def format_svg_sample(prompt: str, svg_code: str) -> str:
    """Format a prompt-SVG pair into the chat template expected by the model."""
    messages = [
        {"role": "system", "content": "You are an expert SVG code generator. Generate clean, valid SVG code based on the user's description."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": svg_code},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# Example
sample = format_svg_sample(
    prompt="A red circle centered on a white background",
    svg_code='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
)
print("Formatted training sample:")
print(sample[:500])

import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA = Path('./dataset')

train_df = pd.read_csv(DATA / 'train.csv')

new_prompt = []
for r in tqdm(range(train_df.shape[0])):
    new_prompt.append(format_svg_sample(train_df.loc[r, 'prompt'], train_df.loc[r, 'svg']))
train_df['text'] = new_prompt

from datasets import Dataset
from datasets import load_dataset


train_dataset = load_dataset("csv", data_files="./dataset/train.csv")
train_dataset = Dataset.from_pandas(train_df[['text']])
train_dataset

# ---------- 5. Training Setup (using SFTTrainer from TRL) ----------
from transformers import TrainingArguments

# These are reference training arguments — adjust based on your GPU and dataset size
training_args = TrainingArguments(
    output_dir="./svg-lora-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # Effective batch size = 2 * 8 = 16
    learning_rate=2e-4,                  # LoRA typically uses higher LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                           # Use BF16 mixed precision
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    optim="paged_adamw_8bit",            # Memory-efficient optimizer
    gradient_checkpointing=True,         # Trade compute for memory
    max_grad_norm=0.3,
    report_to="none",                    # Set to "wandb" for experiment tracking
)

print("Training arguments configured.")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# ---------- 6. Training Loop (reference — requires a dataset) ----------
# Uncomment and adapt once you have your SVG dataset loaded

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,         # Your HuggingFace Dataset object
    processing_class=tokenizer,
    # max_length=2048,                  # SVGs can be long — adjust as needed
)

trainer.train()

# Save the LoRA adapter (small file — typically 10-50 MB)
model.save_pretrained("./svg-lora-adapter")
tokenizer.save_pretrained("./svg-lora-adapter")

print("Training loop ready — uncomment and provide your dataset to start training!")

# ---------- 7. Loading and Merging LoRA Weights ----------
# After training, you can either:
#   (a) Load the adapter on top of the base model
#   (b) Merge the adapter into the base model for faster inference

from peft import PeftModel

# Option (a): Load adapter
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
model = PeftModel.from_pretrained(model, "./svg-lora-adapter")

# Option (b): Merge and unload (no overhead at inference time)
# model = model.merge_and_unload()
# model.save_pretrained("./svg-model-merged")

print("After training, merge LoRA weights for zero-overhead inference:")
print("  model = model.merge_and_unload()")
print("  # W' = W + (alpha/r) * B @ A  →  single matrix, no adapter overhead")