import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig
from trl import SFTTrainer

set_seed(42)

MODEL_ID = "PORTULAN/gervasio-8b-portuguese-ptpt-decoder"
NEW_MODEL_NAME = "Gervasio-LoRA"
DATA_FILE = "pt_no_failed_state_cleaned.jsonl"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

dataset = load_dataset("json", data_files = DATA_FILE, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs = 3,
    
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    group_by_length=True,

    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

    bf16=True,
    fp16=False,

    save_strategy="steps",
    save_steps = 100,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,

    logging_steps=10,
    logging_first_step=True,
    report_to="none",

    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=True
)

print("Starting Training...")
trainer.train()

print(f"Saving to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("Done.")