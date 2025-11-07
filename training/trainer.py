#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Hardcoded LoRA fine-tuning for chat-style CMAS goal generation.

- Uses a fixed base model path
- Uses a fixed training dataset (change TRAIN_JSONL below to switch)
- Uses fixed LoRA + TrainingArguments (no CLI)

Input format (JSONL; one example per line):
{"messages":[
  {"role":"system","content":"Create a CMAS goal with the following instructor goal and context"},
  {"role":"user","content":"Context:\n...\n\nOperator instruction:\n..."},
  {"role":"assistant","content":"ProductA: transportPart(...); packing"}
]}
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===================== HARD-CODED SETTINGS =====================

# Base model directory (local path with config.json, tokenizer, *.safetensors)
BASE_MODEL_DIR = "/home/lpr5476/base_llm/"   # <-- change if different

# Choose ONE dataset to train on.
#  - With context:   "/mnt/data/lora_cmas_with_context.jsonl"
#  - No context:     "/mnt/data/lora_cmas_no_context.jsonl"
TRAIN_JSONL   = "lora_cmas_with_context.train.jsonl"   # <-- set to no-context if desired

# Output directory (LoRA adapter + tokenizer will be saved here)
OUTPUT_DIR    = "./lora_cmas_with_context"

# Sequence length and split
MAX_SEQ_LEN   = 2048
VAL_FRACTION  = 0.05

# Mixed precision / quantization
USE_BF16      = True
USE_FP16      = False and not USE_BF16
LOAD_IN_4BIT  = False  # set True if you have bitsandbytes installed and want QLoRA

# LoRA configuration (fixed)
LORA_R        = 32
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.05
# For LLaMA-like models these are typical:
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# TrainingArguments (fixed)
EPOCHS        = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.03
WEIGHT_DECAY  = 0.0
LOGGING_STEPS = 10
EVAL_STEPS    = 50
SAVE_STEPS    = 100
SAVE_TOTAL    = 3
GRADIENT_CHECKPOINTING = True

# Offline (recommended on HPC)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ===============================================================


class ChatJsonlDataset(Dataset):
    """Simple chat-style JSONL dataset with `messages`."""
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tok = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if not msgs:
                    continue
                self.samples.append(msgs)
        self.has_chat_template = hasattr(self.tok, "apply_chat_template")

    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        if self.has_chat_template:
            # include assistant content; no generation prompt
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # fallback: simple readable concat
        buf = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                buf.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                buf.append(f"[USER]\n{content}\n")
            elif role == "assistant":
                buf.append(f"[ASSISTANT]\n{content}\n")
        return "\n".join(buf).strip()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx]
        text = self._format_chat(msgs)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,          # collator will pad
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            # NOTE: no 'labels' here; collator will copy input_ids into labels and pad
        }


def main():
    # Tokenizer
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # safer for causal LM padding

    # Base model
    print("Loading base model...")
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        base = prepare_model_for_kbit_training(base)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else None),
            trust_remote_code=True
        )

    # LoRA
    print("Attaching LoRA...")
    lcfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, lcfg)

    # ---- critical for gradient checkpointing with LoRA ----
    model.enable_input_require_grads()
    model.config.use_cache = False  # required with checkpointing
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    # -------------------------------------------------------

    # Dataset + split
    print("Preparing dataset...")
    full_ds = ChatJsonlDataset(TRAIN_JSONL, tok, max_length=MAX_SEQ_LEN)
    n_total = len(full_ds)
    n_val = int(n_total * VAL_FRACTION)
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Dataset: total={n_total}, train={len(train_ds)}, val={len(val_ds)}")

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training args
    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, BATCH_SIZE // 2),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps" if len(val_ds) > 0 else "no",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL,
        report_to="none",
        bf16=USE_BF16,
        fp16=USE_FP16 and not USE_BF16,
        dataloader_num_workers=0,                 # simpler & stable on HPC
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # silence warning & recommended
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        tokenizer=tok,
    )

    print("Starting training…")
    trainer.train()

    print("Saving adapter…")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
