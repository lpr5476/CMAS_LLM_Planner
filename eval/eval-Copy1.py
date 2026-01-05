#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cmas_lev_checkpoint.py

Evaluate a base LLM and a LoRA-tuned LLM on a CMAS MR→goal test set.

Features:
- Greedy decoding (deterministic).
- Computes Levenshtein distance + normalized similarity to gold goals.
- Saves per-example predictions to JSONL for each model.
- Checkpoint / resume: if a predictions file already exists,
  we reuse previously computed examples and skip them.
- Logs progress, summaries, and (optionally) some gold/pred pairs.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===================== HARD-CODED SETTINGS =====================

# Base model directory (same as in training script)
BASE_MODEL_DIR = "/home/lpr5476/base_llm/"

# LoRA adapter directory (where trainer saved adapter)
LORA_DIR       = "./lora_tuned_model"

# Test set file (JSONL, each line has {"id":..., "messages":[...]} with gold assistant)
TEST_JSONL     = "cmas_lora_test.jsonl"

# Generation settings
MAX_SEQ_LEN       = 4096   # for prompt tokenization
MAX_NEW_TOKENS    = 4096 # cap for generation length

# Use greedy decoding (deterministic, no sampling)
DO_SAMPLE         = False  # must be False for pure greedy
TEMPERATURE       = 1.0    # ignored if DO_SAMPLE=False
TOP_P             = 1.0    # ignored if DO_SAMPLE=False

# Precision / quantization (keep consistent with trainer)
USE_BF16      = True
USE_FP16      = False and not USE_BF16
LOAD_IN_4BIT  = False  # set True only if you used QLoRA + bitsandbytes

# Limit eval to first N examples (None = evaluate all)
MAX_EXAMPLES  = None

# Where to save per-model prediction files + metrics
PREDICTIONS_PREFIX = "cmas_eval"   # -> cmas_eval_base_predictions.jsonl, etc.
METRICS_JSON       = f"{PREDICTIONS_PREFIX}_metrics.json"
LOG_FILE           = "cmas_eval.log"

# Checkpoint / resume behavior
RESUME_FROM_CHECKPOINT = True # reuse existing prediction JSONL to skip done examples

# Logging detail for individual examples
LOG_EACH_EXAMPLE        = True     # whether to log some examples' GOLD/PRED
PRINT_FIRST_N_EXAMPLES  = 10       # how many examples to log in full
TRUNCATE_PREVIEW_CHARS  = 400      # char limit for GOLD/PRED preview in logs

# Offline flags (HPC-friendly)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ===============================================================
# ------------------------- LOGGING -----------------------------
# ===============================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

# ===============================================================
# ---------------------- DATA LOADING --------------------------
# ===============================================================


def load_test_data(path: str) -> List[Dict[str, Any]]:
    """Load chat-style test data from JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if not msgs or len(msgs) < 2:
                continue
            ex_id = obj.get("id", str(i))
            examples.append({"id": ex_id, "messages": msgs})
    return examples


# ===============================================================
# ------------------------ METRICS -----------------------------
# ===============================================================


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # ensure n >= m
    if n < m:
        a, b = b, a
        n, m = m, n

    previous_row = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        current_row = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = previous_row[j] + 1
            delete_cost = current_row[j - 1] + 1
            replace_cost = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[m]


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """
    1 - (levenshtein / max_len), in [0,1].
    1.0 = identical strings, 0.0 = completely different (at max_len edits).
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max_len


# ===============================================================
# ------------------ PROMPT / GENERATION -----------------------
# ===============================================================


def build_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Build prompt text from all messages except the final assistant message,
    using the tokenizer's chat template if available.
    """
    if len(messages) < 2:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            buf = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                buf.append(f"[{role.upper()}]\n{content}\n")
            return "\n".join(buf).strip()

    prompt_messages = messages[:-1]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        buf = []
        for m in prompt_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            buf.append(f"[{role.upper()}]\n{content}\n")
        prompt_text = "\n".join(buf).strip()
    return prompt_text


def generate_answer(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device: Optional[torch.device] = None,
) -> str:
    """
    Given chat messages (with gold assistant included),
    build prompt from messages[:-1] and generate model's assistant output
    using greedy decoding.
    """
    prompt_text = build_prompt_text(tokenizer, messages)
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    if device is None:
        device = next(model.parameters()).device

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,              # False => greedy
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # The generated sequence includes the prompt + new tokens.
    full_ids = outputs[0]
    gen_ids = full_ids[input_ids.shape[1]:]  # strip prompt tokens
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ===============================================================
# --------------------- EVAL W/ CHECKPOINT ---------------------
# ===============================================================


def eval_model_on_examples(
    tag: str,
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run inference on test examples and compute Levenshtein metrics.

    Checkpoint / resume:
    - If RESUME_FROM_CHECKPOINT is True and the predictions file for this tag
      already exists, we:
        * load previous rows,
        * initialize metrics from them,
        * skip examples whose IDs are already present,
        * append new results to the same file.
    """
    logger.info(f"=== Evaluating model: {tag} ===")
    start_time = time.time()
    model.eval()

    pred_file = f"{PREDICTIONS_PREFIX}_{tag.lower()}_predictions.jsonl"

    # ----- Load existing predictions if resuming -----
    per_example: List[Dict[str, Any]] = []
    done_ids = set()
    total_dist = 0.0
    total_norm_sim = 0.0
    n = 0

    if RESUME_FROM_CHECKPOINT and os.path.exists(pred_file):
        logger.info(f"[{tag}] Resuming from existing predictions file: {pred_file}")
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex_id = row.get("id")
                if ex_id is None:
                    continue
                done_ids.add(ex_id)
                per_example.append(row)
                d = float(row.get("levenshtein", 0.0))
                s = float(row.get("norm_similarity", 0.0))
                total_dist += d
                total_norm_sim += s
                n += 1

        logger.info(
            f"[{tag}] Loaded {n} existing predictions from checkpoint. "
            f"Will skip those IDs and continue with remaining examples."
        )
    else:
        logger.info(f"[{tag}] No existing predictions found (or resume disabled). Starting fresh.")

    # Optionally limit examples after we know what's done
    if max_examples is not None:
        logger.info(f"[{tag}] MAX_EXAMPLES is set to {max_examples}.")
        examples = examples[:max_examples]

    # ----- Open predictions file (append or write) -----
    file_mode = "a" if (RESUME_FROM_CHECKPOINT and os.path.exists(pred_file)) else "w"
    pred_f = open(pred_file, file_mode, encoding="utf-8")

    device = next(model.parameters()).device

    # ----- Main loop over examples -----
    for ex in examples:
        ex_id = ex["id"]

        # Skip if already evaluated
        if ex_id in done_ids:
            continue

        messages = ex["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            continue

        gold = messages[-1].get("content", "")

        pred = generate_answer(model, tokenizer, messages, device=device)

        dist = levenshtein(pred, gold)
        norm_sim = normalized_levenshtein_similarity(pred, gold)

        total_dist += dist
        total_norm_sim += norm_sim
        n += 1

        record = {
            "id": ex_id,
            "gold": gold,
            "pred": pred,
            "levenshtein": dist,
            "norm_similarity": norm_sim,
        }
        per_example.append(record)

        # Write to file immediately (checkpoint)
        pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        pred_f.flush()

        # Optional logging for first N examples
        if LOG_EACH_EXAMPLE and n <= PRINT_FIRST_N_EXAMPLES:
            gold_preview = gold.replace("\n", " ")[:TRUNCATE_PREVIEW_CHARS]
            pred_preview = pred.replace("\n", " ")[:TRUNCATE_PREVIEW_CHARS]
            logger.info(f"[{tag}] Example {n} | id={ex_id} | lev={dist} | norm_sim={norm_sim:.3f}")
            logger.info(f"[{tag}]   GOLD: {gold_preview}")
            logger.info(f"[{tag}]   PRED: {pred_preview}")

        if n % 20 == 0:
            logger.info(f"[{tag}] Processed {n} examples so far (including resumed).")

    pred_f.close()

    # ----- Finalize metrics -----
    if n == 0:
        logger.warning(f"[{tag}] No valid examples evaluated.")
        return {"tag": tag, "n_examples": 0, "predictions_file": pred_file}

    avg_dist = total_dist / n
    avg_norm_sim = total_norm_sim / n

    elapsed = time.time() - start_time
    logger.info(f"[{tag}] EVALUATION COMPLETE in {elapsed:.1f}s")
    logger.info(f"[{tag}] Examples evaluated (including resumed): {n}")
    logger.info(f"[{tag}] Avg Levenshtein distance:       {avg_dist:.3f}")
    logger.info(f"[{tag}] Avg normalized similarity (1-d/max): {avg_norm_sim:.3f}")
    logger.info(f"[{tag}] Predictions file: {pred_file}")

    # Top-5 worst examples by Levenshtein
    per_example_sorted = sorted(per_example, key=lambda r: r["levenshtein"], reverse=True)
    worst_k = per_example_sorted[:5]
    logger.info(f"[{tag}] Top-5 worst examples by Levenshtein:")
    for w in worst_k:
        logger.info(
            f"  id={w['id']} | lev={w['levenshtein']} | norm_sim={w['norm_similarity']:.3f}"
        )

    sample_outputs = per_example[:5]

    return {
        "tag": tag,
        "n_examples": n,
        "avg_levenshtein": avg_dist,
        "avg_norm_similarity": avg_norm_sim,
        "samples": sample_outputs,
        "predictions_file": pred_file,
    }


# ===============================================================
# ------------------ MODEL LOADING HELPERS ----------------------
# ===============================================================


def load_base_model_and_tokenizer():
    """Load base model + tokenizer with same settings as trainer."""
    logger.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    logger.info("Loading base model...")
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else None),
            trust_remote_code=True,
        )

    return base, tok


# ===============================================================
# ----------------------------- MAIN ----------------------------
# ===============================================================


def main():
    logger.info("==========================================")
    logger.info("Starting CMAS evaluation (greedy + checkpoint)…")
    logger.info(f"BASE_MODEL_DIR = {BASE_MODEL_DIR}")
    logger.info(f"LORA_DIR       = {LORA_DIR}")
    logger.info(f"TEST_JSONL     = {TEST_JSONL}")
    logger.info(f"RESUME_FROM_CHECKPOINT = {RESUME_FROM_CHECKPOINT}")
    logger.info("==========================================")

    # Load test data
    logger.info(f"Loading test data from: {TEST_JSONL}")
    examples = load_test_data(TEST_JSONL)
    logger.info(f"Total test examples loaded: {len(examples)}")
    if MAX_EXAMPLES is not None:
        logger.info(f"MAX_EXAMPLES is set to {MAX_EXAMPLES} (will slice examples).")

    # ----- Evaluate BASE model -----
    base_model, tokenizer = load_base_model_and_tokenizer()
    base_metrics = eval_model_on_examples(
        tag="BASE",
        model=base_model,
        tokenizer=tokenizer,
        examples=examples,
        max_examples=MAX_EXAMPLES,
    )
    del base_model
    torch.cuda.empty_cache()

    # ----- Evaluate LoRA model -----
    logger.info("Loading LoRA model for evaluation...")
    lora_base, _ = load_base_model_and_tokenizer()
    lora_model = PeftModel.from_pretrained(lora_base, LORA_DIR)
    lora_metrics = eval_model_on_examples(
        tag="LORA",
        model=lora_model,
        tokenizer=tokenizer,
        examples=examples,
        max_examples=MAX_EXAMPLES,
    )

    # Save aggregate metrics to JSON
    all_metrics = {
        "base": base_metrics,
        "lora": lora_metrics,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved aggregate metrics to: {METRICS_JSON}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

