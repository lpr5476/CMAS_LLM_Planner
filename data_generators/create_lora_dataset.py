#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_lora_datasets.py

Build two LoRA fine-tuning datasets from a CMAS crane dataset.

Inputs:
  - A JSON array file like generated_goals_crane_randomized.json containing items with:
      {
        "instruction": <str>,
        "input": {"context": <str>},
        "output": {"goal_canonical_simple": <str>, ...}
      }
    Items with non-dict output (e.g., "Cannot generate goal...") are skipped.

Outputs (JSONL, one example per line, Chat-style):
  With-context (rules-based system prompt) and No-context (no system prompt),
  each split into train/test → 4 files total:
    1) <with_context>.train.jsonl
    2) <with_context>.test.jsonl
    3) <no_context>.train.jsonl
    4) <no_context>.test.jsonl

  Message shapes:
    - With-context system: see SYSTEM_PROMPT_WITH_CONTEXT
      user:   "Context:\n{context}\n\nOperator instruction:\n{instruction}"
      assistant: goal_canonical_simple

    - No-context system: [none]
      user:   "{instruction}"
      assistant: goal_canonical_simple

Usage:
  python create_lora_datasets.py \
    --input /path/generated_goals_crane_randomized.json \
    --with-context /path/lora_cmas_with_context.jsonl \
    --no-context /path/lora_cmas_no_context.jsonl \
    [--test-ratio 0.1] [--seed 42] \
    [--with-context-train ... --with-context-test ... --no-context-train ... --no-context-test ...]

If explicit train/test paths are not provided, the tool derives them by
inserting ".train"/".test" before the ".jsonl" extension of the provided
"--with-context" and "--no-context" paths.
"""

import json
import random
import argparse
from pathlib import Path

# System prompt for the WITH-CONTEXT dataset: concise but explicit rules.
SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a CMAS goal synthesis assistant. Use the provided Context to craft a single "
    "CMAS goal in the canonical simple form. Follow these rules: "
    "1) Treat Context as authoritative when it conflicts with the instruction. "
    "2) Do not invent tools, parameters, or entities not present or implied by Context. "
    "3) Respect names, units, limits, and capabilities as stated in Context. "
    "4) Output only the goal string (no explanations, quotes, or extra text). "
    "5) Be precise and feasible within the CMAS model assumptions."
)

# Some previous files had typos in the key name—be tolerant
GOAL_KEYS = [
    "goal_canonical_simple",
    "goal_cannoical_simple",
    "goal_cannocial_simple",
]

def extract_goal_simple(output_obj: dict) -> str | None:
    if not isinstance(output_obj, dict):
        return None
    for k in GOAL_KEYS:
        v = output_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def make_record(system_prompt: str | None, user_content: str, assistant_content: str) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_datasets(src_path: Path) -> tuple[list[dict], list[dict], int, int]:
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON array.")

    with_ctx: list[dict] = []
    no_ctx: list[dict] = []
    total = len(data)
    kept = 0

    for row in data:
        if not isinstance(row, dict):
            continue

        instr = (row.get("instruction") or "").strip()
        inp = row.get("input") or {}
        ctx = (inp.get("context") or "").strip() if isinstance(inp, dict) else ""
        goal_simple = extract_goal_simple(row.get("output"))

        if not instr or not goal_simple:
            continue  # skip nonsense/negatives or malformed rows

        # Dataset 1: with context (uses rules-based system prompt)
        user_with = f"Context:\n{ctx}\n\nOperator instruction:\n{instr}"
        with_ctx.append(make_record(SYSTEM_PROMPT_WITH_CONTEXT, user_with, goal_simple))

        # Dataset 2: no context (no system prompt at all, user is raw instruction)
        user_no = instr
        no_ctx.append(make_record(None, user_no, goal_simple))

        kept += 1

    return with_ctx, no_ctx, total, kept

def split_train_test(records: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    cut = int(len(idxs) * (1 - test_ratio))
    train_idx = idxs[:cut]
    test_idx = idxs[cut:]
    train = [records[i] for i in train_idx]
    test = [records[i] for i in test_idx]
    return train, test

def main():
    ap = argparse.ArgumentParser(description="Create LoRA finetuning datasets (with/without context) from CMAS crane data, with train/test split.")
    ap.add_argument("--input",        default="data/input_output/generated_goals_crane_randomized.json", help="Path to source JSON (array).")
    ap.add_argument("--with-context", default="data/input_output/lora_cmas_with_context.jsonl",         help="Base output path for with-context dataset; train/test suffixes are added if explicit paths not given.")
    ap.add_argument("--no-context",   default="data/input_output/lora_cmas_no_context.jsonl",           help="Base output path for no-context dataset; train/test suffixes are added if explicit paths not given.")
    ap.add_argument("--with-context-train", type=str, help="Explicit path for with-context train JSONL.")
    ap.add_argument("--with-context-test",  type=str, help="Explicit path for with-context test JSONL.")
    ap.add_argument("--no-context-train",   type=str, help="Explicit path for no-context train JSONL.")
    ap.add_argument("--no-context-test",    type=str, help="Explicit path for no-context test JSONL.")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of data to use for test split (0.0-1.0). Default: 0.1")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before split. Default: 42")
    args = ap.parse_args()

    src = Path(args.input)

    # Build full datasets
    with_ctx, no_ctx, total, kept = build_datasets(src)

    # Split into train/test
    train_with, test_with = split_train_test(with_ctx, args.test_ratio, args.seed)
    train_no, test_no     = split_train_test(no_ctx, args.test_ratio, args.seed)

    # Resolve output paths
    def derive_pair(base: Path, override_train: str | None, override_test: str | None) -> tuple[Path, Path]:
        if override_train and override_test:
            return Path(override_train), Path(override_test)
        stem = base.name
        if stem.endswith(".jsonl"):
            core = stem[:-6]
            ext = ".jsonl"
        else:
            core = stem
            ext = ""
        train_path = base.with_name(core + ".train" + ext)
        test_path  = base.with_name(core + ".test" + ext)
        return Path(override_train) if override_train else train_path, Path(override_test) if override_test else test_path

    base_with = Path(args.with_context)
    base_no   = Path(args.no_context)
    out_with_train, out_with_test = derive_pair(base_with, args.with_context_train, args.with_context_test)
    out_no_train,   out_no_test   = derive_pair(base_no,   args.no_context_train,   args.no_context_test)

    # Write outputs
    write_jsonl(out_with_train, train_with)
    write_jsonl(out_with_test,  test_with)
    write_jsonl(out_no_train,   train_no)
    write_jsonl(out_no_test,    test_no)

    print(f"Source rows:       {total}")
    print(f"Kept rows:         {kept}")
    print(f"With-context:      train={len(train_with)}  test={len(test_with)} -> {out_with_train} , {out_with_test}")
    print(f"No-context:        train={len(train_no)}    test={len(test_no)}   -> {out_no_train} , {out_no_test}")

if __name__ == "__main__":
    main()
