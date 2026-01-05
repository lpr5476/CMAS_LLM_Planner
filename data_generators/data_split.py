#!/usr/bin/env python
"""
make_lora_chat_data_train_test.py

Convert MR dataset (mr_with_vars_aug.jsonl) into chat-style JSONL
for LoRA training on Llama 3, with only TRAIN and TEST splits.

Each output line:
{
  "id": "...",
  "goal_type": "...",
  "messages": [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "<context + goal_type + instruction>"},
    {"role": "assistant", "content": "<goal_canonical JSON>"}
  ]
}
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

SYSTEM_PROMPT = """You are a CMAS goal planner.

Your job is to convert a human operator’s natural-language instruction into a machine-readable CMAS goal JSON, given a textual context that lists agents, parts, skills, interfaces, and variables.

You MUST follow these rules:

1. Output FORMAT:
   - Return exactly ONE JSON object.
   - The top-level schema is:
     {
       "part": <string>,
       "sequence": [
         {
           "skill": <string>,
           "interface": <string>,
           "variables": { ... optional nested dicts ... }
         },
         ...
       ]
     }
   - Use double quotes for all keys and strings.
   - No comments, no trailing commas, no extra text before or after the JSON.

2. Grounding in CONTEXT:
   - Use ONLY resource names, part names, skills, interfaces, and variable names that appear in the provided context or in the training examples.
   - Do NOT invent new agents, skills, interfaces, parts, or variable IDs.
   - Preserve exact spelling and casing from the context (e.g., "MakeAhole", "KitScrew", "CraneInterface").

3. Faithfulness to INSTRUCTION:
   - The sequence must reflect the operator’s requested task: which part to operate on, which skills to call, and in what order.
   - If the instruction implies specific counts, coordinates, or diameters and they appear in the training data, reflect them in the variables.
   - Do NOT add or remove steps that are not implied by the instruction and context.

4. Style:
   - Do NOT explain your reasoning.
   - Do NOT include natural language commentary, Markdown, or code fences.
   - Respond ONLY with the JSON object.
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
    return records


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_chat_example(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one MR record into a chat-style example.

    Input:
      - context
      - goal_type
      - original_instruction

    Output:
      - goal_canonical (as JSON string)
    """
    rec_id = str(rec.get("id", ""))
    goal_type = rec.get("goal_type", "single")
    context = (rec.get("context") or "").strip()
    instr = (rec.get("original_instruction") or "").strip()
    goal = rec.get("goal_canonical")

    if not context or not instr or goal is None:
        raise ValueError(f"Record {rec_id} missing context/instruction/goal_canonical")

    # Assistant output: minified JSON of goal_canonical
    goal_json = json.dumps(goal, ensure_ascii=False)

    user_content = (
        "### CONTEXT\n"
        f"{context}\n\n"
        "### GOAL TYPE\n"
        f"{goal_type}\n\n"
        "### OPERATOR INSTRUCTION\n"
        f"{instr}"
    )

    chat_rec = {
        "id": rec_id,
        "goal_type": goal_type,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": goal_json},
        ],
    }
    return chat_rec


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    seed: int = 42,
):
    rnd = random.Random(seed)
    rnd.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    train = data[:n_train]
    test = data[n_train:]
    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Build chat-style train/test data for CMAS LoRA training."
    )
    parser.add_argument("--input", required=True, help="Input MR JSONL (e.g., mr_with_vars_aug.jsonl)")
    parser.add_argument("--train-out", default="cmas_lora_train.jsonl")
    parser.add_argument("--test-out", default="cmas_lora_test.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Fraction of data to use for training (rest is test).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[INFO] Loading MR records from {args.input}")
    records = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(records)} records")

    chat_records: List[Dict[str, Any]] = []
    for rec in records:
        chat_records.append(build_chat_example(rec))

    print(f"[INFO] Built {len(chat_records)} chat examples")

    train, test = split_train_test(chat_records, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[INFO] Split into train={len(train)}, test={len(test)}")

    save_jsonl(args.train_out, train)
    save_jsonl(args.test_out, test)
    print(f"[INFO] Wrote:\n  {args.train_out}\n  {args.test_out}")


if __name__ == "__main__":
    main()

