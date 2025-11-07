 CMAS_LLM_Planner

Turn operator natural-language instructions into CMAS-compatible variants and planning artifacts, using a local LLM*(no cloud dependencies). 
Models tensors live in `LLM_Files/` and are ignored by Git as they are too big

## What this project does 

1) **Parse a CMAS model** into a compact, NLP-friendly context.
2) **Generate instruction variants** (humanized, paraphrases, scrambled, ungrammatical, Swedish, nonsense) using a local LLM.
3) **Assemble datasets** (JSON/JSONL) for fine-tuning and evaluation (with/without context).
4) **Train** a LoRA adapter on Llama 3:8B local model and evaluate results.
5) **Merges** the produced goals back into the CMAS file for simulation/execution.

# CMAS_LLM_Planner – File Structure (Brief Descriptions)
```
CMAS_LLM_Planner/
├─ README.md                      — Project overview and usage
├─ .gitignore                     — Ignores models/binaries and build cruft
├─ LLM_Files/                     — (ignored) place local LLM models here (e.g., model.gguf)
│
├─ extractor/
│  └─ card_extractor_updated.py   — Parse .cmas → NLP-friendly context/goals
│
├─ data/
│  ├─ cmas_files/                 — Source CMAS models
│  │  ├─ drill_a.cmas             — Drill scenario CMAS file
│  │  ├─ kitting.cmas             — Kitting scenario CMAS file
│  │  └─ llm_crane.cmas           — Crane scenario CMAS file
│  │
│  └─ input_output/               — Generated artifacts & datasets
│     ├─ generated_goals_crane.json            — Goals extracted from CMAS
│     ├─ generated_goals_crane_expanded.json   — Goals + instruction variants
│     ├─ generated_goals_crane_randomized.json — Extra randomized variants (if used)
│     ├─ lora_cmas_no_context.train.jsonl      — LoRA training set (instruction only)
│     ├─ lora_cmas_no_context.test.jsonl       — LoRA test set (instruction only)
│     ├─ lora_cmas_with_context.train.jsonl    — LoRA training set (with CMAS context)
│     └─ lora_cmas_with_context.test.jsonl     — LoRA test set (with CMAS context)
│
├─ data_generators/
│  ├─ instruct_variants.py        — Generate linguistic variants via local LLM
│  ├─ generate_goal_variants_crane.py  — Pipeline: crane-specific goal variants
│  ├─ generate_goal_variants_drill.py  — Pipeline: drill-specific goal variants
│  ├─ expand_crane_with_variants.py    — Augment crane set with extra variants
│  └─ create_lora_dataset.py      — Build LoRA-ready JSONL (with/without context)
│
├─ cmas_merger/
│  └─ merge_goal_patch.py         — Apply goal patches back into a .cmas file
│
├─ training/
│  ├─ trainer.py                  — LoRA/finetune entry
│  └─ training_notebook.ipynb     — Training experiments notebook
│
└─ eval/
   └─ eval_notebook.ipynb         — Evaluation/analysis notebook
```
