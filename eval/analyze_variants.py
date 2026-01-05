#!/usr/bin/env python
"""
analyze_variants_lexical.py

Post-hoc analysis of instruction variants produced by canon.py.

Now includes:
  - Lexical metrics: Distinct-1/2, token-level Jaccard, precision.
  - Semantic similarity: cosine similarity between LLaMA embeddings of
    canonical and variant instructions (no sentence-transformers).

Input: JSONL where each record has:
  - "original_instruction" (str) or
  - "instruction_variants": {
        "normal": ["<canonical>", ...],
        "paraphrase": [...],
        "scrambled": [...],
        "typo": [...],
        "nonsense": [...]
    }

Output JSON (example structure):
{
  "global": {
    "n_records_with_variants": 66
  },
  "modes": {
    "paraphrase": {
      "n_texts": ...,
      "distinct_1": ...,
      "distinct_2": ...,
      "lexical_jaccard_stats": {...},
      "lexical_precision_stats": {...},
      "semantic_similarity_stats": {...}
    },
    ...
  }
}

Usage:
  python analyze_variants_lexical.py \
      --input output/mr_dataset_with_vars.jsonl \
      --output metrics_lexical_semantic.json \
      --model-name-or-path /path/to/Meta-Llama-3-8B-Instruct \
      --device cuda
"""

import argparse
import json
import os
import sys
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------- IO -----------------------------------------------

def load_records_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}")
    return out


# -------------------------- TEXT UTILS ----------------------------------------

def tokenize(text: str) -> List[str]:
    # Simple whitespace tokenizer; good enough for short instructions
    return text.strip().split()


def distinct_n(texts: List[str], n: int) -> float:
    """
    Distinct-n as in Li et al.:
      (# unique n-grams) / (# total n-grams)
    """
    all_ngrams: List[Tuple[str, ...]] = []
    for t in texts:
        toks = tokenize(t)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            all_ngrams.append(tuple(toks[i : i + n]))
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / float(total)


def lexical_overlap(canonical: str, variant: str) -> Tuple[float, float]:
    """
    Simple lexical similarity between canonical and variant.

    Returns:
      jaccard: |intersection(tokens)| / |union(tokens)|
      precision: |intersection(tokens)| / |tokens(variant)|
    """
    c_toks = set(tokenize(canonical.lower()))
    v_toks = tokenize(variant.lower())
    if not c_toks or not v_toks:
        return 0.0, 0.0

    v_set = set(v_toks)
    inter = c_toks.intersection(v_set)
    union = c_toks.union(v_set)

    jaccard = len(inter) / float(len(union))
    precision = len(inter) / float(len(v_toks))
    return jaccard, precision


def summarize_floats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    if len(vals) == 1:
        return {
            "count": 1,
            "mean": float(vals[0]),
            "std": 0.0,
            "min": float(vals[0]),
            "max": float(vals[0]),
        }
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


# -------------------------- LLaMA EMBEDDING -----------------------------------

TOKENIZER = None
MODEL = None


def init_llama(model_name_or_path: str, device: str = "cpu") -> None:
    """
    Initialize global tokenizer + model for LLaMA embeddings.
    Uses AutoModelForCausalLM and computes mean-pooled last-layer hidden states.
    """
    global TOKENIZER, MODEL

    if TOKENIZER is not None and MODEL is not None:
        return

    print(f"[INFO] Loading LLaMA model for embeddings from: {model_name_or_path}", file=sys.stderr)
    TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path)
    MODEL = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if device == "cuda":
        MODEL.to("cuda")
    else:
        MODEL.to("cpu")

    MODEL.eval()
    print("[INFO] LLaMA model ready for embeddings.", file=sys.stderr)


@torch.no_grad()
def embed_with_llama(text: str, max_length: int = 128) -> torch.Tensor:
    """
    Get a simple sentence embedding from LLaMA by mean-pooling
    the last hidden layer over non-padding tokens.
    Returns a 1D tensor of shape [hidden_dim] on CPU.
    """
    if TOKENIZER is None or MODEL is None:
        raise RuntimeError("LLaMA tokenizer/model not initialized. Call init_llama() first.")

    inputs = TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # Move to same device as model
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    outputs = MODEL(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]          # [1, seq_len, hidden_dim]
    mask = inputs["attention_mask"].unsqueeze(-1).type_as(hidden)  # [1, seq_len, 1]

    masked = hidden * mask
    summed = masked.sum(dim=1)                 # [1, hidden_dim]
    counts = mask.sum(dim=1)                   # [1, 1]
    mean_vec = summed / counts.clamp(min=1.0)  # [1, hidden_dim]

    return mean_vec[0].cpu()  # 1D tensor


def semantic_similarity_llama(canonical: str, variant: str) -> float:
    """
    Cosine similarity between LLaMA embeddings of canonical and variant.
    """
    va = embed_with_llama(canonical)
    vb = embed_with_llama(variant)

    va_norm = torch.norm(va)
    vb_norm = torch.norm(vb)
    denom = (va_norm * vb_norm).item()
    if denom == 0.0:
        return 0.0
    sim = torch.dot(va, vb).item() / denom
    return float(sim)


# -------------------------- MAIN ANALYSIS -------------------------------------

def analyze_variants(
    records: List[Dict[str, Any]],
    modes: List[str] = None,
) -> Dict[str, Any]:
    if modes is None:
        modes = ["paraphrase", "scrambled", "typo", "nonsense"]

    texts_per_mode: Dict[str, List[str]] = {m: [] for m in modes}
    jacc_per_mode: Dict[str, List[float]] = {m: [] for m in modes}
    prec_per_mode: Dict[str, List[float]] = {m: [] for m in modes}
    sim_per_mode: Dict[str, List[float]] = {m: [] for m in modes}

    # Cache canonical embeddings so we don't recompute them
    canon_embed_cache: Dict[str, torch.Tensor] = {}

    def get_canon_embedding(canon: str) -> torch.Tensor:
        if canon not in canon_embed_cache:
            canon_embed_cache[canon] = embed_with_llama(canon)
        return canon_embed_cache[canon]

    n_records_with_variants = 0

    for rec in records:
        iv = rec.get("instruction_variants")
        if not iv or not isinstance(iv, dict):
            continue

        canonical = (rec.get("original_instruction") or "").strip()
        if not canonical:
            normal_list = iv.get("normal") or []
            if normal_list:
                canonical = str(normal_list[0]).strip()
        if not canonical:
            # skip records with no canonical at all
            continue

        n_records_with_variants += 1

        # Precompute canonical embedding for this record
        canon_vec = get_canon_embedding(canonical)

        for mode in modes:
            v_list = iv.get(mode) or []
            for v in v_list:
                s = str(v).strip()
                if not s:
                    continue
                texts_per_mode[mode].append(s)

                # Lexical metrics
                jacc, prec = lexical_overlap(canonical, s)
                jacc_per_mode[mode].append(jacc)
                prec_per_mode[mode].append(prec)

                # Semantic similarity via LLaMA
                v_vec = embed_with_llama(s)
                va_norm = torch.norm(canon_vec)
                vb_norm = torch.norm(v_vec)
                denom = (va_norm * vb_norm).item()
                if denom == 0.0:
                    sim = 0.0
                else:
                    sim = torch.dot(canon_vec, v_vec).item() / denom
                sim_per_mode[mode].append(float(sim))

    metrics: Dict[str, Any] = {"global": {}, "modes": {}}
    metrics["global"]["n_records_with_variants"] = n_records_with_variants

    for mode in modes:
        texts = texts_per_mode[mode]
        d1 = distinct_n(texts, n=1)
        d2 = distinct_n(texts, n=2)

        j_stats = summarize_floats(jacc_per_mode[mode])
        p_stats = summarize_floats(prec_per_mode[mode])
        s_stats = summarize_floats(sim_per_mode[mode])

        metrics["modes"][mode] = {
            "n_texts": len(texts),
            "distinct_1": d1,
            "distinct_2": d2,
            "lexical_jaccard_stats": j_stats,
            "lexical_precision_stats": p_stats,
            "semantic_similarity_stats": s_stats,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Distinct-N, lexical overlap, and LLaMA-based semantic "
            "similarity for instruction variants (no sentence-transformers)."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with records that contain 'instruction_variants'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON with metrics.",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="HF model name or local path for LLaMA (e.g., Meta-Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the LLaMA model.",
    )
    args = parser.parse_args()

    records = load_records_jsonl(args.input)
    print(f"[INFO] Loaded {len(records)} records from {args.input}", file=sys.stderr)

    # Init LLaMA for semantic similarity
    init_llama(args.model_name_or_path, device=args.device)

    modes = ["paraphrase", "scrambled", "typo", "nonsense"]
    metrics = analyze_variants(records, modes=modes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Wrote metrics to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
