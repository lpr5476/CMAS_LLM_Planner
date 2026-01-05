#!/usr/bin/env python
"""
canon.py

Generate instruction variants (paraphrase, scrambled, typo/ungrammatical, nonsense)
for each record in an MR dataset using a local LLaMA 3 8B Instruct model (HF transformers).

Expected input records (JSON or JSONL), each something like:
{
  "id": "10",
  "source_index": 10,
  "goal_type": "single" | "multi",
  "context": "...",
  "goal_canonical": {...},
  "goal_canonical_simple": "...",
  "original_instruction": "..."  # canonical operator instruction
}

Output: same records, plus an "instruction_variants" field:
{
  "normal":      ["<canonical>"],
  "paraphrase":  ["...", "..."],
  "scrambled":   ["...", "..."],
  "typo":        ["...", "..."],
  "nonsense":    ["...", "..."]
}
"""

import argparse
import json
import os
import re
import sys
import random
from typing import Any, Dict, List, Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# -------------------------- IO HELPERS ----------------------------------------


def load_mr_records(path: str) -> List[Dict[str, Any]]:
    """Load MR records from .json (list) or .jsonl (one JSON object per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in JSON file")
    return data


def save_records_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Save list of dicts as JSONL."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------- SIMILARITY CONFIG ---------------------------------

# Similarity thresholds (used only if SBERT is available)
PARA_SIM_MIN = 0.80
PARA_SIM_MAX = 0.95
SCRAM_SIM_MIN = 0.80
TYPO_SIM_MIN = 0.95
NONSENSE_SIM_MAX = 0.40

SBERT_MODEL = None  # will be set by _init_sbert()


def _init_sbert():
    """Optional Sentence-BERT for semantic similarity. Falls back gracefully."""
    global SBERT_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[INFO] Loaded SBERT model for semantic similarity.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] SBERT not available ({e}); similarity scores will be ignored.", file=sys.stderr)
        SBERT_MODEL = None


def semantic_similarity(a: str, b: str) -> Optional[float]:
    """Return cosine similarity in [−1, 1], or None if SBERT not available."""
    if SBERT_MODEL is None:
        return None
    import numpy as np  # local import to keep dependency optional

    emb = SBERT_MODEL.encode([a, b], convert_to_numpy=True)
    v1, v2 = emb[0], emb[1]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return None
    return float(np.dot(v1, v2) / denom)


def filter_by_similarity(
    original: str,
    candidates: List[str],
    sim_min: Optional[float] = None,
    sim_max: Optional[float] = None,
) -> List[str]:
    """
    Keep only candidates whose SBERT similarity to the original lies in [sim_min, sim_max].
    If SBERT is not available, return candidates unchanged.
    """
    if SBERT_MODEL is None:
        return [c.strip() for c in candidates if c.strip()]

    out: List[str] = []
    for c in candidates:
        s = c.strip()
        if not s:
            continue
        sim = semantic_similarity(original, s)
        if sim is None:
            out.append(s)
            continue
        if sim_min is not None and sim < sim_min:
            continue
        if sim_max is not None and sim > sim_max:
            continue
        out.append(s)
    return out


# -------------------------- HF LLaMA INIT -------------------------------------


DEFAULT_MAX_NEW_TOKENS = 256


def init_generation_pipeline(
    model_name_or_path: str,
    device: str = "cuda",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Tuple[Any, Dict[str, Any]]:
    """Initialize a HuggingFace text-generation pipeline for a local LLaMA model."""
    print(f"[INFO] Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if device == "cuda":
        model = model.to("cuda")
        pipe_device = 0
    else:
        model = model.to("cpu")
        pipe_device = -1

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=pipe_device,
    )

    base_gen_kwargs = {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    return gen_pipe, base_gen_kwargs


def generate_continuation(
    gen_pipe: Any,
    prompt: str,
    gen_kwargs: Dict[str, Any],
) -> str:
    """
    Generate text continuation given a prompt.
    Returns only the part AFTER the prompt, if possible.
    """
    out = gen_pipe(prompt, **gen_kwargs)[0]["generated_text"]
    if out.startswith(prompt):
        return out[len(prompt):].strip()
    return out.strip()


# -------------------------- PARSING NUMBERED LISTS ----------------------------


def parse_numbered_list(text: str) -> List[str]:
    """
    Parse a simple numbered or bulleted list from the model output.
    Accepts formats like:
      1. ...
      2) ...
      - ...
      • ...
    Returns a list of item strings (deduplicated, non-empty).
    """
    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # "1. text" or "2) text"
        m = re.match(r"^\s*(\d+)[\.\)]\s*(.+)$", line)
        if m:
            items.append(m.group(2).strip())
            continue

        # "- text" or "• text"
        m2 = re.match(r"^\s*[-•]\s*(.+)$", line)
        if m2:
            items.append(m2.group(1).strip())
            continue

    # Deduplicate & drop empties
    out: List[str] = []
    seen = set()
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


# -------------------------- SYSTEM MESSAGE ------------------------------------


SYS_MESSAGE = """
You are an instruction variant generator.

Your job is to rewrite short operator instructions into several linguistic variants.

General rules for all NON-NONSENSE variants:
- Treat the given instruction as already correct and canonical.
- Preserve the SAME underlying actions and their order.
- Do NOT add, remove, or reorder actions.
- Every named process and every packing step in the original instruction
  MUST be mentioned in every non-nonsense variant.
- Copy every special name, product ID, station name, and number
  EXACTLY as it appears in the original instruction (treat them as fixed tokens).
- Use short, imperative sentences as an operator might say them.

General rules for ALL responses:
- NEVER output JSON.
- NEVER explain what you are doing.
- NEVER write things like "Here are two variants" or "Note that...".
- ONLY output the numbered list requested by the user.
""".strip()


# -------------------------- REQUIRED OPS FILTER -------------------------------

_PROCESS_RE = re.compile(r"runprocess\s*(\d+)", re.IGNORECASE)


def extract_required_ops(canonical: str) -> Dict[str, Any]:
    """
    From the canonical instruction, extract:
      - processes: list of process IDs like "runprocess1", "runprocess2"
      - requires_packing: True if any 'pack' or 'packing' appears
    """
    text = canonical.lower()
    procs: List[str] = []
    for m in _PROCESS_RE.finditer(text):
        num = m.group(1)
        label = f"runprocess{num}"
        if label not in procs:
            procs.append(label)

    requires_packing = bool(re.search(r"\bpack(ing)?\b", text))
    return {
        "processes": procs,
        "requires_packing": requires_packing,
    }


def _candidate_mentions_process(text: str, proc_label: str) -> bool:
    """
    Check if candidate text mentions the given process.
    Accept variants like 'RunProcess1', 'Process1', 'Process 1'.
    """
    text = text.lower()
    m = re.search(r"(\d+)$", proc_label)
    if not m:
        # fallback: direct substring
        return proc_label in text

    num = m.group(1)
    # Accept 'runprocess1', 'process1', 'process 1'
    if proc_label in text:
        return True
    if f"process{num}" in text:
        return True
    if f"process {num}" in text:
        return True
    return False


def candidate_has_required_ops(candidate: str, ops: Dict[str, Any]) -> bool:
    """
    For non-nonsense variants:
      - every process in canonical must be mentioned
      - packing must be mentioned if present in canonical
    """
    text = candidate.lower()

    # All processes must be present
    for proc in ops.get("processes", []):
        if not _candidate_mentions_process(text, proc):
            return False

    # Packing must be present if canonical has it
    if ops.get("requires_packing", False):
        if not re.search(r"\bpack(ing)?\b", text):
            return False

    return True


def filter_by_required_ops(cands: List[str], ops: Dict[str, Any]) -> List[str]:
    """Filter candidate strings by required-ops constraint."""
    out: List[str] = []
    for s in cands:
        s2 = s.strip()
        if not s2:
            continue
        if candidate_has_required_ops(s2, ops):
            out.append(s2)
    return out


# -------------------------- LLM CALL PER MODE ---------------------------------


def call_llm_variants_mode_list(
    gen_pipe: Any,
    base_gen_kwargs: Dict[str, Any],
    mode: str,
    canonical_instruction: str,
    num: int = 2,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> List[str]:
    """
    Ask the model for a small numbered/bulleted list for a SINGLE mode:
      - mode: "paraphrase" | "scrambled" | "typo" | "nonsense"

    The model returns plain text; we parse it with parse_numbered_list().
    """
    mode = mode.lower()
    canon = canonical_instruction.strip()

      if mode == "paraphrase":
        mode_desc = "PARAPHRASES"
        extra = f"""
Task: Write EXACTLY {num} different natural-sounding operator instructions that
describe EXACTLY the same actions and step order as OPERATOR_INSTRUCTION.

Requirements:
- Use short, direct, imperative sentences.
- Keep the SAME number of actions and the SAME order of actions.
- Do NOT add, remove, or merge actions.
- Every named process and every packing step in OPERATOR_INSTRUCTION
  MUST be mentioned in each paraphrase.
- Copy all special names, product IDs, station names, and numbers EXACTLY.
- HOWEVER, change the wording as much as possible:
  - Use different verbs, connectors, and sentence structure.
  - Avoid reusing long phrases from OPERATOR_INSTRUCTION.
  - Each paraphrase should sound like it was written by a different person.
"""

    elif mode == "scrambled":
        mode_desc = "SCRAMBLED"
        extra = f"""
Task: Write EXACTLY {num} scrambled variants of this instruction.

Requirements:
- Keep the SAME actions and the SAME order as OPERATOR_INSTRUCTION.
- Express the SAME underlying meaning.
- Use awkward or jumbled word order that is still understandable to a human operator.
- You may:
  - Move phrases to strange positions (e.g., time phrases in the middle),
  - Drop or repeat small function words like "the", "then", "and".
- Do NOT move an action earlier or later in the sequence.
- Do NOT remove or merge actions.
- Every named process and every packing step in OPERATOR_INSTRUCTION
  MUST be mentioned in each scrambled variant.
- Copy all special names, product IDs, station names, and numbers EXACTLY.
"""

    elif mode == "typo":
        mode_desc = "UNGRAMMATICAL"
        extra = f"""
Task: Write EXACTLY {num} ungrammatical variants of this instruction.

Requirements:
- Keep the SAME actions and the SAME order as OPERATOR_INSTRUCTION.
- Be a single sentence an operator might realistically type or say quickly.
- Introduce 1–3 small, realistic spelling or grammar mistakes, such as:
  - misspellings,
  - missing articles or prepositions,
  - slightly wrong word order.
- Do NOT remove or merge actions.
- Every named process and every packing step in OPERATOR_INSTRUCTION
  MUST be mentioned in each ungrammatical variant.
- Copy all special names, product IDs, station names, coordinates, and numbers EXACTLY
  (do NOT introduce typos inside these special tokens).
"""

    elif mode == "nonsense":
        mode_desc = "NONSENSE"
        extra = f"""
Task: Write EXACTLY {num} nonsense operator instructions.

Requirements:
- Each sentence must clearly NOT describe the real goal implied by OPERATOR_INSTRUCTION.
- The tasks should be obviously wrong, impossible, or unrelated to the original goal.
- Still sound like sentences in an industrial/manufacturing environment
  (e.g., robots, machines, stations, factories), but with absurd or mismatched content.
- Do NOT simply rephrase or slightly distort OPERATOR_INSTRUCTION.
- Prefer to change at least one of:
  - the main task type (e.g., painting instead of transporting, shredding instead of drilling),
  - the parts/products involved,
  - the quantities or locations (e.g., exaggerated or ridiculous counts).
- Avoid copying long phrases from OPERATOR_INSTRUCTION.
  Use mostly different wording so the sentence is not a paraphrase.
- You do NOT need to preserve the original actions or order at all.
"""


    prompt = f"System:\n{SYS_MESSAGE}\n\nUser:\n{task}\n\nAssistant:\n"

    gen_kwargs = dict(base_gen_kwargs)
    gen_kwargs.update({
        "temperature": temperature,
        "top_p": top_p,
    })

    raw = generate_continuation(gen_pipe, prompt, gen_kwargs)
    if not raw:
        return []

    # Debug: comment out if too noisy
    print(f"=== RAW LLM OUTPUT (mode={mode}) ===")
    print(raw)
    print("=== END RAW OUTPUT ===")

    return parse_numbered_list(raw)


# -------------------------- RULE-BASED SCRAMBLED & TYPO -----------------------

def scramble_tokens_once(tokens: List[str]) -> List[str]:
    """Simple token-level scrambling: random swaps and small shuffles."""
    t = tokens[:]
    if len(t) < 2:
        return t

    # random adjacent swaps
    num_swaps = max(1, len(t) // 5)
    for _ in range(num_swaps):
        if len(t) < 2:
            break
        i = random.randint(0, len(t) - 2)
        t[i], t[i + 1] = t[i + 1], t[i]

    # occasional global shuffle of a small window
    if len(t) > 4:
        start = random.randint(0, len(t) - 3)
        end = min(len(t), start + random.randint(2, 4))
        window = t[start:end]
        random.shuffle(window)
        t[start:end] = window
    return t


def generate_scrambled_variants_rule_based(original: str, n: int = 2) -> List[str]:
    """
    Rule-based scrambling: keep tokens identical, change word order/grammar.
    """
    tokens = original.split()
    seen = set()
    out: List[str] = []
    attempts = 0
    max_attempts = n * 5

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        new_tokens = scramble_tokens_once(tokens)
        s = " ".join(new_tokens).strip()
        if not s or s == original or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


QWERTY_LINE = "qwertyuiopasdfghjklzxcvbnm"
SPECIAL_TOKEN_RE = re.compile(r"([A-Z][A-Za-z0-9_]*|\w*\d+\w*)")


def is_special_token(tok: str) -> bool:
    """
    Heuristic: treat tokens with digits or initial capital (skill names, IDs, coords)
    as special; we avoid introducing typos into these.
    """
    return bool(SPECIAL_TOKEN_RE.fullmatch(tok))


def neighbor_char(c: str) -> str:
    """Very simple 'keyboard neighbor' approximation along QWERTY_LINE."""
    low = c.lower()
    if low not in QWERTY_LINE:
        return c
    idx = QWERTY_LINE.index(low)
    candidates = []
    if idx > 0:
        candidates.append(QWERTY_LINE[idx - 1])
    if idx < len(QWERTY_LINE) - 1:
        candidates.append(QWERTY_LINE[idx + 1])
    if not candidates:
        return c
    rep = random.choice(candidates)
    return rep.upper() if c.isupper() else rep


def inject_typos_into_token(tok: str, char_noise_prob: float = 0.08) -> str:
    """Apply character-level noise to a single token."""
    out_chars: List[str] = []
    for ch in tok:
        if random.random() > char_noise_prob:
            out_chars.append(ch)
            continue
        op = random.choice(["sub", "del", "ins"])
        if op == "sub":
            out_chars.append(neighbor_char(ch))
        elif op == "del":
            continue  # delete char
        elif op == "ins":
            out_chars.append(ch)
            out_chars.append(neighbor_char(ch))
    if not out_chars:
        return tok
    return "".join(out_chars)


def generate_typo_variants_rule_based(original: str, n: int = 2, char_noise_prob: float = 0.08) -> List[str]:
    """
    Rule-based typo/ungrammatical variants:
    - keep special tokens (skills, IDs, coordinates) untouched
    - inject realistic misspellings into function words, etc.
    """
    base_tokens = original.split()
    replacements = {
        "you": "u",
        "are": "r",
        "please": "pls",
        "before": "b4",
    }

    seen = set()
    out: List[str] = []
    attempts = 0
    max_attempts = n * 5

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        new_tokens: List[str] = []
        for tok in base_tokens:
            core = tok.strip(".,!?")
            prefix = tok[: len(tok) - len(tok.lstrip(".,!?"))]
            suffix = tok[len(tok.rstrip(".,!?")) :]

            if is_special_token(core):
                # Keep skill names, IDs, coords intact
                new_tok = tok
            else:
                # Maybe apply SMS-style shortcut
                low_core = core.lower()
                if low_core in replacements and random.random() < 0.4:
                    new_core = replacements[low_core]
                else:
                    new_core = inject_typos_into_token(core, char_noise_prob=char_noise_prob)
                new_tok = f"{prefix}{new_core}{suffix}"

            new_tokens.append(new_tok)

        candidate = " ".join(new_tokens).strip()
        if not candidate or candidate == original or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)

    return out


# -------------------------- PER-RECORD LOGIC ----------------------------------


def take_n_unique(cands: List[str], n: int = 2) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in cands:
        s2 = s.strip()
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
        if len(out) >= n:
            break
    return out


def generate_variants_for_record(
    gen_pipe: Any,
    base_gen_kwargs: Dict[str, Any],
    record: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Generate variants for a single MR record.

    Returns:
    {
      "normal":      [canonical],
      "paraphrase":  [up to 2 strings],
      "scrambled":   [up to 2 strings],
      "typo":        [up to 2 strings],
      "nonsense":    [up to 2 strings]
    }
    """
    canonical = (record.get("original_instruction") or "").strip()
    variants: Dict[str, List[str]] = {
        "normal": [],
        "paraphrase": [],
        "scrambled": [],
        "typo": [],
        "nonsense": [],
    }

    if not canonical:
        return variants

    variants["normal"].append(canonical)

    # Extract required ops from canonical for filtering
    ops_info = extract_required_ops(canonical)

    # 1) Paraphrases (LLM + required ops + semantic filter)
    para_cands_raw = call_llm_variants_mode_list(
        gen_pipe,
        base_gen_kwargs,
        mode="paraphrase",
        canonical_instruction=canonical,
        num=4,               # oversample, then filter
        temperature=0.7,
        top_p=0.9,
    )
    para_cands_ops = filter_by_required_ops(para_cands_raw, ops_info)
    para_cands = filter_by_similarity(
        canonical,
        para_cands_ops,
        sim_min=PARA_SIM_MIN,
        sim_max=PARA_SIM_MAX,
    )
    variants["paraphrase"] = take_n_unique(para_cands, 2)

    # 2) Scrambled (rule-based first; fallback to LLM if needed)
    scram_opts_rb = generate_scrambled_variants_rule_based(canonical, n=4)
    scram_opts_rb = filter_by_similarity(
        canonical,
        scram_opts_rb,
        sim_min=SCRAM_SIM_MIN,
        sim_max=None,
    )
    scram_opts_rb = filter_by_required_ops(scram_opts_rb, ops_info)

    if scram_opts_rb:
        scram_cands = scram_opts_rb
    else:
        # Fallback to LLM scrambled mode if rule-based yields nothing
        scram_cands_raw = call_llm_variants_mode_list(
            gen_pipe,
            base_gen_kwargs,
            mode="scrambled",
            canonical_instruction=canonical,
            num=4,
            temperature=0.8,
            top_p=0.9,
        )
        scram_cands_ops = filter_by_required_ops(scram_cands_raw, ops_info)
        scram_cands = filter_by_similarity(
            canonical,
            scram_cands_ops,
            sim_min=SCRAM_SIM_MIN,
            sim_max=None,
        )

    variants["scrambled"] = take_n_unique(scram_cands, 2)

    # 3) Typo / ungrammatical (rule-based; fallback to LLM if really needed)
    typo_opts_rb = generate_typo_variants_rule_based(canonical, n=4, char_noise_prob=0.08)
    typo_opts_rb = filter_by_similarity(
        canonical,
        typo_opts_rb,
        sim_min=TYPO_SIM_MIN,
        sim_max=None,
    )
    typo_opts_rb = filter_by_required_ops(typo_opts_rb, ops_info)

    if typo_opts_rb:
        typo_cands = typo_opts_rb
    else:
        # Fallback to LLM typo mode
        typo_cands_raw = call_llm_variants_mode_list(
            gen_pipe,
            base_gen_kwargs,
            mode="typo",
            canonical_instruction=canonical,
            num=4,
            temperature=0.7,
            top_p=0.9,
        )
        typo_cands_ops = filter_by_required_ops(typo_cands_raw, ops_info)
        typo_cands = filter_by_similarity(
            canonical,
            typo_cands_ops,
            sim_min=TYPO_SIM_MIN,
            sim_max=None,
        )

    variants["typo"] = take_n_unique(typo_cands, 2)

    # 4) Nonsense (LLM + semantic filter to keep them far)
    nonsense_cands_raw = call_llm_variants_mode_list(
        gen_pipe,
        base_gen_kwargs,
        mode="nonsense",
        canonical_instruction=canonical,
        num=4,
        temperature=0.9,
        top_p=0.95,
    )
    # No required-ops filter here (we WANT mismatches).
    nonsense_cands = filter_by_similarity(
        canonical,
        nonsense_cands_raw,
        sim_min=None,
        sim_max=NONSENSE_SIM_MAX,
    )
    variants["nonsense"] = take_n_unique(nonsense_cands, 2)

    return variants


# -------------------------- MAIN ----------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate instruction variants (paraphrase, scrambled, typo, nonsense) "
            "per MR record using a local LLaMA 3 8B Instruct model."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to MR dataset (JSON or JSONL), e.g., mr_dataset.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path with added 'instruction_variants' field.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Model path or directory for the local LLaMA 3 8B Instruct model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max new tokens to generate per prompt.",
    )

    args = parser.parse_args()

    # Seed for reproducible rule-based noise
    random.seed(42)

    records = load_mr_records(args.input)
    print(f"[INFO] Loaded {len(records)} MR records from {args.input}")

    gen_pipe, base_gen_kwargs = init_generation_pipeline(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    # Optional semantic similarity
    _init_sbert()

    out_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        print(f"[INFO] Processing record {idx+1}/{len(records)} (id={rec.get('id')})")
        try:
            vars_for_rec = generate_variants_for_record(gen_pipe, base_gen_kwargs, rec)
        except Exception as e:
            print(f"[WARN] Failed to generate variants for record {idx}: {e}")
            vars_for_rec = {
                "normal": [],
                "paraphrase": [],
                "scrambled": [],
                "typo": [],
                "nonsense": [],
            }
        rec_out = dict(rec)
        rec_out["instruction_variants"] = vars_for_rec
        out_records.append(rec_out)

    save_records_jsonl(out_records, args.output)
    print(f"[INFO] Wrote {len(out_records)} records with instruction_variants to {args.output}")


if __name__ == "__main__":
    main()
