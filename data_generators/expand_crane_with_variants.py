#!/usr/bin/env python3
# expand_crane_with_variants.py
import json, argparse, copy, os
from typing import Dict, Any, List, Tuple, Set

# Import your existing variant generator
from instruct_variants import make_variants  # returns {"original_instruction", "variants": {...}}

NEGATIVE_OUTPUT_TEXT = "Cannot generate goal with these instructions"

def _iter_variant_texts(variant_obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Yield (variant_type, text) for all mappable buckets."""
    v = (variant_obj or {}).get("variants", {}) or {}
    out: List[Tuple[str, str]] = []
    for btype in ["humanized", "paraphrases", "scrambled", "ungrammatical", "swedish"]:
        for s in v.get(btype, []) or []:
            if isinstance(s, str):
                s2 = s.strip()
                if s2:
                    out.append((btype, s2))
    return out

def _iter_nonsense(variant_obj: Dict[str, Any]) -> List[str]:
    """Return nonsense texts (objects with {'text': '...', 'unmappable': true})."""
    v = (variant_obj or {}).get("variants", {}) or {}
    res: List[str] = []
    for obj in v.get("nonsense", []) or []:
        if isinstance(obj, dict):
            txt = (obj.get("text") or "").strip()
            if txt:
                res.append(txt)
    return res

def expand_with_variants(
    goals_path: str,
    out_path: str,
    model_name: str = "llama3:8B",
):
    with open(goals_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("Input file must be a JSON array of items.")

    expanded: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    def add_row(row: Dict[str, Any]):
        # Deduplicate by instruction string
        instr = (row.get("instruction") or "").strip()
        if instr and instr not in seen_texts:
            expanded.append(row)
            seen_texts.add(instr)

    # 0) Always keep originals
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("instruction"), str):
            add_row(it)

    # 1) Generate + add mapped variants and nonsense-to-negative rows
    for it in items:
        instr = (it.get("instruction") or "").strip()
        if not instr:
            continue

        vobj = make_variants(instr, model=model_name)

        # mapped buckets -> copy input/output unchanged
        for vtype, vtext in _iter_variant_texts(vobj):
            new_item = copy.deepcopy(it)
            new_item["instruction"] = vtext
            # Never include meta
            if "meta" in new_item:
                del new_item["meta"]
            add_row(new_item)

        # nonsense -> same file, but output is the fixed negative text
        for ntext in _iter_nonsense(vobj):
            neg_item = {
                "instruction": ntext,
                # keep the same input context if present, helpful for model rejection training
                "input": it.get("input"),
                "output": NEGATIVE_OUTPUT_TEXT,
            }
            add_row(neg_item)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(expanded)} rows -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", default="data/input_output/generated_goals_crane.json",
                    help="Path to generated goals JSON (array of items)")
    ap.add_argument("--out", default="data/input_output/generated_goals_crane_expanded.json",
                    help="Output JSON path (single file with originals + variants + nonsense)")
    ap.add_argument("--model", default="llama3:8B",
                    help="Model name passed to instruct_variants.make_variants")
    args = ap.parse_args()

    expand_with_variants(
        goals_path=args.goals,
        out_path=args.out,
        model_name=args.model,
    )

