#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drill dataset generator that produces the same output format as the crane generator.

What this script guarantees:
- Single randomized CONTEXT per run (NameTag length/width; Hole Diameter bounds).
- CONTEXT text includes AGENTS/INTERFACES/SKILLS as provided.
- Produces EXACTLY NUM_SAMPLES items by repeatedly sampling variants:
    • Single hole @ (x,y) with default/min/max diameter
    • Two holes @ (x1,y1) and (x2,y2)
    • Three holes along a diagonal
    • Four corners
    • Center
    • Evenly spaced along top/bottom/left/right for n ∈ {2,3}
- Output file (JSON list): data/input_output/generated_goals.json
- Each item mirrors crane format:
    {
      "instruction": <natural language variant>,
      "input": {"context": <ENV_SUMMARY>},
      "output": {
         "plan": [ ...human steps... ],
         "goal_canonical": {"part": "NameTag", "sequence": [ ... ]},
         "goal_canonical_simple": "NameTag: ... ; ..."
      }
    }
"""

import json
import os
import random
import argparse
from typing import Dict, Any, Tuple, List

# ================= SETTINGS =================
OUTPUT_DIR = os.path.join(".", "data", "input_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED  = 42
NUM_SAMPLES  = 1000  # <- Exact number of rows to generate

# Ranges (mm) for randomizing the single context
NAME_TAG_LENGTH_RANGE = (30.0, 80.0)
NAME_TAG_WIDTH_RANGE  = (15.0, 40.0)
DIAMETER_RANGE        = (3.0, 8.0)

DOMAIN = "drill"
# ===========================================


def set_seed(seed: int) -> None:
    random.seed(seed)


def fmt_num(v: float) -> str:
    """Format like your examples (strip trailing .0; round to 3 dp otherwise)."""
    try:
        if float(v).is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(round(float(v), 3))


def clamp_int(v: float) -> int:
    return max(0, int(round(v)))


def sample_name_tag_dims() -> Tuple[float, float]:
    L = random.uniform(*NAME_TAG_LENGTH_RANGE)
    W = random.uniform(*NAME_TAG_WIDTH_RANGE)
    return round(L, 3), round(W, 3)


def sample_diameter_bounds() -> Tuple[float, float, float]:
    d1 = random.uniform(*DIAMETER_RANGE)
    d2 = random.uniform(*DIAMETER_RANGE)
    lo, hi = (d1, d2) if d1 <= d2 else (d2, d1)
    lo, hi = round(lo, 3), round(hi, 3)
    default = round((lo + hi) / 2.0, 3)
    return lo, hi, default


def synth_context_text(L: float, W: float, d_lo: float, d_hi: float) -> str:
    """Exact context layout provided in the prompt."""
    Ls, Ws = fmt_num(L), fmt_num(W)
    dlos, dhis = fmt_num(d_lo), fmt_num(d_hi)
    return (
        "CONTEXT:\n"
        "AGENTS\n"
        "- DrillBit   [RESOURCE] { Interfaces: Hole, Chuck_B10, Skills: MakeAhole }\n"
        "- DrillStation[RESOURCE] { Interfaces: Chuck_B10, Skills: Drill }\n"
        f"- NameTag    [PART] {{ Var: length={Ls}, width={Ws} }}\n\n"
        "INTERFACES\n"
        "- Chuck_B10  (agent=DrillBit): x:REAL=[,], y:REAL=[,]\n"
        "- Chuck_B10  (agent=DrillStation): x:REAL=[,], y:REAL=[,]\n"
        f"- Hole       (agent=DrillBit): x:REAL=[,], y:REAL=[,], Diameter:REAL=[{dlos},{dhis}]\n\n"
        "SKILLS\n"
        "- Drill     (agent=DrillStation) requires Chuck_B10(x,y)\n"
        "- MakeAhole (agent=DrillBit) requires Hole(x,y,Diameter)"
    )


# ---------- coordinate helpers (int-friendly for operator text) ----------

def rand_xy(L: int, W: int) -> Tuple[int, int]:
    return random.randint(0, max(0, L)), random.randint(0, max(0, W))


def center_xy(L: int, W: int) -> Tuple[int, int]:
    return clamp_int(L * 0.5), clamp_int(W * 0.5)


def diag3(L: int, W: int) -> List[Tuple[int, int]]:
    return [
        (clamp_int(L * 0.25), clamp_int(W * 0.25)),
        (clamp_int(L * 0.50), clamp_int(W * 0.50)),
        (clamp_int(L * 0.75), clamp_int(W * 0.75)),
    ]


def corners(L: int, W: int) -> List[Tuple[int, int]]:
    return [(0, 0), (L, 0), (0, W), (L, W)]


def evenly_spaced_top(n: int, L: int) -> List[Tuple[int, int]]:
    y = 0
    xs = [clamp_int((L * (i + 1) / (n + 1))) for i in range(n)]
    return [(x, y) for x in xs]


def evenly_spaced_bottom(n: int, L: int, W: int) -> List[Tuple[int, int]]:
    y = W
    xs = [clamp_int((L * (i + 1) / (n + 1))) for i in range(n)]
    return [(x, y) for x in xs]


def evenly_spaced_left(n: int, W: int) -> List[Tuple[int, int]]:
    x = 0
    ys = [clamp_int((W * (i + 1) / (n + 1))) for i in range(n)]
    return [(x, y) for y in ys]


def evenly_spaced_right(n: int, L: int, W: int) -> List[Tuple[int, int]]:
    x = L
    ys = [clamp_int((W * (i + 1) / (n + 1))) for i in range(n)]
    return [(x, y) for y in ys]


# ---------- Canonical goal + plan (crane-compatible) ----------

def build_goal_sequence(points: List[Tuple[int, int]], diameter: float) -> List[Dict[str, Any]]:
    """Canonical goal sequence matching crane format (no 'action', uses 'interface' + 'variables').

    For each hole coordinate (x,y):
      1) DrillStation performs Drill via Chuck_B10(x,y)
      2) DrillBit performs MakeAhole via Hole(x,y,Diameter)
    """
    seq: List[Dict[str, Any]] = []
    for (x, y) in points:
        seq.append({
            "interface": "Chuck_B10",
            "skill": "Drill",
            "variables": {"x": int(x), "y": int(y)}
        })
        seq.append({
            "interface": "Hole",
            "skill": "MakeAhole",
            "variables": {"x": int(x), "y": int(y), "Diameter": float(diameter)}
        })
    return seq


def seq_to_plan_steps(seq: List[Dict[str, Any]]) -> List[str]:
    steps: List[str] = []
    for step in seq:
        skill = step.get("skill", "")
        vars_ = step.get("variables", {})
        if skill == "Drill":
            steps.append(f"Drill({vars_.get('x')},{vars_.get('y')})")
        elif skill == "MakeAhole":
            d = vars_.get("Diameter")
            if d is not None:
                steps.append(f"MakeAhole({vars_.get('x')},{vars_.get('y')},D={fmt_num(d)})")
            else:
                steps.append(f"MakeAhole({vars_.get('x')},{vars_.get('y')})")
    return steps


def build_goal_canonical_simple(part: str, seq: List[Dict[str, Any]]) -> str:
    """Compact single-line like crane's goal_canonical_simple."""
    frags: List[str] = []
    for step in seq:
        skill = step.get("skill")
        v = step.get("variables", {})
        if skill == "Drill":
            frags.append(f"Drill({part}, x={v.get('x')}, y={v.get('y')})")
        elif skill == "MakeAhole":
            d = v.get("Diameter")
            if d is None:
                frags.append(f"MakeAhole({part}, x={v.get('x')}, y={v.get('y')})")
            else:
                frags.append(f"MakeAhole({part}, x={v.get('x')}, y={v.get('y')}, Diameter={fmt_num(d)})")
    return f"{part}: " + " ; ".join(frags) if frags else f"{part}: <empty>"


def make_item(context_text: str, operator_text: str, coords: List[Tuple[int, int]], diameter: float) -> Dict[str, Any]:
    """Produce one training item matching crane format for the drill domain."""
    part = "NameTag"
    seq = build_goal_sequence(coords, diameter)
    plan = seq_to_plan_steps(seq)
    goal_obj = {"part": part, "sequence": seq}
    return {
        "instruction": operator_text,
        "input": {"context": context_text},
        "output": {
            "goal_canonical": goal_obj,
            "goal_canonical_simple": build_goal_canonical_simple(part, seq)
        }
    }


# ---------- SAMPLING CATALOG OF VARIANTS ----------

def sample_variant(L: int, W: int, d_lo: float, d_hi: float, d_default: float, context_text: str) -> Dict[str, Any]:
    """
    Randomly choose a variant family and instantiate it with fresh coordinates (and diameter choice).
    Returns a single training item (crane-compatible format).
    """
    families = [
        "single_default", "single_min", "single_max",
        "two_explicit", "three_diag",
        "four_corners", "center",
        "edge_top_2", "edge_top_3",
        "edge_bottom_2", "edge_bottom_3",
        "edge_left_2", "edge_left_3",
        "edge_right_2", "edge_right_3",
    ]
    choice = random.choice(families)

    if choice == "single_default":
        x, y = rand_xy(L, W)
        return make_item(context_text, f"Drill one hole at ({x},{y}).", [(x, y)], d_default)

    if choice == "single_min":
        x, y = rand_xy(L, W)
        return make_item(context_text, f"Make a hole ({x},{y}) at min diameter.", [(x, y)], d_lo)

    if choice == "single_max":
        x, y = rand_xy(L, W)
        return make_item(context_text, f"Make a hole ({x},{y}) at max diameter.", [(x, y)], d_hi)

    if choice == "two_explicit":
        x1, y1 = rand_xy(L, W)
        x2, y2 = rand_xy(L, W)
        # Avoid identical points
        if x1 == x2 and y1 == y2:
            x2 = min(L, x2 + 1)
        return make_item(context_text, f"Drill two holes at ({x1},{y1}) and ({x2},{y2}).", [(x1, y1), (x2, y2)], d_default)

    if choice == "three_diag":
        pts = diag3(L, W)
        return make_item(context_text,
                         f"Drill three holes at ({pts[0][0]},{pts[0][1]}), ({pts[1][0]},{pts[1][1]}), ({pts[2][0]},{pts[2][1]}).",
                         pts, d_default)

    if choice == "four_corners":
        return make_item(context_text, "Drill holes in the four corners.", corners(L, W), d_default)

    if choice == "center":
        cx, cy = center_xy(L, W)
        return make_item(context_text, "Drill a hole in the center of the tag.", [(cx, cy)], d_default)

    if choice == "edge_top_2":
        pts = evenly_spaced_top(2, L)
        return make_item(context_text, "Drill 2 evenly spaced holes along the top edge.", pts, d_default)

    if choice == "edge_top_3":
        pts = evenly_spaced_top(3, L)
        return make_item(context_text, "Drill 3 evenly spaced holes along the top edge.", pts, d_default)

    if choice == "edge_bottom_2":
        pts = evenly_spaced_bottom(2, L, W)
        return make_item(context_text, "Drill 2 evenly spaced holes along the bottom edge.", pts, d_default)

    if choice == "edge_bottom_3":
        pts = evenly_spaced_bottom(3, L, W)
        return make_item(context_text, "Drill 3 evenly spaced holes along the bottom edge.", pts, d_default)

    if choice == "edge_left_2":
        pts = evenly_spaced_left(2, W)
        return make_item(context_text, "Drill 2 evenly spaced holes along the left edge.", pts, d_default)

    if choice == "edge_left_3":
        pts = evenly_spaced_left(3, W)
        return make_item(context_text, "Drill 3 evenly spaced holes along the left edge.", pts, d_default)

    if choice == "edge_right_2":
        pts = evenly_spaced_right(2, L, W)
        return make_item(context_text, "Drill 2 evenly spaced holes along the right edge.", pts, d_default)

    if choice == "edge_right_3":
        pts = evenly_spaced_right(3, L, W)
        return make_item(context_text, "Drill 3 evenly spaced holes along the right edge.", pts, d_default)

    # Fallback (shouldn't happen)
    x, y = rand_xy(L, W)
    return make_item(context_text, f"Drill one hole at ({x},{y}).", [(x, y)], d_default)


# ----------------- MAIN -----------------

def generate(output_path: str, num_samples: int = NUM_SAMPLES) -> None:
    set_seed(RANDOM_SEED)

    # 1) Randomize ONE context
    Lf, Wf = sample_name_tag_dims()
    d_lo, d_hi, d_default = sample_diameter_bounds()
    context_text = synth_context_text(Lf, Wf, d_lo, d_hi)

    # For operator coords, we like ints
    L, W = clamp_int(Lf), clamp_int(Wf)

    # 2) Sample variants
    items: List[Dict[str, Any]] = []
    while len(items) < num_samples:
        item = sample_variant(L, W, d_lo, d_hi, d_default, context_text)
        items.append(item)

    # 3) Write single JSON file like crane generator
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    print(f"Wrote {len(items)} training items to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate drill goal variants (crane-compatible format).")
    p.add_argument("-o", "--output", default=os.path.join("data", "input_output", "generated_goals.json"))
    p.add_argument("-n", "--num", type=int, default=NUM_SAMPLES, help="Number of items to generate")
    args = p.parse_args()
    generate(args.output, num_samples=args.num)


if __name__ == "__main__":
    main()
