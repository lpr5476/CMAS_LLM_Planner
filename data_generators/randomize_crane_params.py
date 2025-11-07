#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
randomize_crane_params.py

Expand a CMAS crane dataset by randomizing x,y coordinates consistently
within each sample. Does NOT invent any skills, interfaces, or IDs—only
replaces numeric coordinate values already present.

- Reads an input JSON array like generated_goals_crane.json
- Writes an expanded JSON with randomized parameterized copies
- Keeps original items and appends randomized ones until target_count

Coordinates affected across:
  - output.goal_canonical.sequence[*].variables (Location / OffsetLocation)
  - output.goal_canonical_simple (reconstructed from canonical)
  - input.context (string replacements of anchor coordinates)

Usage:
  python randomize_crane_params.py \
    --input /path/generated_goals_crane.json \
    --output /path/generated_goals_crane_randomized.json \
    --target 1500 \
    --seed 42
"""
import json, argparse, random
from copy import deepcopy

# Old anchor coordinates expected in the base file
OLD_ANCHORS = {
    "Source1":  (55, 82),
    "Source2":  (145, 82),
    "Process1": (450, 82),
    "Process2": (650, 82),
    "Sink":     (945, 82),
}
OLD_OFFSET_Y = 200  # canonical offset Y used in file

# Ranges to sample NEW coordinates (integers). Keep left-to-right ordering.
NEW_RANGES = {
    "Source1":  {"x": (40, 90),   "y": (70, 120), "off_y": (190, 230)},
    "Source2":  {"x": (120, 180), "y": (70, 120), "off_y": (190, 230)},
    "Process1": {"x": (420, 500), "y": (70, 120), "off_y": (190, 230)},
    "Process2": {"x": (620, 700), "y": (70, 120), "off_y": (190, 230)},
    "Sink":     {"x": (900, 980), "y": (70, 120), "off_y": (190, 230)},
}

RES_ORDER = ["Source1","Source2","Process1","Process2","Sink"]


def _sample_new_coords():
    """Sample a consistent, ordered coordinate map for all resources."""
    new = {}
    last_x = -10**9
    for name in RES_ORDER:
        xr = NEW_RANGES[name]["x"]
        yr = NEW_RANGES[name]["y"]
        ory = NEW_RANGES[name]["off_y"]
        ok = False
        for _ in range(50):
            x = random.randint(*xr)
            if x > last_x:
                ok = True
                break
        if not ok:
            x = max(xr[0], last_x + 1)
        y = random.randint(*yr)
        oy = random.randint(*ory)
        new[name] = (x, y, oy)
        last_x = x
    return new


def _build_old_to_new_map(new_anchor_map):
    """Map old (x,y) -> new (x,y) and (x,OLD_OFFSET_Y) -> new off_y."""
    mapping_xy = {}
    mapping_offset_y = {}
    for name, (ox, oy) in OLD_ANCHORS.items():
        newx, newy, newoffy = new_anchor_map[name]
        mapping_xy[(ox, oy)] = (newx, newy)
        mapping_offset_y[(ox, OLD_OFFSET_Y)] = newoffy  # keyed by x + old off y
    return mapping_xy, mapping_offset_y


def _apply_mapping_to_goal(goal, mapping_xy, mapping_offy):
    """Mutate goal_canonical in place according to mappings."""
    seq = goal.get("sequence", [])
    for step in seq:
        vars = step.get("variables", {})
        if isinstance(vars, dict):
            for key, val in list(vars.items()):
                if isinstance(val, dict) and "x" in val and "y" in val:
                    x, y = val.get("x"), val.get("y")
                    if isinstance(x, int) and isinstance(y, int):
                        new = mapping_xy.get((x, y))
                        if new:
                            val["x"], val["y"] = new[0], new[1]
                        else:
                            # Offset locations: same x, y == OLD_OFFSET_Y
                            if y == OLD_OFFSET_Y:
                                new_off_y = mapping_offy.get((x, OLD_OFFSET_Y))
                                if new_off_y:
                                    val["y"] = new_off_y
    return goal


def _rebuild_simple(goal):
    """Re-generate goal_canonical_simple string from canonical structure."""
    part = goal.get("part", "ProductA")
    chunks = []
    for step in goal.get("sequence", []):
        skill = step.get("skill")
        if skill == "transportPart":
            v = step.get("variables", {})
            of = v.get("OffsetFromLocation", {})
            fr = v.get("fromLocation", {})
            ot = v.get("offsetToLocation", {})
            to = v.get("toLocation", {})
            seg = (
                f"transportPart(OffsetFromLocation({of.get('x')},{of.get('y')}), "
                f"fromLocation({fr.get('x')},{fr.get('y')}), "
                f"offsetToLocation({ot.get('x')},{ot.get('y')}), "
                f"toLocation({to.get('x')},{to.get('y')}))"
            )
            chunks.append(seg)
        elif isinstance(skill, str):
            chunks.append(skill)
    base = f"{part}: " + " ; ".join(chunks)
    EOS = "<|eot_id|>"
    return base + (" " + EOS if not base.endswith(EOS) else "")


def _replace_context_coords(context_text, new_anchor_map):
    """
    Replace numeric coordinates in the AGENTS context for the known resources.
    Only rewrites:
      - Location(x=?, y=?)
      - OffsetLocation(x=?, y=?)
      - trailing x=?, y=? pairs on the same resource lines
    """
    s = context_text
    # direct replacements for exact anchor pairs found in the base file
    for name, (oldx, oldy) in OLD_ANCHORS.items():
        newx, newy, newoffy = new_anchor_map[name]
        s = s.replace(f"Location(x={oldx}, y={oldy})", f"Location(x={newx}, y={newy})")
        s = s.replace(f"OffsetLocation(x={oldx}, y={OLD_OFFSET_Y})", f"OffsetLocation(x={newx}, y={newoffy})")
        s = s.replace(f"x={oldx}, y={oldy}", f"x={newx}, y={newy}")
    return s


def expand(input_path, output_path, target_count=1500, seed=42):
    random.seed(seed)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array.")

    # Select only items that have structured outputs (others are passed through but not replicated)
    base = []
    for it in data:
        if isinstance(it, dict) and isinstance(it.get("output"), dict) and "goal_canonical" in it["output"]:
            base.append(it)

    out = list(data)  # keep originals first
    need = max(0, target_count - len(out))
    i = 0
    while i < need and base:
        src = deepcopy(base[i % len(base)])

        # sample new coordinate map
        new_map = _sample_new_coords()
        mapping_xy, mapping_offy = _build_old_to_new_map(new_map)

        # update goal canonical in place
        goal = src["output"]["goal_canonical"]
        _apply_mapping_to_goal(goal, mapping_xy, mapping_offy)

        # rebuild simple string
        src["output"]["goal_canonical_simple"] = _rebuild_simple(goal)

        # update input context string
        ctx = src.get("input", {}).get("context")
        if isinstance(ctx, str):
            src["input"]["context"] = _replace_context_coords(ctx, new_map)

        out.append(src)
        i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} items to {output_path}")
    return len(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input_output/generated_goals_crane_expanded.json")
    ap.add_argument("--output", default="data/input_output/generated_goals_crane_randomized.json")
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    expand(args.input, args.output, args.target, args.seed)
