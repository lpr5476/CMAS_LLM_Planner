#!/usr/bin/env python
"""
augment_mr_dataset.py

Generic augmentation for MR records by:
  1) Randomizing geometry in context (Loc/offsetLoc) and propagating to goal.
  2) Randomizing numeric variables that are shared between the instruction
     and goal_canonical, using domains inferred from context.

Matching guarantees:
- For geometry:
    Loc(x=..., y=..., z=...) and offsetLoc(...) in context are matched to
    goal_canonical.variables.Loc / offsetLoc dicts with the same triple.
- For text-bound numeric variables:
    A numeric leaf in goal_canonical is only randomized if:
      * its numeric value appears as a unique numeric token in
        original_instruction, AND
      * a safe domain can be inferred from context (part dims or interface
        REAL=[min,max]).

The script keeps sampling until it reaches a target total number of records
or it can no longer augment anything.

Input JSONL record schema (per line):
{
  "id": "0",
  "source_index": 0,
  "goal_type": "single",
  "context": "... CMAS-like text ...",
  "goal_canonical": { ... },
  "goal_canonical_simple": "...",
  "original_instruction": "...",
  "instruction_variants": { ... }  # preserved but not modified here
}

Usage (example):
  python augment_mr_dataset.py \
      --input mr_with_vars.jsonl \
      --output mr_with_vars_aug.jsonl \
      --target-size 5000 \
      --max-epochs 200 \
      --seed 42
"""

import argparse
import copy
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Tuple


# ==================== FLOAT PRECISION CONTROL ====================

# Cap all generated/written floats to <= 4 decimals
MAX_DECIMALS = 4

def round4(x: float) -> float:
    return round(float(x), MAX_DECIMALS)

def fmt4(x: float) -> str:
    """String form with up to 4 decimals, trimming trailing zeros."""
    s = f"{float(x):.{MAX_DECIMALS}f}"
    return s.rstrip("0").rstrip(".")

def round_floats_in_obj(obj: Any) -> Any:
    """Recursively round all floats in nested dict/list structures to 4 decimals."""
    if isinstance(obj, dict):
        return {k: round_floats_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats_in_obj(v) for v in obj]
    if isinstance(obj, float):
        return round4(obj)
    return obj


# ==================== IO ====================

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


def save_records_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ==================== GEOMETRY (Loc/offsetLoc + Location/OffsetLocation) ====================

# 3D: kitting screws
GEOM3D_PATTERN = re.compile(
    r"(Loc|offsetLoc)\s*\(\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)\s*,\s*z\s*=\s*([0-9.]+)\s*\)"
)

# 2D: crane resources
GEOM2D_PATTERN = re.compile(
    r"(Location|OffsetLocation)\s*\(\s*x\s*=\s*([0-9.-]+)\s*,\s*y\s*=\s*([0-9.-]+)\s*\)"
)


# ---------- 3D helpers (Loc / offsetLoc) ----------

def extract_geom3d_entries(context: str) -> List[Dict[str, Any]]:
    """
    Extract Loc/offsetLoc triples from context.

    Returns entries like:
      {
        "kind": "Loc" or "offsetLoc",
        "x": float, "y": float, "z": float,
        "old_text": "Loc(x=0.5, y=0.53, z=0.08)"
      }
    """
    entries: List[Dict[str, Any]] = []
    for m in GEOM3D_PATTERN.finditer(context):
        kind = m.group(1)
        x = float(m.group(2))
        y = float(m.group(3))
        z = float(m.group(4))
        old_text = m.group(0)
        entries.append({
            "kind": kind,
            "x": x,
            "y": y,
            "z": z,
            "old_text": old_text,
        })
    return entries


def compute_geom3d_domains(entries: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    xs = [e["x"] for e in entries]
    ys = [e["y"] for e in entries]
    zs = [e["z"] for e in entries]
    return {
        "x": (min(xs), max(xs)),
        "y": (min(ys), max(ys)),
        "z": (min(zs), max(zs)),
    }


def sample_geom3d(domains: Dict[str, Tuple[float, float]], rng: random.Random) -> Tuple[float, float, float]:
    (xmin, xmax) = domains["x"]
    (ymin, ymax) = domains["y"]
    (zmin, zmax) = domains["z"]
    x_new = round4(rng.uniform(xmin, xmax))
    y_new = round4(rng.uniform(ymin, ymax))
    z_new = round4(rng.uniform(zmin, zmax))
    return x_new, y_new, z_new


def update_goal_xyz_with_mapping(
    obj: Any,
    mapping3d: Dict[Tuple[float, float, float], Tuple[float, float, float]]
) -> Any:
    """
    Recursively walk goal_canonical and replace any dict with numeric x,y,z
    that matches an old triple in mapping3d.
    """
    if isinstance(obj, dict):
        # If this dict looks like a 3D coord
        if set(obj.keys()) >= {"x", "y", "z"}:
            try:
                ox = float(obj["x"])
                oy = float(obj["y"])
                oz = float(obj["z"])
                key = (ox, oy, oz)
            except (TypeError, ValueError):
                key = None
            if key in mapping3d:
                nx, ny, nz = mapping3d[key]
                obj["x"] = round4(nx)
                obj["y"] = round4(ny)
                obj["z"] = round4(nz)

        # Recurse into children
        for k, v in obj.items():
            obj[k] = update_goal_xyz_with_mapping(v, mapping3d)
        return obj
    elif isinstance(obj, list):
        return [update_goal_xyz_with_mapping(x, mapping3d) for x in obj]
    else:
        return obj


def update_goal_simple_loc(
    gcs: str,
    mapping3d: Dict[Tuple[float, float, float], Tuple[float, float, float]]
) -> str:
    """
    Update patterns like 'Loc=(0.5,0.53,0.08)' in goal_canonical_simple
    according to mapping3d. This is specific to the kitting prints.
    """
    out = gcs
    for (ox, oy, oz), (nx, ny, nz) in mapping3d.items():
        old = f"Loc=({ox:g},{oy:g},{oz:g})"
        new = f"Loc=({nx:.4f},{ny:.4f},{nz:.4f})"
        out = out.replace(old, new)
    return out


def augment_geometry3d_inplace(rec: Dict[str, Any], rng: random.Random) -> bool:
    """
    Randomize 3D Loc/offsetLoc geometry in-place for a single record (kitting).
    Returns True if anything changed.
    """
    context = rec.get("context", "")
    entries = extract_geom3d_entries(context)
    if not entries:
        return False

    domains = compute_geom3d_domains(entries)
    mapping3d: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
    new_context = context

    for e in entries:
        ox, oy, oz = e["x"], e["y"], e["z"]
        nx, ny, nz = sample_geom3d(domains, rng)
        mapping3d[(ox, oy, oz)] = (nx, ny, nz)

        old_text = e["old_text"]
        new_text = f"{e['kind']}(x={nx:.4f}, y={ny:.4f}, z={nz:.4f})"
        # Replace once to avoid messing identical triples multiple times
        new_context = new_context.replace(old_text, new_text, 1)

    rec["context"] = new_context

    # Update goal_canonical
    gc = rec.get("goal_canonical")
    if isinstance(gc, dict):
        gc = update_goal_xyz_with_mapping(gc, mapping3d)
        rec["goal_canonical"] = gc

    # Update goal_canonical_simple if present
    gcs = rec.get("goal_canonical_simple")
    if isinstance(gcs, str):
        rec["goal_canonical_simple"] = update_goal_simple_loc(gcs, mapping3d)

    return True


# ---------- 2D helpers (Location / OffsetLocation) ----------

def extract_geom2d_entries(context: str) -> List[Dict[str, Any]]:
    """
    Extract Location/OffsetLocation pairs from context (crane, etc.).

    Returns entries like:
      {
        "kind": "Location" or "OffsetLocation",
        "x": float, "y": float,
        "old_text": "Location(x=450, y=82)"
      }
    """
    entries: List[Dict[str, Any]] = []
    for m in GEOM2D_PATTERN.finditer(context):
        kind = m.group(1)
        x = float(m.group(2))
        y = float(m.group(3))
        old_text = m.group(0)
        entries.append({
            "kind": kind,
            "x": x,
            "y": y,
            "old_text": old_text,
        })
    return entries


def compute_geom2d_domains(entries: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    xs = [e["x"] for e in entries]
    ys = [e["y"] for e in entries]
    return {
        "x": (min(xs), max(xs)),
        "y": (min(ys), max(ys)),
    }


def sample_geom2d(domains: Dict[str, Tuple[float, float]], rng: random.Random, integer: bool = True) -> Tuple[float, float]:
    (xmin, xmax) = domains["x"]
    (ymin, ymax) = domains["y"]

    if integer:
        xmin_i, xmax_i = int(xmin), int(xmax)
        ymin_i, ymax_i = int(ymin), int(ymax)
        x_new = rng.randint(xmin_i, xmax_i)
        y_new = rng.randint(ymin_i, ymax_i)
    else:
        x_new = rng.uniform(xmin, xmax)
        y_new = rng.uniform(ymin, ymax)
    return x_new, y_new


def update_goal_xy_with_mapping(
    obj: Any,
    mapping2d: Dict[Tuple[float, float], Tuple[float, float]]
) -> Any:
    """
    Recursively walk goal_canonical and replace any dict with numeric x,y
    (no z) that matches an old (x,y) in mapping2d.
    This covers crane toLocation/offsetToLocation/fromLocation/OffsetFromLocation.
    """
    if isinstance(obj, dict):
        # If this dict looks like a 2D coord (x,y but no z)
        if "x" in obj and "y" in obj and "z" not in obj:
            try:
                ox = float(obj["x"])
                oy = float(obj["y"])
                key = (ox, oy)
            except (TypeError, ValueError):
                key = None
            if key in mapping2d:
                nx, ny = mapping2d[key]
                obj["x"] = nx
                obj["y"] = ny

        # Recurse into children
        for k, v in obj.items():
            obj[k] = update_goal_xy_with_mapping(v, mapping2d)
        return obj
    elif isinstance(obj, list):
        return [update_goal_xy_with_mapping(x, mapping2d) for x in obj]
    else:
        return obj


def update_goal_simple_xy(
    gcs: str,
    mapping2d: Dict[Tuple[float, float], Tuple[float, float]]
) -> str:
    """
    Update '(x,y)' coordinate pairs in goal_canonical_simple.
    We assume crane simple strings contain things like:
      OffsetFromLocation(55,200), fromLocation(55,82), ...

    We only touch the '(x,y)' part.
    """
    out = gcs
    for (ox, oy), (nx, ny) in mapping2d.items():
        # Match "(55,200)" exactly
        pattern = r"\(" + f"{ox:g}" + r"," + f"{oy:g}" + r"\)"
        repl = f"({int(nx)},{int(ny)})"
        out = re.sub(pattern, repl, out)
    return out


def augment_geometry2d_inplace(rec: Dict[str, Any], rng: random.Random) -> bool:
    """
    Randomize 2D Location/OffsetLocation geometry in-place for a single record (crane).
    Returns True if anything changed.
    """
    context = rec.get("context", "")
    entries = extract_geom2d_entries(context)
    if not entries:
        return False

    domains = compute_geom2d_domains(entries)
    mapping2d: Dict[Tuple[float, float], Tuple[float, float]] = {}
    new_context = context

    # Crane coordinates are integers in your models; keep them integral.
    for e in entries:
        ox, oy = e["x"], e["y"]
        nx, ny = sample_geom2d(domains, rng, integer=True)
        mapping2d[(ox, oy)] = (nx, ny)

        old_text = e["old_text"]
        new_text = f"{e['kind']}(x={int(nx)}, y={int(ny)})"
        new_context = new_context.replace(old_text, new_text, 1)

    rec["context"] = new_context

    # Update goal_canonical (toLocation/fromLocation/offsetToLocation/OffsetFromLocation)
    gc = rec.get("goal_canonical")
    if isinstance(gc, dict):
        gc = update_goal_xy_with_mapping(gc, mapping2d)
        rec["goal_canonical"] = gc

    # Update goal_canonical_simple if present
    gcs = rec.get("goal_canonical_simple")
    if isinstance(gcs, str):
        rec["goal_canonical_simple"] = update_goal_simple_xy(gcs, mapping2d)

    return True


# ---------- Combined geometry augmentation ----------

def augment_geometry_inplace(rec: Dict[str, Any], rng: random.Random) -> bool:
    """
    Entry point used by the main script.

    - For kitting records with Loc/offsetLoc: calls augment_geometry3d_inplace
    - For crane records with Location/OffsetLocation: calls augment_geometry2d_inplace

    Both can fire on the same record if it ever uses both patterns.
    """
    changed3d = augment_geometry3d_inplace(rec, rng)
    changed2d = augment_geometry2d_inplace(rec, rng)
    return changed3d or changed2d


# ==================== TEXT-BOUND NUMERIC VARIABLES ====================

def parse_part_dims(context: str) -> Dict[str, Dict[str, float]]:
    """
    Parse part geometry like:
      - NameTag    [PART] { Var: length=60, width=30 }

    Returns: { part_name: {"length": 60.0, "width": 30.0}, ... }
    """
    dims: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(
        r"-\s*([\w\s]+?)\s*\[PART\]\s*{[^}]*Var:\s*length\s*=\s*([\d.]+)\s*,\s*width\s*=\s*([\d.]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(context):
        name = m.group(1).strip()
        length = float(m.group(2))
        width = float(m.group(3))
        dims[name] = {"length": length, "width": width}
    return dims


def parse_interface_ranges(context: str) -> Dict[str, Dict[str, float]]:
    """
    Parse interface REAL ranges like:
      Diameter:REAL=[3,8]
      x:REAL=[,]  (ignored if bounds missing)

    Returns: { var_name: {"min": a, "max": b}, ... }
    """
    ranges: Dict[str, Dict[str, float]] = {}
    pattern = re.compile(r"(\w+)\s*:\s*REAL\s*=\s*\[\s*([^,]*?)\s*,\s*([^]]*?)\s*\]")
    for m in pattern.finditer(context):
        var_name = m.group(1)
        raw_min = m.group(2).strip()
        raw_max = m.group(3).strip()
        if not raw_min or not raw_max:
            continue
        try:
            vmin = float(raw_min)
            vmax = float(raw_max)
        except ValueError:
            continue
        ranges[var_name] = {"min": vmin, "max": vmax}
    return ranges


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def collect_numeric_leaves(
    obj: Any,
    path: Tuple[Any, ...] = ()
) -> List[Dict[str, Any]]:
    """
    Recursively collect numeric leaves from goal_canonical, returning
    a list of { "path": (..), "name": <leaf key>, "value": number }.

    We skip paths containing 'Loc' or 'offsetLoc' because those are
    handled by geometry augmentation.
    """
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = path + (k,)
            if is_number(v):
                # skip geometry handled elsewhere
                if "Loc" in new_path or "offsetLoc" in new_path:
                    continue
                out.append({"path": new_path, "name": k, "value": v})
            else:
                out.extend(collect_numeric_leaves(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = path + (i,)
            out.extend(collect_numeric_leaves(v, new_path))
    return out


def find_numeric_tokens(instr: str) -> List[Dict[str, Any]]:
    """
    Find numeric tokens in the instruction.

    Returns list of dicts:
      { "text": "10", "value": 10.0, "start": i, "end": j, "index": k }
    """
    tokens: List[Dict[str, Any]] = []
    pattern = re.compile(r"\d+(\.\d+)?")
    for idx, m in enumerate(pattern.finditer(instr)):
        text = m.group(0)
        try:
            val = float(text)
        except ValueError:
            continue
        tokens.append({
            "text": text,
            "value": val,
            "start": m.start(),
            "end": m.end(),
            "index": idx,
        })
    return tokens


def infer_domain_for_var(
    var_name: str,
    original_value: float,
    part_name: str,
    part_dims: Dict[str, Dict[str, float]],
    iface_ranges: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Infer a sampling domain for a numeric variable based on:
      - interface REAL ranges, or
      - part geometry (x,y inside NameTag length/width).

    Returns a dict like:
      { "type": "int"|"float", "min": ..., "max": ... }
    or {} if no safe domain.
    """
    # 1) Direct REAL domain from interface
    if var_name in iface_ranges:
        dom = iface_ranges[var_name]
        if float(original_value).is_integer():
            return {"type": "int", "min": int(dom["min"]), "max": int(dom["max"])}
        else:
            return {"type": "float", "min": dom["min"], "max": dom["max"]}

    # 2) Geometry-based domain for x/y
    if var_name in ("x", "y") and part_name in part_dims:
        dims = part_dims[part_name]
        if var_name == "x":
            maxv = dims["length"]
        else:
            maxv = dims["width"]
        if float(original_value).is_integer():
            return {"type": "int", "min": 0, "max": int(maxv)}
        else:
            return {"type": "float", "min": 0.0, "max": maxv}

    return {}


def sample_from_domain(dom: Dict[str, Any], rng: random.Random, original: float) -> float:
    """
    Sample a new value from a domain. Try to avoid returning the original
    value when the range allows.
    """
    t = dom.get("type", "float")
    vmin = dom["min"]
    vmax = dom["max"]

    if vmin == vmax:
        # no room to move
        return original

    if t == "int":
        vmin = int(vmin)
        vmax = int(vmax)
        # ensure at least one different value
        for _ in range(10):
            val = rng.randint(vmin, vmax)
            if val != int(original):
                return val
        return int(original)
    else:
        for _ in range(10):
            val = round4(rng.uniform(vmin, vmax))
            if abs(val - original) > 1e-6:
                return val
        return round4(original)


def set_at_path(obj: Any, path: Tuple[Any, ...], new_value: float) -> Any:
    """
    Set a numeric leaf at the given path in a nested dict/list structure.
    Mutates in place and returns obj.
    """
    if not path:
        return round4(new_value) if isinstance(new_value, float) else new_value
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    last = path[-1]
    if isinstance(new_value, float):
        cur[last] = round4(new_value)
    else:
        cur[last] = new_value
    return obj


def replace_tokens_in_instruction(instr: str, token_map: Dict[int, float]) -> str:
    """
    Given mapping {token_index -> new_value}, replace those numeric tokens
    with new values, preserving all other text.
    """
    tokens = find_numeric_tokens(instr)
    if not tokens:
        return instr

    replacements: List[Tuple[int, int, str]] = []
    for t in tokens:
        idx = t["index"]
        if idx not in token_map:
            continue
        new_val = token_map[idx]
        new_text = fmt4(new_val)
        replacements.append((t["start"], t["end"], new_text))

    if not replacements:
        return instr

    replacements.sort(key=lambda x: x[0])
    out_parts = []
    last_pos = 0
    for start, end, rep in replacements:
        out_parts.append(instr[last_pos:start])
        out_parts.append(rep)
        last_pos = end
    out_parts.append(instr[last_pos:])
    return "".join(out_parts)


def augment_numeric_inplace(rec: Dict[str, Any], rng: random.Random) -> bool:
    """
    Randomize numeric variables that are shared between
    original_instruction and goal_canonical, using domains inferred from context.

    Returns True if anything changed, False otherwise.
    """
    instr = (rec.get("original_instruction") or "").strip()
    if not instr:
        return False

    goal = rec.get("goal_canonical") or {}
    context = rec.get("context", "")
    part_name = goal.get("part", "")

    part_dims = parse_part_dims(context)
    iface_ranges = parse_interface_ranges(context)

    instr_tokens = find_numeric_tokens(instr)
    if not instr_tokens:
        return False

    # Map value -> token indices
    val_to_token_indices: Dict[float, List[int]] = {}
    for t in instr_tokens:
        v = t["value"]
        val_to_token_indices.setdefault(v, []).append(t["index"])

    leaves = collect_numeric_leaves(goal)

    candidates: List[Dict[str, Any]] = []
    for leaf in leaves:
        val = float(leaf["value"])
        if val not in val_to_token_indices:
            continue
        idxs = val_to_token_indices[val]
        if len(idxs) != 1:
            # ambiguous in text
            continue
        token_index = idxs[0]
        var_name = leaf["name"]
        dom = infer_domain_for_var(var_name, val, part_name, part_dims, iface_ranges)
        if not dom:
            continue
        candidates.append({
            "path": tuple(leaf["path"]),
            "var_name": var_name,
            "original_value": val,
            "token_index": token_index,
            "domain": dom,
        })

    if not candidates:
        return False

    # Sample new values
    token_map: Dict[int, float] = {}
    path_map: Dict[Tuple[Any, ...], float] = {}
    for c in candidates:
        new_val = sample_from_domain(c["domain"], rng, c["original_value"])
        if isinstance(new_val, float):
            new_val = round4(new_val)
        path_map[c["path"]] = new_val
        token_map[c["token_index"]] = new_val

    # Update goal_canonical
    for path, new_val in path_map.items():
        goal = set_at_path(goal, path, new_val)
    rec["goal_canonical"] = goal

    # Update original_instruction
    new_instr = replace_tokens_in_instruction(instr, token_map)
    rec["original_instruction"] = new_instr

    # Update goal_canonical_simple if present (best-effort)
    gcs = rec.get("goal_canonical_simple")
    if isinstance(gcs, str):
        updated = gcs
        for c in candidates:
            old_text = fmt4(c["original_value"])
            new_text = fmt4(path_map[c["path"]])
            # global replace is fine because we only picked unique tokens
            updated = updated.replace(old_text, new_text)
        rec["goal_canonical_simple"] = updated

    return True


# ==================== COMBINED AUGMENTATION ====================

def augment_one_record(base_rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Produce a single augmented copy of base_rec by:
      - randomizing Loc/offsetLoc geometry in context+goal (if present)
      - randomizing text-bound numeric variables in goal+instruction (if applicable)

    Returns a NEW record dict, or None if no changes are possible.
    """
    new_rec = copy.deepcopy(base_rec)
    changed_geom = augment_geometry_inplace(new_rec, rng)
    changed_num = augment_numeric_inplace(new_rec, rng)

    if not (changed_geom or changed_num):
        return None

    # Final hard cap: round all floats in goal_canonical to <= 4 decimals
    if isinstance(new_rec.get("goal_canonical"), dict):
        new_rec["goal_canonical"] = round_floats_in_obj(new_rec["goal_canonical"])

    new_rec["augmentation"] = {
        "geometry_randomization": changed_geom,
        "numeric_randomization": changed_num,
    }
    return new_rec


# ==================== MAIN (TARGET SIZE LOOP) ====================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment MR dataset by matching variables in context/goal/instruction "
            "and randomizing them safely until a target dataset size is reached."
        )
    )
    parser.add_argument("--input", required=True, help="Input MR JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--target-size",
        type=int,
        default=5000,
        help="Target minimum number of records after augmentation.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Max passes over base records while trying to grow the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base_records = load_records_jsonl(args.input)
    n_base = len(base_records)
    print(f"[INFO] Loaded {n_base} base records from {args.input}", file=sys.stderr)

    out_records: List[Dict[str, Any]] = []
    out_records.extend(base_records)

    # Track per-base ID augmentation counts to generate unique IDs
    aug_counts: Dict[str, int] = {}

    if len(out_records) >= args.target_size:
        print(
            f"[INFO] Already have {len(out_records)} >= target {args.target_size}, no augmentation needed.",
            file=sys.stderr,
        )
        save_records_jsonl(out_records, args.output)
        return

    epoch = 0
    while len(out_records) < args.target_size and epoch < args.max_epochs:
        epoch += 1
        print(
            f"[INFO] Augmentation epoch {epoch}, current size {len(out_records)}",
            file=sys.stderr,
        )
        added_this_epoch = 0

        for base_rec in base_records:
            if len(out_records) >= args.target_size:
                break

            new_rec = augment_one_record(base_rec, rng)
            if new_rec is None:
                continue

            base_id = str(base_rec.get("id", "base"))
            count = aug_counts.get(base_id, 0) + 1
            aug_counts[base_id] = count

            # Ensure ID uniqueness
            new_rec["id"] = f"{base_id}_aug{count}"

            out_records.append(new_rec)
            added_this_epoch += 1

        if added_this_epoch == 0:
            print(
                "[WARN] No new augmentations produced in this epoch; "
                "stopping early. You may not reach the target size.",
                file=sys.stderr,
            )
            break

    print(
        f"[INFO] Final dataset size: {len(out_records)} (target={args.target_size})",
        file=sys.stderr,
    )
    if len(out_records) < args.target_size:
        print(
            f"[WARN] Could not reach target size; only {len(out_records)} records generated.",
            file=sys.stderr,
        )

    save_records_jsonl(out_records, args.output)
    print(f"[INFO] Wrote augmented dataset to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
