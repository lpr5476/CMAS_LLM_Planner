#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from collections import defaultdict
from typing import Any, Dict, List

Json = Any

# ---------- basic CMAS helpers ----------
def prop_get(node: Dict[str, Any], key: str):
    props = node.get("Properties")
    if isinstance(props, list):
        for item in props:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str) and item[0] == key:
                v = item[1]
                if isinstance(v, dict):
                    return v.get("LocalValue")
                return v
    return None

def get_entity_type(n: Dict[str, Any]) -> str:
    return (n.get("EntityType") or prop_get(n, "EntityType") or "").upper()

def get_agent_base_type(n: Dict[str, Any]) -> str:
    return (n.get("AgentBaseType") or prop_get(n, "AgentBaseType") or "").upper()

def get_name(n: Dict[str, Any]) -> str:
    return prop_get(n, "Name") or n.get("Name") or ""

def get_id(n: Dict[str, Any]) -> str:
    return prop_get(n, "ID") or n.get("ID") or ""

def get_description(n: Dict[str, Any]) -> str:
    return prop_get(n, "Description") or n.get("Description") or ""

def iter_relations_rows(n: Dict[str, Any]):
    """Yield (label, [payload1, payload2, ...]) even when a row contains multiple payloads."""
    for rel in n.get("Relations", []):
        if isinstance(rel, list) and len(rel) >= 1 and isinstance(rel[0], str):
            yield rel[0], rel[1:]

# ---------- ID index & resolution (ID or inline only; NO name resolution) ----------
def build_id_index(root: Json) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    def walk(x: Json):
        if isinstance(x, dict):
            nid = get_id(x)
            if nid: idx[nid] = x
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(root)
    return idx

def resolve_inline_or_id(obj: Json, id_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(obj, dict) and "LocalValue" in obj:
        return resolve_inline_or_id(obj["LocalValue"], id_index)
    if isinstance(obj, dict):
        if get_entity_type(obj):          # inline node
            return obj
        rid = obj.get("ID") or obj.get("RefID") or obj.get("$ref")
        if isinstance(rid, str) and rid in id_index:
            return id_index[rid]
        return {}
    if isinstance(obj, str) and obj in id_index:   # ID string
        return id_index[obj]
    return {}

# ---------- scanners ----------
def find_agents(root: Json) -> List[Dict[str, Any]]:
    agents: List[Dict[str, Any]] = []
    def walk(x: Json):
        if isinstance(x, dict):
            if get_entity_type(x) == "AGENT": agents.append(x)
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(root)
    return agents

def collect_variables_under(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect *all* VARIABLEs under `node`, including children of variable-objects.
    We recurse even when the current node is a VARIABLE to reach its 'Items'."""
    out: List[Dict[str, Any]] = []
    def walk(x: Json):
        if isinstance(x, dict):
            if get_entity_type(x) == "VARIABLE":
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(node)
    seen = set(); dedup: List[Dict[str, Any]] = []
    for var in out:
        # Prefer deduplication by name (if present). Some CMAS exports reuse IDs
        # which would incorrectly collapse distinct named variables (e.g., x and y).
        name = get_name(var)
        vid = get_id(var)
        key = name if name else vid
        if key and key not in seen:
            seen.add(key)
            dedup.append(var)
    return dedup


def collect_agent_vars_from_Variables_relation(agent_node: Dict[str, Any], id_index: Dict[str, Dict[str, Any]]):
    """Read ONLY the agent's own 'Variables' relation payloads (multi-payload aware)."""
    vars_nodes = []
    for label, payloads in iter_relations_rows(agent_node):
        if re.fullmatch(r'Variables', label, flags=re.IGNORECASE):
            for p in payloads:
                node = resolve_inline_or_id(p, id_index) or {}
                # collect VARIABLEs under this payload (or the node itself if it is a VARIABLE)
                col = collect_variables_under(node)
                if not col and get_entity_type(node) == "VARIABLE":
                    col = [node]
                vars_nodes.extend(col)
    # de-dup by name
    seen = set(); out = []
    for v in vars_nodes:
        nm = get_name(v)
        if nm and nm not in seen:
            seen.add(nm); out.append(v)
    return out

def fmt_agent_vars(vars_nodes: List[Dict[str, Any]]) -> str:
    # Render composite variable-objects as grouped children (e.g. Location(x,y)).
    # Prefer concrete scalar values when present (DefaultValue/Value or equal bounds).
    # Avoid printing a variable-object name as its own value (e.g. Location=Location).
    vals = {}
    tokens: List[str] = []
    # collect names to detect references to other variables
    names_set = {get_name(v) for v in vars_nodes if get_name(v)}
    # track scalar child names of composite variables to avoid duplicate prints later
    composite_children: set = set()

    for v in vars_nodes:
        nm = get_name(v)
        if not nm:
            continue

        # Composite variable -> render as grouped children with inline concrete values when available
        if is_composite_variable(v):
            kids = get_direct_child_variables(v)
            kid_reprs: List[str] = []
            # Determine ordering: spatial coords first
            kid_names = [get_name(c) for c in kids if get_name(c)]
            head = [k for k in kid_names if k and k.lower() in {"x", "y", "z"}]
            ordered_names = head + [k for k in kid_names if k not in head]
            # remember these children so we don't print them again as separate scalars
            for k in ordered_names:
                if k:
                    composite_children.add(k.lower())
            # Map names to node for value lookup
            name_to_node = {get_name(c): c for c in kids if get_name(c)}
            for k in ordered_names:
                node = name_to_node.get(k)
                if not node:
                    continue
                # try to extract a concrete scalar
                cv = prop_get(node, "DefaultValue") or prop_get(node, "Value")
                lo = prop_get(node, "LowerBound"); hi = prop_get(node, "UpperBound")
                if cv is not None and not (isinstance(cv, str) and cv in names_set):
                    kid_reprs.append(f"{k}={cv}")
                elif lo is not None and hi is not None and str(lo) == str(hi):
                    kid_reprs.append(f"{k}={lo}")
                else:
                    # fallback to name-only if no concrete value
                    kid_reprs.append(k)
            tokens.append(f"{nm}(" + ", ".join(kid_reprs) + ")")
            continue

        dv = prop_get(v, "DefaultValue") or prop_get(v, "Value")
        lo = prop_get(v, "LowerBound"); hi = prop_get(v, "UpperBound")

        # If dv is a reference to another variable (common for variable-objects), skip it
        if dv is not None:
            if isinstance(dv, str) and dv in names_set:
                # skip non-concrete reference
                pass
            elif isinstance(dv, dict):
                # try to extract a LocalValue if available and concrete
                lv = dv.get("LocalValue") if isinstance(dv, dict) else None
                if isinstance(lv, (str, int, float)) and (not (isinstance(lv, str) and lv in names_set)):
                    vals[nm] = lv
                else:
                    # skip complex/non-concrete DefaultValue
                    pass
            else:
                vals[nm] = dv
        elif lo is not None and hi is not None and str(lo) == str(hi):
            vals[nm] = lo
        else:
            # no concrete value; skip to avoid noisy variable-object-like prints
            pass

    # ordering: composite tokens first, then length/width/height, then remaining scalar vals sorted
    parts: List[str] = []
    parts.extend(tokens)
    # remove scalar entries that are children of composite variables (e.g., x,y when Location(x,y) exists)
    filtered_scalar_keys = [k for k in vals.keys() if (k and k.lower() not in composite_children)]
    scalar_order = [k for k in ["length", "width", "height"] if k in filtered_scalar_keys]
    scalar_rest = [k for k in sorted(filtered_scalar_keys) if k not in {"length", "width", "height"}]
    for k in scalar_order + scalar_rest:
        parts.append(f"{k}={vals[k]}")

    return ", ".join(parts)

def find_interfaces_under_agent(agent_node: Dict[str, Any], id_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, payloads in iter_relations_rows(agent_node):
        if not re.search(r'\binterfaces?\b', label, re.IGNORECASE):
            continue
        for p in payloads:
            node = resolve_inline_or_id(p, id_index)
            if get_entity_type(node) == "INTERFACE":
                out.append(node)
    # de-dup by ID, preserve order
    seen = set(); uniq = []
    for i in out:
        key = get_id(i) or id(i)
        if key not in seen:
            seen.add(key); uniq.append(i)
    return uniq

def find_skills_under_interface(iface_node: Dict[str, Any], id_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, payloads in iter_relations_rows(iface_node):
        if not re.search(r'\bskills?\b', label, re.IGNORECASE):
            continue
        for p in payloads:
            node = resolve_inline_or_id(p, id_index)
            if get_entity_type(node) == "PROCESSPLAN":
                out.append(node)
    # de-dup by id
    seen = set(); uniq = []
    for n in out:
        key = get_id(n) or id(n)
        if key not in seen:
            seen.add(key); uniq.append(n)
    return uniq

def fmt_interface_vars(vars_nodes: List[Dict[str, Any]]) -> str:
    items = []
    head, tail = [], []
    for v in vars_nodes:
        nm = get_name(v)
        vt = (prop_get(v, "ValueType") or v.get("ValueType") or "").upper()
        lo = prop_get(v, "LowerBound"); hi = prop_get(v, "UpperBound")
        (head if nm and nm.lower() in {"x","y","diameter"} else tail).append((nm, vt, lo, hi))
    head.sort(key=lambda t: ["x","y","diameter"].index(t[0].lower()))
    tail.sort(key=lambda t: (t[0] or "").lower())
    for nm, vt, lo, hi in head + tail:
        if vt == "REAL" and lo is not None and hi is not None:
            items.append(f"{nm}:REAL=[{lo},{hi}]")
        elif vt == "REAL":
            items.append(f"{nm}:REAL")
        else:
            items.append(f"{nm}")
    return ", ".join(items)
def is_composite_variable(var: Dict[str, Any]) -> bool:
    """True for variable-objects that own child VARIABLES via an 'Items' relation."""
    try:
        for label, payloads in iter_relations_rows(var):
            if str(label).lower() == "items":
                for p in payloads:
                    if isinstance(p, dict) and get_entity_type(p) == "VARIABLE":
                        return True
    except Exception:
        pass
    return False

def get_direct_child_variables(var_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return direct VARIABLE children of a composite variable (under its 'Items' relation)."""
    out: List[Dict[str, Any]] = []
    try:
        for label, payloads in iter_relations_rows(var_obj):
            if str(label).lower() == "items":
                for p in payloads:
                    if isinstance(p, dict) and get_entity_type(p) == "VARIABLE":
                        out.append(p)
    except Exception:
        pass
    return out

# ---------- main ----------
def summarize(path: str) -> str:
    data = json.load(open(path, "r", encoding="utf-8"))
    id_index = build_id_index(data)
    agents = find_agents(data)

    # Collections
    agent_rows = []   # (name, kind, iface_names[], skill_names[], var_str_or_empty, description)
    iface_rows = []   # (iface_name, owner_name, var_str)
    skill_rows = set()

    for a in agents:
        aname = get_name(a)
        kind = (get_agent_base_type(a) or "").upper()
        kind = kind if kind in ("RESOURCE","PART") else "AGENT"
        descr = get_description(a)

        # interfaces
        ifaces = find_interfaces_under_agent(a, id_index)
        iface_names = [get_name(i) for i in ifaces]

        # interface details + skills
        skills = []
        for i in ifaces:
            vars_nodes = collect_variables_under(i)
            iface_rows.append((get_name(i), aname, fmt_interface_vars(vars_nodes)))
            for sk in find_skills_under_interface(i, id_index):
                sname = get_name(sk)
                if sname:
                    skills.append(sname)
                    # param order same as interface vars
                    leaf_vars = [v for v in vars_nodes if not is_composite_variable(v)]
                    params = [get_name(v) for v in leaf_vars if get_name(v)]
                    head = [p for p in params if p.lower() in {"x","y","diameter"}]
                    tail = [p for p in params if p.lower() not in {"x","y","diameter"}]
                    params_ordered = head + sorted(tail, key=str.lower)
                                            # Replaced with grouped display of composite variables
                    tokens = []
                    for v in vars_nodes:
                        vname = get_name(v)
                        if not vname: continue
                        if is_composite_variable(v):
                            kids = get_direct_child_variables(v)
                            kid_names = [get_name(c) for c in kids if get_name(c)]
                            head = [k for k in kid_names if k and k.lower() in {"x","y","z"}]
                            kid_names = head + [k for k in kid_names if k not in head]
                            # join with comma+space for readability
                            tokens.append(f"{vname}(" + ", ".join(kid_names) + ")")
                        else:
                            tokens.append(vname)
                    simple = [t for t in tokens if "(" not in t]
                    composite = [t for t in tokens if "(" in t]
                    # collect child names from composites to avoid duplicating them as simple params
                    composite_children = set()
                    for comp in composite:
                        m = re.search(r"\((.*)\)", comp)
                        if m:
                            for ch in m.group(1).split(','):
                                composite_children.add(ch.strip().lower())

                    # Remove any simple tokens that appear as children of composite params
                    filtered_simple = [t for t in simple if t and t.lower() not in composite_children]
                    head_s = [t for t in filtered_simple if t.lower() in {"x","y","z"}]
                    tail_s = [t for t in filtered_simple if t not in head_s]
                    display = composite + head_s + sorted(tail_s, key=str.lower)
                    skill_rows.add((sname, aname, get_name(i), tuple(display)))
# agent-level variables (read from Variables relation for ALL agents)
        agent_vars_nodes = collect_agent_vars_from_Variables_relation(a, id_index)
        var_str = fmt_agent_vars(agent_vars_nodes)

        # stable
        iface_names = list(dict.fromkeys(iface_names))
        skills = sorted(set(skills), key=str.lower)
        agent_rows.append((aname, kind, iface_names, skills, var_str, descr))

    # sort
    agent_rows.sort(key=lambda t: ({"RESOURCE":0,"PART":1}.get(t[1],2), t[0].lower()))
    iface_rows.sort(key=lambda t: (t[0].lower(), t[1].lower()))
    skill_rows = sorted(skill_rows, key=lambda t: (t[0].lower(), t[1].lower(), t[2].lower()))

    # render
    lines = []
    lines.append(" AGENTS")
    for aname, kind, iface_names, skills, var_str, descr in agent_rows:
        if kind == "PART":
            # PARTs: show Description/Variables when present; otherwise empty braces
            pieces = []
            if descr:
                pieces.append(f"Description: {descr}")
            if var_str:
                pieces.append(f"Variables: {var_str}")
            inside = ", ".join(pieces)
            line = f"- {aname:<11}[{kind}] {{ {inside} }}" if inside else f"- {aname:<11}[{kind}] {{ }}"
        else:
            pieces = []
            if descr:
                pieces.append(f"Description: {descr}")
            if iface_names:
                pieces.append(f"Interfaces: {', '.join(iface_names)}")
            if skills:
                pieces.append(f"Skills: {', '.join(skills)}")
            if var_str:
                pieces.append(f"Variables: {var_str}")
            inside = ", ".join(pieces)
            # SINGLE closing brace — fixes the stray '}}'
            line = f"- {aname:<11}[{kind}] {{ {inside} }}" if inside else f"- {aname:<11}[{kind}] {{ }}"
        lines.append(line)

    # Interfaces are collected (iface_rows) but not printed as a separate block.

    lines.append("\nSKILLS")
    for sname, owner, iname, params in skill_rows:
        lines.append(f"- {sname:<9} (agent={owner}) requires {iname}({','.join(params)})")

    return "\n".join(lines)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize CMAS JSON (generic; multi-payload Relations; agent vars for all agents).")
    ap.add_argument("input", help="Path to CMAS file (e.g., /mnt/data/drill.cmas)")
    ap.add_argument("-o", "--output", help="Optional output path (default: stdout)", default=None)
    args = ap.parse_args()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summarize(args.input) + "\n")

    print(summarize(args.input))
