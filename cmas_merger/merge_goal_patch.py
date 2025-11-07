import json, uuid, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------ helpers: properties & traversal ------------------

def _prop(name: str, class_name: str, desc: str, local_value, ro: bool=False, inherit: bool=True, ptype: str="[STRINGEDIT]"):
    return [name, {
        "Name": name,
        "ClassName": class_name,
        "Description": desc,
        "CanInherit": inherit,
        "Override": True,
        "ReadOnly": ro,
        "ParentID": "",
        "PropertyType": ptype,
        "LocalValue": local_value
    }]

def _get_prop(node: Dict[str, Any], key: str):
    for pr in node.get("Properties", []):
        if pr and pr[0] == key:
            return pr[1].get("LocalValue")
    return None

def _set_prop(node: Dict[str, Any], key: str, value):
    for pr in node.get("Properties", []):
        if pr and pr[0] == key:
            pr[1]["LocalValue"] = value
            return
    node.setdefault("Properties", []).append(_prop(key, "cmas.ontology.Entity", f"Auto-added {key}", value))

def _walk_entities(node):
    if isinstance(node, dict):
        if node.get("Container") == "Entity":
            yield node
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _walk_entities(v)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                yield from _walk_entities(v)

def _is_part_agent(ent: Dict[str, Any]) -> bool:
    et = ent.get("EntityType") or _get_prop(ent, "Type") or ""
    if str(et).upper() != "AGENT":
        return False
    abt = _get_prop(ent, "AgentBaseType") or ent.get("AgentBaseType") or ""
    return str(abt).upper() == "PART"

def _canonical_name(ent: Dict[str, Any]) -> str:
    return (_get_prop(ent, "Name") or ent.get("Name") or "").strip()

def _entity_id(ent: Dict[str, Any]) -> Optional[str]:
    return _get_prop(ent, "ID") or ent.get("ID")

# ------------------ find Interface/Skill under the PART ------------------

def _part_interfaces(part_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for rel in part_node.get("Relations", []):
        if isinstance(rel, list) and rel and rel[0] == "Interfaces":
            out.extend(rel[1:])
    return out

def _find_interface_by_name(part_node: Dict[str, Any], iface_name: str) -> Optional[Dict[str, Any]]:
    if not iface_name:
        return None
    for iface in _part_interfaces(part_node):
        nm = _get_prop(iface, "Name") or iface.get("Name") or ""
        if nm.strip().lower() == iface_name.strip().lower():
            return iface
    return None

def _interface_skills(iface_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for rel in iface_node.get("Relations", []):
        if isinstance(rel, list) and rel and rel[0] == "Skills":
            out.extend(rel[1:])
    return out

def _find_skill_by_name_under_interface(iface_node: Dict[str, Any], skill_name: str) -> Optional[Dict[str, Any]]:
    if not iface_node or not skill_name:
        return None
    for sk in _interface_skills(iface_node):
        nm = _get_prop(sk, "Name") or sk.get("Name") or ""
        if nm.strip().lower() == skill_name.strip().lower():
            return sk
    return None

# ------------------ Goal entity builders ------------------

def _make_variable_entity(var_name: str, value):
    vtype = "REAL" if isinstance(value, (int, float)) else ("BOOLEAN" if isinstance(value, bool) else "STRING")
    props = [
        _prop("Parent","cmas.ontology.Entity","Parent object","", ptype="[PARENTEDITOR]"),
        _prop("Name","cmas.ontology.Entity","A user friendly name of the entity", var_name),
        _prop("ID","cmas.ontology.Entity","ID is a unique identifier for this specific entity", f"{uuid.uuid4()}-var", ptype="[IDEDIT, MODELLING, HIDEBASIC]"),
        _prop("Description","cmas.ontology.Entity","A user description of the entity",""),
        _prop("Type","cmas.ontology.Entity","Entity base type","VARIABLE", ro=True, inherit=False, ptype="[HIDEBASIC]"),
        _prop("VariableType","cmas.ontology.variables.Variable","Type of variable","PRIMITIVE", ro=True, inherit=False, ptype="[INHERITANCE, HIDEBASIC]"),
        _prop("ReadOnly","cmas.ontology.variables.Variable","Read only variable","false"),
        _prop("StoreLongTerm","cmas.ontology.variables.Variable","Store the value in a database","false"),
        _prop("Source","cmas.ontology.variables.Variable","Source field for an adapter",""),
        _prop("Address","cmas.ontology.variables.Variable","Address field for an adapter",""),
        _prop("Error","cmas.ontology.variables.Variable","Variable error","false", ptype="[INHERITANCE, HIDE]"),
        _prop("Value","cmas.ontology.variables.Value","Value", value),
        _prop("ValueType","cmas.ontology.variables.Value","Type of value", vtype, ro=True, inherit=False),
        _prop("UpperBound","cmas.ontology.variables.Value","Upper bound",""),
        _prop("LowerBound","cmas.ontology.variables.Value","Lower bound",""),
        _prop("Unit","cmas.ontology.variables.Value","Value unit","")
    ]
    return {
        "Container": "Entity",
        "Protected": False,
        "Properties": props,
        "Relations": [],
        "EntityType": "VARIABLE",
        "TypeOfVariable": "PRIMITIVE",
        "ValueType": vtype
    }

def _make_goal_entity(goal: Dict[str, Any]) -> Dict[str, Any]:
    gid = goal.get("id") or str(uuid.uuid4())
    props = [
        _prop("Parent","cmas.ontology.Entity","Parent object","", ptype="[PARENTEDITOR]"),
        _prop("Name","cmas.ontology.Entity","A user friendly name of the entity", goal["name"]),
        _prop("ID","cmas.ontology.Entity","ID is a unique identifier for this specific entity", gid, ptype="[IDEDIT, MODELLING, HIDEBASIC]"),
        _prop("Description","cmas.ontology.Entity","A user description of the entity",""),
        _prop("Type","cmas.ontology.Entity","Entity base type","GOAL", ro=True, inherit=False, ptype="[HIDEBASIC]")
    ]
    var_entities = []
    for k, v in (goal.get("parameters") or {}).items():
        var_entities.append(_make_variable_entity(k, v))
    return {
        "Container": "Entity",
        "Protected": False,
        "Properties": props,
        "Relations": [["Variables", *var_entities]],
        "EntityType": "GOAL"
    }

def _goal_id(ent: Dict[str, Any]) -> Optional[str]:
    for pr in ent.get("Properties", []):
        if pr and pr[0] == "ID":
            return pr[1].get("LocalValue")
    return None

# ------------------ Processplan resolution ------------------

def _replace_goal_names_with_ids(sfc: Dict[str, Any], name_to_id: Dict[str, str]) -> Dict[str, Any]:
    import copy
    sfc = copy.deepcopy(sfc)
    def walk(node):
        if isinstance(node, dict):
            if node.get("Type") == "StatementGoal":
                if "GoalRefName" in node:
                    gname = node.pop("GoalRefName")
                    gid = name_to_id.get(gname)
                    if not gid:
                        raise ValueError(f"Goal name '{gname}' not found among created goals")
                    node["Goal"] = gid
            for v in list(node.values()):
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(sfc)
    return sfc

# ------------------ main merge ------------------

def merge_goal_patch(cmas_tree: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    # 1) locate PART
    agent_hint = patch.get("agent_hint") or {}
    part_by_id, part_by_name = {}, {}
    for ent in _walk_entities(cmas_tree):
        if _is_part_agent(ent):
            pid = _entity_id(ent)
            nm = _canonical_name(ent)
            if pid: part_by_id[pid] = ent
            if nm: part_by_name[nm.lower()] = ent
    part = None
    if agent_hint.get("id") and agent_hint["id"] in part_by_id:
        part = part_by_id[agent_hint["id"]]
    elif agent_hint.get("name") and agent_hint["name"].strip().lower() in part_by_name:
        part = part_by_name[agent_hint["name"].strip().lower()]
    else:
        raise ValueError("PART not found by agent_hint")

    # 2) wipe old Goals
    rels = part.get("Relations", [])
    rels = [r for r in rels if not (isinstance(r, list) and r and r[0] == "Goals")]
    part["Relations"] = rels

    # 3) build new Goals + capture name->id
    goals_rel = ["Goals"]
    name_to_id: Dict[str, str] = {}
    iface_template_keys = []  # copy keys that include 'interface'
    skill_template_keys = []  # copy keys with 'skill' or 'process'
    # If a template goal exists (first old goal), harvest its property key names
    # (We wiped above, so search before wiping in a copy? Not available now. That's OK; we'll default later.)

    # Resolve interface/skill IDs from patch (if present), else via PART
    def resolve_iface_name(goal_item: Dict[str, Any]) -> Optional[str]:
        ri = goal_item.get("requires_interface") or {}
        if ri.get("name"):
            return ri["name"]
        gname = goal_item.get("name") or ""
        prefix = "".join([c for c in gname if c.isalpha()])
        return prefix or None

    def resolve_skill_name(goal_item: Dict[str, Any]) -> Optional[str]:
        rs = goal_item.get("requires_skill") or {}
        if rs.get("name"):
            return rs["name"]
        return goal_item.get("processplan_name") or rs.get("name")


    for g in patch.get("goals", []):
        g_ent = _make_goal_entity(g)
        gid = _goal_id(g_ent)
        gname = next((pr[1]["LocalValue"] for pr in g_ent["Properties"] if pr[0]=="Name"), g.get("name"))
        name_to_id[gname] = gid

        # attach interface/skill bindings
        iface_name = resolve_iface_name(g)
        skill_name = resolve_skill_name(g)
        # Choose property keys: prefer template-like names; else defaults
        keys_iface = iface_template_keys or ["Interface"]
        keys_skill = skill_template_keys or ["ProcessPlanName"]
        if iface_name:
            for key in keys_iface:
                # remove any existing prop of same name
                g_ent["Properties"] = [pr for pr in g_ent["Properties"] if pr[0] != key]
                g_ent["Properties"].append(_prop(key, "cmas.ontology.interfaces.Goal", "Execution interface", iface_name))
        if skill_name:
            for key in keys_skill:
                g_ent["Properties"] = [pr for pr in g_ent["Properties"] if pr[0] != key]
                g_ent["Properties"].append(_prop(key, "cmas.ontology.interfaces.Goal", "Process plan / skill", skill_name))

        goals_rel.append(g_ent)

    part["Relations"].append(goals_rel)

    # 4) processplan install (if provided)
    if patch.get("processplan"):
        sfc = _replace_goal_names_with_ids(patch["processplan"], name_to_id)
        sfc_text = json.dumps(sfc, separators=(",", ":"))
        # write Processplan property
        updated = False
        for pr in part.get("Properties", []):
            if pr and pr[0] == "Processplan":
                pr[1]["LocalValue"] = sfc_text
                if "SFCEDITOR" not in pr[1].get("PropertyType",""):
                    pr[1]["PropertyType"] = "[SFCEDITOR, MODELLING]"
                updated = True
                break
        if not updated:
            part.setdefault("Properties", []).append([
                "Processplan",
                {
                    "Name": "Processplan",
                    "ClassName": "cmas.ontology.agents.Part",
                    "Description": "Open a SFC editior for the part process plan",
                    "CanInherit": True,
                    "Override": True,
                    "ReadOnly": False,
                    "ParentID": "",
                    "PropertyType": "[SFCEDITOR, MODELLING]",
                    "LocalValue": sfc_text
                }
            ])

    return cmas_tree

def write_copy(original_path: Path, updated_tree: Dict[str, Any], suffix: str = "_with_goals") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = original_path.with_name(original_path.stem + f"{suffix}_{ts}" + original_path.suffix)
    out_path.write_text(json.dumps(updated_tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge a GoalPatch (with processplan) into a CMAS tree for a PART; replace existing goals, install SFC, preserve original.")
    ap.add_argument("--cmas", required=True, help="Path to CMAS JSON (e.g. drillTXT.txt)")
    ap.add_argument("--patch", required=True, help="Path to GoalPatch JSON")
    ap.add_argument("--out-suffix", default="_with_goals", help="Suffix for new file name")
    args = ap.parse_args()

    cmas_path = Path(args.cmas)
    patch_path = Path(args.patch)

    cmas = json.loads(cmas_path.read_text(encoding="utf-8", errors="ignore"))
    patch = json.loads(patch_path.read_text(encoding="utf-8"))

    updated = merge_goal_patch(cmas, patch)
    out_path = write_copy(cmas_path, updated, suffix=args.out_suffix)
    print(str(out_path))

if __name__ == "__main__":
    main()