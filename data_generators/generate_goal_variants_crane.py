#!/usr/bin/env python3
"""
Generate goal variants in canonical JSON and a compact semantic format.
Writes two files under data/input_output/:
 - generated_goals.json (training items with instruction/input/output)
 - generated_goals_semantic.json (compact semantic-only records)

Usage:
  python data_generators/generate_goal_variants_with_semantic.py --per-base 4

This script follows project conventions: small CLI, deterministic templates, no external deps.
See .github/copilot-instructions.md for guidance.
"""

import json
import os
import argparse
from itertools import product

# Deterministic small template sets (use actual skill names from ENV_SUMMARY/CONTEXT)
SYN_TRANSPORT = ["transportPart"]
SYN_RUN = ["RunProcess1", "RunProcess2"]
SYN_PACK = ["packing"]

ENV_SUMMARY = """
AGENTS
- Crane      [RESOURCE] { Description: Resource to transport products, Interfaces: CraneInterface, Skills: transportPart, Variables: atX=0, atY=0, inUse=0, setX=0, setY=0, targetX=0, targetY=0, vacuum=0 }
- Process1   [RESOURCE] { Description: Process 1, Interfaces: ProcessInterface, CraneInterface, Skills: RunProcess1, Variables: Location(x=450, y=82), OffsetLocation(x=450, y=200), processRunning=false, proximitySensor=false, startProcess=false }
- Process2   [RESOURCE] { Description: Process 1, Interfaces: ProcessInterface, Skills: RunProcess2, Variables: Location(x=650, y=82), OffsetLocation(x=650, y=200), processRunning=false, proximitySensor=false, startProcess=false }
- Sink       [RESOURCE] { Description: Used to pack and put away products, Interfaces: InterfaceSink, Skills: packing, Variables: Location(x=945, y=82), OffsetLocation(x=945, y=200) }
- Source1    [RESOURCE] { Description: Source 1 is used to deploy Product A, Interfaces: InterfaceSource, Variables: Location(x=55, y=82), OffsetLocation(x=55, y=200), sensor1=false }
- Source2    [RESOURCE] { Description: Source 2 is used to deploy Product B, Interfaces: InterfaceSource, Variables: Location(x=145, y=82), OffsetLocation(x=145, y=200), sensor2=false }
- ProductA   [PART] { }
- ProductB   [PART] { }

SKILLS
- packing   (agent=Sink) requires InterfaceSink()
- RunProcess1 (agent=Process1) requires ProcessInterface()
- RunProcess2 (agent=Process2) requires ProcessInterface()
- transportPart (agent=Crane) requires CraneInterface(toLocation(x, y),offsetToLocation(x, y),fromLocation(x, y),OffsetFromLocation(x, y))"""


# Structured context dict: agents -> {type, interfaces, skills, vars}
CONTEXT = {
    "Crane": {
        "type": "RESOURCE",
        "interfaces": ["CraneInterface"],
        "skills": ["transportPart"],
        "vars": {"atX":0, "atY":0, "inUse":0, "setX":0, "setY":0, "targetX":0, "targetY":0, "vacuum":0}
    },
    "Process1": {
        "type": "RESOURCE",
        "interfaces": ["ProcessInterface","CraneInterface"],
        "skills": ["RunProcess1"],
        "vars": {"Location":{"x":450,"y":82}, "OffsetLocation":{"x":450,"y":200}, "processRunning":False, "proximitySensor":False, "startProcess":False, "x":450, "y":82}
    },
    "Process2": {
        "type": "RESOURCE",
        "interfaces": ["ProcessInterface"],
        "skills": ["RunProcess2"],
        "vars": {"Location":{"x":650,"y":82}, "OffsetLocation":{"x":650,"y":200}, "processRunning":False, "proximitySensor":False, "startProcess":False, "x":650, "y":82}
    },
    "Sink": {
        "type": "RESOURCE",
        "interfaces": ["InterfaceSink"],
        "skills": ["packing"],
        "vars": {"Location":{"x":945,"y":82}, "OffsetLocation":{"x":945,"y":200}, "x":945, "y":82}
    },
    "Source1": {
        "type": "RESOURCE",
        "interfaces": ["InterfaceSource"],
        "skills": [],
        "vars": {"Location":{"x":55,"y":82}, "OffsetLocation":{"x":55,"y":200}, "sensor1":False, "x":55, "y":82}
    },
    "Source2": {
        "type": "RESOURCE",
        "interfaces": ["InterfaceSource"],
        "skills": [],
        "vars": {"Location":{"x":145,"y":82}, "OffsetLocation":{"x":145,"y":200}, "sensor2":False, "x":145, "y":82}
    },
    "ProductA": {"type":"PART","interfaces":[],"skills":[],"vars":{}},
    "ProductB": {"type":"PART","interfaces":[],"skills":[],"vars":{}}
}

# Skills metadata
SKILLS = {
    "packing": {"agent":"Sink","requires":"InterfaceSink"},
    "RunProcess1": {"agent":"Process1","requires":"ProcessInterface"},
        "RunProcess2": {"agent":"Process2","requires":"ProcessInterface"},
        "transportPart": {"agent":"Crane","requires":"CraneInterface","params":["toLocation(x,y)","offsetToLocation(x,y)","fromLocation(x,y)","OffsetFromLocation(x,y)"]}
}


def build_goal(part, seq):
    return {"part": part, "sequence": seq}


def seq_to_goal_text(seq, part):
    pieces = []
    for step in seq:
        if step["action"] == "transport":
            pieces.append(f"{part} -> {step['to']}")
        elif step["action"] == "run_process":
            if pieces:
                pieces[-1] = pieces[-1] + f" ({step['skill']})"
    return " then ".join(pieces)


def seq_to_plan_steps(seq, part):
    steps = []
    for step in seq:
        skill = step.get('skill', '')
        # Transport steps -> concise human-friendly form
        if skill == 'transportPart' or step.get('variables'):
            frm = step.get('from')
            to = step.get('to')
            if frm and to:
                steps.append(f"Transport {part} from {frm} to {to}")
            elif to:
                steps.append(f"Transport {part} to {to}")
            elif frm:
                steps.append(f"Transport {part} from {frm}")
            else:
                steps.append(f"Transport {part}")
        # Run processes -> produce just the skill name (e.g. RunProcess1)
        elif isinstance(skill, str) and skill.startswith('RunProcess'):
            steps.append(skill)
        # Packing -> concise pack line
        elif skill == 'packing':
            steps.append(f"Pack {part}")
        # Fallback: preserve a readable check line if present
        elif 'var' in step:
            agent = step.get('interface') or step.get('agent', 'Unknown')
            steps.append(f"{agent}: check {step.get('var')}")
    return steps


def seq_to_semantic(seq, part):
    # Produce compact semantic goal format: "Agent:Goal(interface?,skill,params)"
    pieces = []
    for step in seq:
        if step.get("skill") == "transportPart" or step.get('variables'):
            # semantic form: Crane:transportPart(part, from=..., to=..., params)
            agent = step.get('interface') or SKILLS.get('transportPart', {}).get('agent', 'Crane')
            params = []
            frm = step.get('from')
            to = step.get('to')
            if frm:
                params.append(f"from={frm}")
            if to:
                params.append(f"to={to}")
            vars_map = step.get('variables', {})
            for key in ("fromLocation", "OffsetFromLocation", "toLocation", "offsetToLocation"):
                for k in sorted(vars_map.keys()):
                    if k.startswith(key):
                        v = vars_map[k]
                        params.append(f"{k}({v['x']},{v['y']})")
            pieces.append(f"{agent}:transportPart({part}, {', '.join(params)})")
        elif step.get('skill') == 'packing':
            agent = SKILLS.get('packing', {}).get('agent', step.get('interface', 'Sink'))
            pieces.append(f"{agent}:packing({part})")
        elif step.get("action") == "check":
            agent = step.get('agent')
            pieces.append(f"{agent}:check({step['var']})")
    return " ; ".join(pieces)


def make_paraphrases(part, seq, n_paraphrases=1):
    """Return up to n_paraphrases paraphrases summarizing the whole sequence.

    Build a deterministic, human-friendly sentence that follows the canonical
    sequence order: transport steps, process runs, final transport, packing.
    If n_paraphrases > 1 produce simple rephrasings until the quota is met.
    """
    pieces = []
    for step in seq:
        skill = step.get('skill', '')
        # Transport -> human-friendly transport phrase
        if skill == 'transportPart' or step.get('variables'):
            frm = step.get('from')
            to = step.get('to')
            if frm and to:
                pieces.append(f"transportPart {part} from {frm} to {to}")
            elif to:
                pieces.append(f"transportPart {part} to {to}")
            elif frm:
                pieces.append(f"transportPart {part} from {frm}")
            else:
                pieces.append(f"transportPart {part}")
        elif isinstance(skill, str) and skill.startswith('RunProcess'):
            pieces.append(skill)
        elif skill == 'packing':
            pieces.append(f"packing {part}")

    if not pieces:
        base = f"Move {part} as required."
    else:
        base = ", then ".join(pieces) + "."

    paraphrases = [base]

    # Simple deterministic variants for n_paraphrases > 1
    if n_paraphrases > 1:
        # variant 1: use 'and then' instead of ', then'
        v1 = base.replace(', then', ' and then')
        if v1 not in paraphrases:
            paraphrases.append(v1)
        # variant 2: join run processes with ' and ' if there are multiple
        runs = [s for s in seq if s.get('skill','').startswith('RunProcess')]
        if len(runs) > 1:
            # create a variant that joins runs into a single clause
            run_names = ' and '.join([r.get('skill') for r in runs])
            pieces2 = []
            for step in seq:
                if step.get('skill','').startswith('RunProcess'):
                    # skip individual runs
                    if run_names:
                        pieces2.append(run_names)
                        run_names = ''
                else:
                    skill = step.get('skill','')
                    if skill == 'transportPart' or step.get('variables'):
                        frm = step.get('from')
                        to = step.get('to')
                        if frm and to:
                            pieces2.append(f"transportPart {part} from {frm} to {to}")
                        elif to:
                            pieces2.append(f"transportPart {part} to {to}")
                        elif frm:
                            pieces2.append(f"transportPart {part} from {frm}")
                    elif skill == 'packing':
                        pieces2.append(f"packing {part}")
            v2 = ', then '.join(pieces2) + '.' if pieces2 else base
            if v2 not in paraphrases:
                paraphrases.append(v2)

    # Trim to requested count and preserve order
    result = []
    for p in paraphrases:
        if len(result) >= n_paraphrases:
            break
        if p not in result:
            result.append(p)
    return result


def generate(output_path, semantic_path=None, num_per_base=1):
    items = []
    semantic_items = []
    parts = ["ProductA", "ProductB"]
    base_process_orders = [
        ["Process1"],
        ["Process2"],
        ["Process1","Process2"],
        ["Process1","Process2"],
        ["Process1","Process2", "Process1"]

    ]
    source_opts = ["Source1", "Source2"]

    def build_defined_vars_for_seq(seq):
        defined = {}
        tcnt = 0
        for step in seq:
            if step.get("action") == "transport":
                tcnt += 1
                to = step.get("to")
                frm = step.get("from")
                to_loc = CONTEXT.get(to, {}).get("vars", {}).get("Location")
                to_off = CONTEXT.get(to, {}).get("vars", {}).get("OffsetLocation")
                from_loc = None
                from_off = None
                if frm:
                    from_loc = CONTEXT.get(frm, {}).get("vars", {}).get("Location")
                    from_off = CONTEXT.get(frm, {}).get("vars", {}).get("OffsetLocation")
                suffix = f"_{tcnt}" if tcnt > 1 else ""
                if to_loc is not None:
                    defined[f"toLocation{suffix}"] = to_loc
                if to_off is not None:
                    defined[f"offsetToLocation{suffix}"] = to_off
                if from_loc is not None:
                    defined[f"fromLocation{suffix}"] = from_loc
                if from_off is not None:
                    defined[f"OffsetFromLocation{suffix}"] = from_off
        return defined

    def build_variable_goal_for_seq(seq):
        # Populate variable goal fields expected by skills: toLocation, offsetToLocation,
        # fromLocation, OffsetFromLocation — add numeric suffixes if multiple transports.
        vars_goal = {}
        for step in seq:
            # detect transport steps by skill or variables
            if step.get('skill') == 'transportPart' or step.get('variables'):
                to = step.get("to")
                frm = step.get("from")
                to_loc = CONTEXT.get(to, {}).get("vars", {}).get("Location")
                to_off = CONTEXT.get(to, {}).get("vars", {}).get("OffsetLocation")
                from_loc = None
                from_off = None
                if frm:
                    from_loc = CONTEXT.get(frm, {}).get("vars", {}).get("Location")
                    from_off = CONTEXT.get(frm, {}).get("vars", {}).get("OffsetLocation")
                # Use base keys (no numeric suffixes) so they match the skill/interface signature
                if to_loc is not None:
                    vars_goal["toLocation"] = {"x": to_loc.get("x"), "y": to_loc.get("y")}
                if to_off is not None:
                    vars_goal["offsetToLocation"] = {"x": to_off.get("x"), "y": to_off.get("y")}
                if from_loc is not None:
                    vars_goal["fromLocation"] = {"x": from_loc.get("x"), "y": from_loc.get("y")}
                if from_off is not None:
                    vars_goal["OffsetFromLocation"] = {"x": from_off.get("x"), "y": from_off.get("y")}
        return vars_goal

    def build_goal_canonical_simple(goal_obj):
        # compact string: Part: Transport(key(x,y), ...)
        part_name = goal_obj.get('part')
        seq = goal_obj.get('sequence', [])
        # Build one fragment per canonical step, using the skill name and parameters.
        fragments = []
        for step in seq:
            skill = step.get('skill')
            # transportPart -> include part, from/to and transport variables
            if skill == 'transportPart' or step.get('variables'):
                vars_map = step.get('variables', {})
                params = []
                # append location params in predictable order
                for k in sorted(vars_map.keys()):
                    v = vars_map[k]
                    params.append(f"{k}({v['x']},{v['y']})")
                # Do not include the part name in simple output for transportPart
                fragments.append(f"transportPart({', '.join(params)})")
            elif isinstance(skill, str) and skill.startswith('RunProcess'):
                # run processes: include only the skill name
                fragments.append(f"{skill}")
            elif skill == 'packing':
                # packing: no parameters in simple output
                fragments.append(f"packing")
            else:
                # fallback: represent unknown step with its skill or a generic marker
                if skill:
                    fragments.append(f"{skill}({part_name})")
        if not fragments:
            base = f"{part_name}: <empty>"
        else:
            base = f"{part_name}: " + " ; ".join(fragments)
        # Append EOS token consistent with tokenizer config
        EOS = "<|eot_id|>"
        return (base + (" " + EOS if not base.endswith(EOS) else ""))

    # Transform the canonical goal sequence: replace agent with interface and
    # embed transport variables directly into the transport step.
    def transform_sequence_with_interfaces(seq):
        new_seq = []
        tcnt = 0
        for step in seq:
            new_step = dict(step)
            # replace agent with interface required by the skill
            if step.get('action') == 'transport' or step.get('skill') == 'transportPart':
                # transport uses transportPart requirements
                req_if = SKILLS.get('transportPart', {}).get('requires')
                if 'agent' in new_step:
                    new_step.pop('agent', None)
                new_step['interface'] = req_if

                # compute variables for this transport step (use base keys, no numeric suffixes)
                to = step.get('to')
                frm = step.get('from')
                to_loc = CONTEXT.get(to, {}).get('vars', {}).get('Location')
                to_off = CONTEXT.get(to, {}).get('vars', {}).get('OffsetLocation')
                from_loc = None
                from_off = None
                if frm:
                    from_loc = CONTEXT.get(frm, {}).get('vars', {}).get('Location')
                    from_off = CONTEXT.get(frm, {}).get('vars', {}).get('OffsetLocation')
                vars_here = {}
                if to_loc is not None:
                    vars_here["toLocation"] = {"x": to_loc.get('x'), "y": to_loc.get('y')}
                if to_off is not None:
                    vars_here["offsetToLocation"] = {"x": to_off.get('x'), "y": to_off.get('y')}
                if from_loc is not None:
                    vars_here["fromLocation"] = {"x": from_loc.get('x'), "y": from_loc.get('y')}
                if from_off is not None:
                    vars_here["OffsetFromLocation"] = {"x": from_off.get('x'), "y": from_off.get('y')}
                if vars_here:
                    new_step['variables'] = vars_here

            elif step.get('action') in ('run_process', 'run_skill') or step.get('skill','').startswith('RunProcess') or step.get('skill')=='packing':
                # determine required interface from the skill name
                skill_name = step.get('skill')
                req_if = SKILLS.get(skill_name, {}).get('requires')
                if 'agent' in new_step:
                    new_step.pop('agent', None)
                new_step['interface'] = req_if
                # run_process/run_skill do not include variables by default
            else:
                # other actions: still remove agent if present
                if 'agent' in new_step:
                    new_step.pop('agent', None)
            # drop the original action field from the canonical sequence
            new_step.pop('action', None)
            new_seq.append(new_step)
        return new_seq

    for part in parts:
        for proc_order in base_process_orders:
            # enforce product-specific source constraints:
            # ProductA only from Source1, ProductB only from Source2
            if part == 'ProductA':
                src_choices = ['Source1']
            elif part == 'ProductB':
                src_choices = ['Source2']
            else:
                src_choices = source_opts
            for src in src_choices:
                seq = []
                seq.append({"action":"transport","agent":"Crane","skill":"transportPart","from":src,"to":proc_order[0]})
                # append process runs, and insert transport steps between consecutive processes
                for i, p in enumerate(proc_order):
                    seq.append({"action":"run_process","agent":p,"skill": f"Run{p}"})
                    # if there's a next process, insert a transport from current to next
                    if i < len(proc_order) - 1:
                        next_proc = proc_order[i+1]
                        seq.append({"action":"transport","agent":"Crane","skill":"transportPart","from": p, "to": next_proc})
                # after processing, crane must pick up from the last process
                last_proc = proc_order[-1]
                seq.append({"action":"transport","agent":"Crane","skill":"transportPart","from": last_proc, "to":"Sink"})
                seq.append({"action":"run_skill","agent":"Sink","skill":"packing"})

                # transform sequence first so plans and semantics use the canonical (interface/vars) form
                transformed_seq = transform_sequence_with_interfaces(seq)
                # ensure no lingering 'action' keys anywhere (defensive)
                cleaned_seq = []
                for s in transformed_seq:
                    cleaned = {k: v for k, v in s.items() if k not in ('action','from','to')}
                    cleaned_seq.append(cleaned)

                goal = build_goal(part, cleaned_seq)
                plan = seq_to_plan_steps(cleaned_seq, part)
                paraphrases = make_paraphrases(part, cleaned_seq, n_paraphrases=num_per_base)

                # use the cleaned_transformed sequence for canonical output
                goal_with_seq = {
                    'part': goal.get('part'),
                    'sequence': cleaned_seq
                }
                for paraphrase in paraphrases:
                    item = {
                        "instruction": paraphrase,
                        "input": {
                            "context": ENV_SUMMARY
                        },
                        "output": {
                            "plan": plan,
                            "goal_canonical": goal_with_seq,
                            "goal_canonical_simple": build_goal_canonical_simple(goal_with_seq)
                        }
                    }
                    items.append(item)

                # also create a compact semantic-only record (one per base variant)
                # semantic-only: provide canonical goal with embedded variables and a compact string
                goal_seq = {
                    'part': goal.get('part'),
                    'sequence': cleaned_seq
                }
                semantic_items.append({"input": {"context": ENV_SUMMARY}, "output": {"goal_canonical": goal_seq, "goal_canonical_simple": build_goal_canonical_simple(goal_seq)}})

    # --- Combined-product variants (ProductA + ProductB together) ---
    # Create items where one instruction describes goals for both products
    # Build Cartesian combinations of process orders for A and B
    for proc_order_A in base_process_orders:
        for proc_order_B in base_process_orders:
            # fixed sources per product
            srcA = 'Source1'
            srcB = 'Source2'

            # build seq for ProductA
            seqA = []
            seqA.append({"action":"transport","agent":"Crane","skill":"transportPart","from":srcA,"to":proc_order_A[0]})
            for i, p in enumerate(proc_order_A):
                seqA.append({"action":"run_process","agent":p,"skill": f"Run{p}"})
                if i < len(proc_order_A) - 1:
                    next_proc = proc_order_A[i+1]
                    seqA.append({"action":"transport","agent":"Crane","skill":"transportPart","from": p, "to": next_proc})
            last_proc_A = proc_order_A[-1]
            seqA.append({"action":"transport","agent":"Crane","skill":"transportPart","from": last_proc_A, "to":"Sink"})
            seqA.append({"action":"run_skill","agent":"Sink","skill":"packing"})

            # build seq for ProductB
            seqB = []
            seqB.append({"action":"transport","agent":"Crane","skill":"transportPart","from":srcB,"to":proc_order_B[0]})
            for i, p in enumerate(proc_order_B):
                seqB.append({"action":"run_process","agent":p,"skill": f"Run{p}"})
                if i < len(proc_order_B) - 1:
                    next_proc = proc_order_B[i+1]
                    seqB.append({"action":"transport","agent":"Crane","skill":"transportPart","from": p, "to": next_proc})
            last_proc_B = proc_order_B[-1]
            seqB.append({"action":"transport","agent":"Crane","skill":"transportPart","from": last_proc_B, "to":"Sink"})
            seqB.append({"action":"run_skill","agent":"Sink","skill":"packing"})

            # transform and clean both sequences
            transformed_A = transform_sequence_with_interfaces(seqA)
            transformed_B = transform_sequence_with_interfaces(seqB)
            cleaned_A = [{k:v for k,v in s.items() if k not in ('action','from','to')} for s in transformed_A]
            cleaned_B = [{k:v for k,v in s.items() if k not in ('action','from','to')} for s in transformed_B]

            goalA = {'part':'ProductA','sequence': cleaned_A}
            goalB = {'part':'ProductB','sequence': cleaned_B}

            # combined plan is A then B (keeps order explicit)
            planA = seq_to_plan_steps(cleaned_A, 'ProductA')
            planB = seq_to_plan_steps(cleaned_B, 'ProductB')
            combined_plan = planA + planB

            # build paraphrases for each and combine (respect num_per_base)
            parasA = make_paraphrases('ProductA', cleaned_A, n_paraphrases=num_per_base)
            parasB = make_paraphrases('ProductB', cleaned_B, n_paraphrases=num_per_base)
            # pair paraphrases deterministically up to num_per_base
            for i in range(max(1, num_per_base)):
                pa = parasA[i % len(parasA)] if parasA else f"Move ProductA as required."
                pb = parasB[i % len(parasB)] if parasB else f"Move ProductB as required."
                combined_instruction = f"{pa} And also {pb}"

                # Build combined simple string with a single trailing EOS
                EOS = "<|eot_id|>"
                sa = build_goal_canonical_simple(goalA)
                sb = build_goal_canonical_simple(goalB)
                if sa.endswith(EOS):
                    sa = sa[: -len(EOS)].rstrip()
                if sb.endswith(EOS):
                    sb = sb[: -len(EOS)].rstrip()

                item = {
                    "instruction": combined_instruction,
                    "input": {"context": ENV_SUMMARY},
                    "output": {
                        "plan": combined_plan,
                        "goal_canonical_multi": {
                            "parts": ["ProductA","ProductB"],
                            "goals": [goalA, goalB]
                        },
                        "goal_canonical_simple": f"{sa} || {sb} {EOS}"
                    }
                }
                items.append(item)

    # ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # write canonical file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    # NOTE: semantic-only output has been intentionally removed (was producing a broken file).
    # write of semantic_items is skipped to avoid regenerating the separate semantic file.
    print(f"Wrote {len(items)} training items to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Generate goal variants (canonical JSON + semantic format).")
    p.add_argument("-o","--output", default="data/input_output/generated_goals_crane.json")
    p.add_argument("--per-base", type=int, default=1, help="NL paraphrases per base variant")
    args = p.parse_args()
    generate(args.output, num_per_base=args.per_base)

if __name__ == "__main__":
    main()
