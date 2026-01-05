#!/usr/bin/env python3
"""
CMAS Semantic-WER (sWER) calculator.

Expected JSONL format (per line):
{
  "id": "...",
  "gold": "<json string>",
  "pred": "<json string>"
}

Tokenization (semantic):
- For each goal: PART=<part>
- For each step in sequence (in order):
    SKILL=<skill>
    IFACE=<interface>
    VAR=<flattened.path>=<value>   (variables flattened, keys sorted for canonicalization)

If JSON parsing fails (e.g., truncated/invalid JSON), we fall back to a best-effort regex tokenizer
so you still get a comparable WER signal + a parse-rate diagnostic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


# --- add these imports (near the top) ---
from json import JSONDecoder, JSONDecodeError

# --- add this decoder + helper (near your other helpers) ---
_DECODER = JSONDecoder()

def loads_first_json(s: str) -> Any:
    """
    Parse the *first* JSON value from a string and ignore trailing characters.
    Uses JSONDecoder.raw_decode under the hood.

    Handles:
      - leading whitespace
      - optional leading non-JSON junk (we scan to first '{' or '[')
      - trailing junk after the JSON value
    """
    if not isinstance(s, str):
        raise TypeError(f"loads_first_json expected str, got {type(s)}")

    s2 = s.lstrip()

    # If the model printed junk before JSON, jump to the first '{' or '['
    if not s2.startswith(("{", "[")):
        m = re.search(r"[\{\[]", s2)
        if not m:
            raise JSONDecodeError("No JSON object/array start found", s2, 0)
        s2 = s2[m.start():]

    obj, _end = _DECODER.raw_decode(s2)
    return obj



# -------------------------
# Tokenization
# -------------------------


def _flatten_vars(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten nested dict/list scalars into (path, value_str)."""
    out: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            v = obj[k]
            newp = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_vars(v, newp))
        return out

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            newp = f"{prefix}[{i}]"
            out.extend(_flatten_vars(v, newp))
        return out

    # scalar
    if isinstance(obj, float):
        val = format(obj, ".10g")  # stable float repr
    else:
        val = str(obj)
    out.append((prefix, val))
    return out


def tokenize_cmas_goal(goal_text: str) -> Tuple[List[str], bool]:
    """
    Returns (tokens, parsed_json_flag).
    parsed_json_flag=True means we successfully json.loads()'d and used semantic tokenization.
    """
    try:
        root = loads_first_json(goal_text)

        tokens: List[str] = []

        # Accept both shapes:
        # A) {"goals":[{part,sequence:[...]}], "parts":[...]}
        # B) {part,sequence:[...]}
        goals = None
        if isinstance(root, dict):
            goals = root.get("goals")
            if goals is None and "sequence" in root:
                goals = [root]

        if not isinstance(goals, list):
            goals = []

        for g in goals:
            if not isinstance(g, dict):
                continue

            part = g.get("part")
            if part is not None:
                tokens.append(f"PART={part}")

            seq = g.get("sequence", [])
            if not isinstance(seq, list):
                continue

            for step in seq:
                if not isinstance(step, dict):
                    continue

                if "skill" in step:
                    tokens.append(f"SKILL={step['skill']}")
                if "interface" in step:
                    tokens.append(f"IFACE={step['interface']}")

                vars_obj = step.get("variables", {})
                if isinstance(vars_obj, dict):
                    for path, val in _flatten_vars(vars_obj):
                        tokens.append(f"VAR={path}={val}")

        return tokens, True

    except Exception:
        # Fallback: regex tokenization over JSON-ish text
        tokens: List[str] = []

        for m in re.finditer(r'"part"\s*:\s*"([^"]+)"', goal_text):
            tokens.append(f"PART={m.group(1)}")
        for m in re.finditer(r'"skill"\s*:\s*"([^"]+)"', goal_text):
            tokens.append(f"SKILL={m.group(1)}")
        for m in re.finditer(r'"interface"\s*:\s*"([^"]+)"', goal_text):
            tokens.append(f"IFACE={m.group(1)}")

        # Best-effort variable leaf extraction (order of appearance).
        # Note: cannot reliably build full nested paths without valid JSON.
        for m in re.finditer(
            r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(?:"([^"]*)"|(-?\d+(?:\.\d+)?))',
            goal_text,
        ):
            k = m.group(1)
            if k in {"skill", "interface", "part", "goals", "sequence", "variables", "parts"}:
                continue
            v = m.group(2) if m.group(2) is not None else m.group(3)
            tokens.append(f"VAR={k}={v}")

        return tokens, False


# -------------------------
# WER (Levenshtein alignment)
# -------------------------

@dataclass
class WerCounts:
    S: int
    D: int
    I: int
    N: int

    @property
    def wer(self) -> float:
        return (self.S + self.D + self.I) / (self.N if self.N > 0 else 1)


def wer_counts(ref: List[str], hyp: List[str]) -> WerCounts:
    """Compute WER counts (S/D/I) via DP alignment."""
    n, m = len(ref), len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]  # 'ok','sub','del','ins'

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"
    back[0][0] = "ok"

    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            hj = hyp[j - 1]
            if ri == hj:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "ok"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dele = dp[i - 1][j] + 1
                best = min(sub, ins, dele)
                dp[i][j] = best
                back[i][j] = "sub" if best == sub else ("ins" if best == ins else "del")

    # backtrace
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "ok":
            i -= 1
            j -= 1
        elif op == "sub":
            S += 1
            i -= 1
            j -= 1
        elif op == "del":
            D += 1
            i -= 1
        elif op == "ins":
            I += 1
            j -= 1
        else:
            break

    return WerCounts(S=S, D=D, I=I, N=n)


# -------------------------
# File runner
# -------------------------

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_jsonl(path: str) -> Dict[str, Any]:
    totalS = totalD = totalI = totalN = 0
    parsed_gold = 0
    parsed_pred = 0
    n = 0

    for rec in iter_jsonl(path):
        gold_tokens, gold_ok = tokenize_cmas_goal(rec["gold"])
        pred_tokens, pred_ok = tokenize_cmas_goal(rec["pred"])

        parsed_gold += int(gold_ok)
        parsed_pred += int(pred_ok)

        c = wer_counts(gold_tokens, pred_tokens)
        totalS += c.S
        totalD += c.D
        totalI += c.I
        totalN += c.N
        n += 1

    corpus_wer = (totalS + totalD + totalI) / (totalN if totalN > 0 else 1)

    return {
        "n": n,
        "corpus_wer": corpus_wer,
        "S": totalS,
        "D": totalD,
        "I": totalI,
        "N": totalN,
        "gold_json_parse_rate": parsed_gold / n if n else 0.0,
        "pred_json_parse_rate": parsed_pred / n if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to predictions JSONL with {id,gold,pred}.")
    args = ap.parse_args()

    out = score_jsonl(args.jsonl)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
