# LOCAL LLM VERSION (replaces Ollama)
# Requires: pip install --upgrade llama-cpp-python

import json
import os
from typing import List, Any, Dict, Optional
from llama_cpp import Llama

# ---------- Local LLM config ----------
MODEL_PATH = os.getenv("LOCAL_LLM_PATH", "LLM_files/model.gguf")
_CTX_SIZE_DEFAULT = 4096

# Singleton model handle so we don't reload between calls
_LLAMA: Optional[Llama] = None


def _get_llama(ctx_size: int = _CTX_SIZE_DEFAULT, n_gpu_layers: int = 0, seed: Optional[int] = None) -> Llama:
    global _LLAMA
    if _LLAMA is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Local model not found at {MODEL_PATH}. "
                "Place a .gguf model at LLM_files/model.gguf or set LOCAL_LLM_PATH."
            )
        _LLAMA = Llama(
            model_path=MODEL_PATH,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            seed=seed
        )
    return _LLAMA


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Converts [{'role': 'system'|'user'|'assistant', 'content': ...}, ...] to a plain prompt.
    Adjust this to your model's chat template if needed.
    """
    sys = []
    convo = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            sys.append(content)
        elif role == "user":
            convo.append(f"User: {content}")
        elif role == "assistant":
            convo.append(f"Assistant: {content}")

    prefix = ""
    if sys:
        prefix = "\n".join(sys).strip() + "\n\n"

    prompt = prefix + "\n".join(convo) + "\nAssistant:"
    return prompt


def local_chat(messages: List[Dict[str, str]], options: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Minimal drop-in replacement for `ollama.chat(...)`.
    Returns: {'message': {'content': <string>}}
    """
    temperature = float(options.get("temperature", 0.6))
    top_p = float(options.get("top_p", 0.9))
    top_k = int(options.get("top_k", 40))
    repeat_penalty = float(options.get("repeat_penalty", 1.1))
    repeat_last_n = int(options.get("repeat_last_n", 64))
    num_predict = int(options.get("num_predict", 800))
    seed = options.get("seed", None)
    ctx_size = int(options.get("ctx_size", _CTX_SIZE_DEFAULT))
    n_gpu_layers = int(options.get("n_gpu_layers", 0))

    llm = _get_llama(ctx_size=ctx_size, n_gpu_layers=n_gpu_layers, seed=seed)
    prompt = _messages_to_prompt(messages)

    out = llm(
        prompt=prompt,
        max_tokens=num_predict,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        stop=["</s>", "\nUser:", "\nSystem:"],  # light guardrails; tweak as needed
    )

    # llama-cpp returns {'choices': [{'text': '...'}], ...}
    text = ""
    try:
        text = out["choices"][0]["text"]
    except Exception:
        text = ""

    return {"message": {"content": text}}


# ---------- Your original logic, now using local_chat ----------

def extract_instructions(goal_json: str) -> List[str]:
    """Read the crane JSON file and print/return all instruction strings.

    goal_json: path to a JSON file containing an array of items.
    Each item is expected to have an 'instruction' field (string).
    """
    try:
        with open(goal_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {goal_json}: {e}")
        return []

    items: List[Any] = data if isinstance(data, list) else []
    instructions: List[str] = []
    for it in items:
        if isinstance(it, dict):
            instr = it.get('instruction')
            if isinstance(instr, str) and instr.strip():
                instructions.append(instr.strip())

    return instructions


opts = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "num_predict": 800,
    "seed": 12345,
    # Optional local-LLM-specific:
    # "ctx_size": 4096,
    # "n_gpu_layers": 0,  # set >0 to offload some layers to GPU (if built with CUDA/Metal)
}


def make_variants(instruction: str) -> Dict[str, Any]:
    """Call local LLM to produce structured variants for a single instruction.

    Returns a parsed dict with keys: original_instruction, variants{humanized, paraphrases,
    scrambled, ungrammatical, swedish, nonsense}. If generation fails, returns an empty dict.
    """

    sys_message = """
You are a CMAS instruction variant generator.

GOAL
Given an OPERATOR_INSTRUCTION that lists CMAS goals/skills in order to be performed by an operator,
produce sets of linguistic variants. "Humanized" and "Paraphrases" must NOT use CMAS jargon (e.g., say “move ProductA to the first process” rather than “transportPart ProductA”), but must still imply the same steps and order.
  1) humanized (how a technician would naturally say it. This can include focusing on key actions, omitting redundant parts, etc.),
  2) paraphrases/alternates (more natural phrasings),
  3) scrambled (words jumbled but still recognizable),
  4) ungrammatical (misspellings, missing words, bad grammar),
  5) Swedish Language (translate the instruction to Swedish).
  6) nonsense (semantically invalid; cannot be mapped to any valid plan).

  YOU will create 2 variants for each of the 6 types above.
  If any numbers are present in the instruction, ensure they are preserved exactly in all variants.
  OUTPUT FORMAT (strict JSON) NO explanation or any other output:
{
  "original_instruction": "<echo OPERATOR_INSTRUCTION verbatim>",
  "variants": {
    "humanized":        ["...", "..."],
    "paraphrases":      ["...", "..."],
    "scrambled":        ["...", "..."],
    "ungrammatical":    ["...", "..."],
    "swedish":          ["...", "..."],
    "nonsense":         [{"text": "...", "unmappable": true}]
  }
}
"""

    messages = [
        {'role': 'system', 'content': sys_message},
        {'role': 'user', 'content': instruction},
    ]

    try:
        response = local_chat(messages=messages, options=opts)
    except Exception as e:
        print(f"Local chat failed: {e}")
        return {}

    content = (response.get('message', {}) or {}).get('content')
    print(content)
    if not content:
        return {}
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Failed to parse variants JSON: {e}")
        return {}


def flatten_variants_for_training(variant_obj: Dict[str, Any], include_swedish: bool = True) -> List[str]:
    """Extract a list of instruction strings suitable for training.

    Includes: humanized, paraphrases, scrambled, ungrammatical, and optionally swedish.
    Excludes: nonsense (by design, unmappable).
    Returns a de-duplicated list, preserving order where possible.
    """
    if not variant_obj:
        return []
    out: List[str] = []
    v = variant_obj.get('variants', {}) if isinstance(variant_obj, dict) else {}
    buckets = [
        v.get('humanized', []) or [],
        v.get('paraphrases', []) or [],
        v.get('scrambled', []) or [],
        v.get('ungrammatical', []) or [],
        (v.get('swedish', []) or []) if include_swedish else [],
    ]
    for bucket in buckets:
        for s in bucket:
            if isinstance(s, str):
                s2 = s.strip()
                if s2 and s2 not in out:
                    out.append(s2)
    return out


if __name__ == '__main__':
    # Simple demo: generate and print variants for the second instruction
    instructions = extract_instructions('data/input_output/generated_goals.json')
    if len(instructions) > 1:
        print(instructions[1])
        variants = make_variants(instructions[1])
        print(json.dumps(variants, indent=2, ensure_ascii=False))
