"""Shared utilities: config, seeds, IO, model loading, answer parsing."""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = path or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Lets the `data` folder live outside the repo (e.g. a mounted Google Drive
    # path in Colab) without touching any call site: every data path is built
    # as ROOT / cfg["paths"]["data_dir"], and pathlib discards ROOT whenever
    # data_dir is itself absolute.
    data_dir_override = os.environ.get("MINI_PROJECT_DATA_DIR")
    if data_dir_override:
        cfg["paths"]["data_dir"] = data_dir_override
    return cfg


def load_prompt_config(path=None):
    path = path or ROOT / "configs" / "prompt.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


# ---------- JSON / JSONL IO ----------

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def append_jsonl(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_ids(path):
    return {r.get("id") for r in read_jsonl(path)}


def read_records(path):
    """Read a list-of-records file whether it is .json (array) or .jsonl."""
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return load_json(path)


# ---------- Model loading ----------

_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
}


def _tokenizer_id_for(model_id, hf_token=None):
    """Use model_id's tokenizer, or the base model's if the checkpoint has none.

    Tianye88/Qwen2.5-1.5B-Instruct-Medical-cpt-sft-v1 ships weights and
    config.json only. Its vocab matches Qwen/Qwen2.5-1.5B-Instruct.
    """
    import json
    from huggingface_hub import hf_hub_download, list_repo_files

    files = set(list_repo_files(model_id, token=hf_token))
    if files & _TOKENIZER_FILES:
        return model_id
    cfg = json.load(open(hf_hub_download(model_id, "config.json", token=hf_token)))
    model_type = cfg.get("model_type", "")
    hidden = cfg.get("hidden_size")
    if model_type == "qwen2" and hidden == 1536:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    if model_type == "qwen2" and hidden == 896:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    if model_type == "qwen2":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    if model_type == "qwen3":
        return "Qwen/Qwen3-1.7B"
    if model_type == "llama":
        return "meta-llama/Llama-3.2-1B"
    raise OSError(f"{model_id} has no tokenizer files and no known base tokenizer")


def load_model_and_tokenizer(model_id, dtype="bfloat16", device_map="auto", hf_token=None):
    """Load an HF causal LM + tokenizer. dtype: bfloat16 | float16 | float32."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_id = _tokenizer_id_for(model_id, hf_token)
    if tok_id != model_id:
        print(f"[common] {model_id} has no tokenizer files; using {tok_id}")
    tok = AutoTokenizer.from_pretrained(tok_id, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        dtype_kw = {"dtype": torch_dtype}
    else:
        dtype_kw = {"torch_dtype": torch_dtype}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, token=hf_token, **dtype_kw
    )
    model.eval()
    return model, tok


# ---------- Answer parsing ----------

ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([A-Za-z])\)?\.?\s*$", re.IGNORECASE)
ANSWER_RE_ANY = re.compile(r"answer\s*[:\-]?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


def parse_answer_letter(text, valid_letters):
    """Extract the final 'Answer: X' letter from a generated trace.

    Returns the letter (uppercased) if it is in valid_letters, else None.
    """
    if not text:
        return None
    # Prefer matches at end of a line; scan all, keep the last valid one.
    best = None
    for m in ANSWER_RE_ANY.finditer(text):
        letter = m.group(1).upper()
        if letter in valid_letters:
            best = letter
    if best is not None:
        return best
    # Fallback: last standalone letter line like "A." or "(B)"
    for line in reversed(text.strip().splitlines()):
        m = re.fullmatch(r"\(?([A-Za-z])\)?\.?", line.strip())
        if m and m.group(1).upper() in valid_letters:
            return m.group(1).upper()
    return None


def letter_token_ids(tokenizer, letters):
    """Map each option letter to a single token id as it appears after 'Answer: '."""
    ids = {}
    for L in letters:
        chosen = None
        for cand in (f" {L}", L, f"({L})", f"{L}."):
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                chosen = toks[0]
                break
        if chosen is None:  # multi-token: use last sub-token
            chosen = tokenizer.encode(f" {L}", add_special_tokens=False)[-1]
        ids[L] = chosen
    return ids
