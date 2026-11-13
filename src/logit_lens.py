"""Step 7 - Logit lens probing (Phase 2a) on the N=100 subset.

For each question, at the token position immediately preceding the model's
final answer, project each layer's residual stream through the model's own
final unembedding matrix W_U (as if that layer were the last). Read off the
probability of each option-letter token across layers.

Looking for: does the eventually-chosen option overtake all distractors EARLY
(genuine internal computation) or only in the last layer(s) (late resolution
the CoT may be dressing up post-hoc)?

Deliverable per question: emergence layer = first layer where the chosen
option's probability exceeds every distractor; earliness = 1 - layer/L.

Ground truth: none external - the reference is the model's own emitted answer
(we run on cases where it matches MedQA gold as a sanity filter).
"""
import torch
from tqdm import tqdm

from .common import append_jsonl, letter_token_ids, read_jsonl_ids


def _arch_hint_for(model_id, hf_token=None):
    """Map a community fine-tune to the official TransformerLens architecture
    name of its base (same architecture => same hook names)."""
    import json as _json
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(model_id, "config.json", token=hf_token)
    cfg = _json.load(open(cfg_path))
    mt = cfg.get("model_type", "")
    n_params = cfg.get("num_hidden_layers", 0) * cfg.get("hidden_size", 0) ** 2
    approx_b = n_params / 1e9
    hints = {
        ("qwen2", 1.5): "Qwen/Qwen2.5-1.5B-Instruct",
        ("qwen2", 0.5): "Qwen/Qwen2.5-0.5B-Instruct",
        ("qwen3", 1.7): "Qwen/Qwen3-1.7B",
        ("llama", 1.0): "meta-llama/Llama-3.2-1B",
        ("llama", 3.0): "meta-llama/Llama-3.2-3B",
        ("gpt2", 0.0): "gpt2",
    }
    for (m, b), name in hints.items():
        if mt == m and abs(approx_b - b) < max(0.6, 0.4 * b):
            return name
    if mt == "llama":
        return "meta-llama/Llama-3.2-1B"
    if mt == "qwen2":
        return "Qwen/Qwen2.5-1.5B-Instruct"
    raise ValueError(f"No TransformerLens architecture hint for model_type={mt} ({model_id})")


def load_tl_model(model_id, device="cuda", hf_token=None, dtype=None, arch_hint=None):
    """Load a model into TransformerLens for hook-based access.

    Official TL names load directly. Community fine-tunes (e.g. the medical
    Qwen2.5-1.5B SFT) are loaded via their base architecture + HF weights:
    HookedTransformer.from_pretrained_no_processing(arch, hf_model=...).
    """
    import os
    from transformer_lens import HookedTransformer
    from transformer_lens.loading_from_pretrained import get_official_model_name
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # HF hub reads the token from the environment; avoid forwarding version-
    # sensitive kwargs through TransformerLens.
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    try:
        get_official_model_name(model_id)
        tl = HookedTransformer.from_pretrained(
            model_id, device=device, dtype=dtype)
    except ValueError:
        from transformers import AutoModelForCausalLM
        import transformers as _tf
        arch = arch_hint or _arch_hint_for(model_id, hf_token)
        if int(_tf.__version__.split(".")[0]) >= 5:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, token=hf_token)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)
        tl = HookedTransformer.from_pretrained_no_processing(
            arch, hf_model=hf_model, device=device, dtype=dtype)
    tl.eval()
    return tl


def _find_answer_token_pos(token_ids, letter_ids, parsed_letter):
    """Position of the answer-letter token in the full sequence (scan from end)."""
    targets = set(letter_ids.values())
    for i in range(len(token_ids) - 1, max(len(token_ids) - 80, -1), -1):
        if token_ids[i] in targets:
            return i
    return len(token_ids) - 1


def logit_lens_single(tl_model, prompt, trace, parsed_letter, letters):
    """Run logit lens for one question. Returns per-layer letter probabilities,
    emergence layer, and earliness score."""
    full_text = prompt + trace
    tok = tl_model.tokenizer
    token_ids = tok(full_text, return_tensors="pt")["input_ids"][0].tolist()
    lids = letter_token_ids(tok, letters)
    ans_pos = _find_answer_token_pos(token_ids, lids, parsed_letter)
    probe_pos = ans_pos - 1  # residual here predicts the answer token

    with torch.no_grad():
        _, cache = tl_model.run_with_cache(full_text, remove_batch_dim=True)

    n_layers = tl_model.cfg.n_layers
    per_layer_probs = {}
    for l in range(n_layers):
        r = cache[f"blocks.{l}.hook_resid_post"][probe_pos, :]
        logits_l = tl_model.unembed(r)
        logp = torch.log_softmax(logits_l.float(), dim=-1)
        per_layer_probs[l] = {L: float(torch.exp(logp[tid])) for L, tid in lids.items()}

    chosen = parsed_letter
    emergence = None
    for l in range(n_layers):
        p = per_layer_probs[l]
        if chosen in p and all(p[chosen] >= p[o] for o in p if o != chosen):
            emergence = l
            break
    if emergence is None:
        emergence = n_layers - 1
    return {
        "answer_position": ans_pos,
        "n_layers": n_layers,
        "per_layer_probs": {str(l): p for l, p in per_layer_probs.items()},
        "emergence_layer": emergence,
        "earliness": 1.0 - emergence / n_layers,
    }


def run_logit_lens(tl_model, items, traces, out_path, only_correct=True):
    """items: subset records; traces: Phase-1 trace records (matched by id)."""
    tr = {r["id"]: r for r in traces}
    done = read_jsonl_ids(out_path)
    todo = [it for it in items if it["id"] not in done and it["id"] in tr]
    if only_correct:
        todo = [it for it in todo if tr[it["id"]].get("correct")]
    print(f"[logit_lens] {len(done)} done, {len(todo)} to run (only_correct={only_correct})")
    results = []
    for item in tqdm(todo, desc="logit_lens"):
        t = tr[item["id"]]
        try:
            letters = sorted(item["options"].keys())
            res = logit_lens_single(tl_model, t["prompt"], t["trace"],
                                    t["parsed_answer"] or t["predicted"], letters)
            rec = {"id": item["id"], "gold": item["gold"],
                   "predicted": t.get("predicted"), **res}
        except Exception as e:
            rec = {"id": item["id"], "error": str(e)}
        append_jsonl(rec, out_path)
        results.append(rec)
    return results
