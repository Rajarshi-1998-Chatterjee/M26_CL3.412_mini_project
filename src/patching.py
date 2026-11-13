"""Step 9 - Activation patching (Phase 2b) on the N=100 subset.

Design (teacher-forced, standard for faithfulness patching):
  - clean run:     full text = clean prompt + clean CoT trace
  - corrupted run: full text = corrupted (evidence-removal) prompt + the SAME
                   clean trace continuation, with one (layer, position) spliced
                   from the cached clean run
The target is the logit of the correct option letter at the answer-decision
position (scanned from the end of the sequence).

Recovery score (intra-example ground truth, gold used only as a filter -
only questions where the clean run matched gold are patched):
    score = (logit_patched - logit_corrupted) / (logit_clean - logit_corrupted)
score ~ 1 -> that location alone causally restores the clean answer
score ~ 0 -> causally inert
"""
import torch
from tqdm import tqdm

from .common import append_jsonl, letter_token_ids, read_jsonl_ids


def _answer_probe_pos(token_ids, letter_ids):
    """Position whose residual predicts the answer letter (scan from the end)."""
    targets = set(letter_ids.values())
    for i in range(len(token_ids) - 1, max(len(token_ids) - 80, -1), -1):
        if token_ids[i] in targets:
            return i - 1
    return len(token_ids) - 2


def _gold_logit(tl_model, full_text, lids, gold_letter):
    ids = tl_model.tokenizer(full_text, return_tensors="pt")["input_ids"][0].tolist()
    probe = _answer_probe_pos(ids, lids)
    with torch.no_grad():
        _, cache = tl_model.run_with_cache(full_text, remove_batch_dim=True)
    last = tl_model.cfg.n_layers - 1
    r = cache[f"blocks.{last}.hook_resid_post"][probe, :]
    return float(tl_model.unembed(r).float()[lids[gold_letter]]), probe


def redaction_positions(prompt, tl_model):
    """Token positions of [REDACTED] in the (corrupted) prompt."""
    ids = tl_model.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    red = tl_model.tokenizer("[REDACTED]", add_special_tokens=False)
    pos = []
    for i in range(len(ids) - len(red) + 1):
        if ids[i:i + len(red)] == red:
            pos.extend(range(i, i + len(red)))
    return pos or [len(ids) // 2]


def run_patching_single(tl_model, clean_full_text, corrupted_full_text,
                        gold_letter, letters, positions=None, layers=None):
    tok = tl_model.tokenizer
    lids = letter_token_ids(tok, letters)
    n_layers = tl_model.cfg.n_layers
    layers = layers if layers is not None else list(range(n_layers))

    with torch.no_grad():
        _, clean_cache = tl_model.run_with_cache(clean_full_text, remove_batch_dim=True)
    logit_clean, _ = _gold_logit(tl_model, clean_full_text, lids, gold_letter)
    logit_corr, corr_probe = _gold_logit(tl_model, corrupted_full_text, lids, gold_letter)

    if positions is None:
        positions = redaction_positions(corrupted_full_text, tl_model) + [corr_probe]

    denom = logit_clean - logit_corr
    patches = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        for pos in positions:
            def patch_fn(act, hook, pos=pos):
                act = act.clone()
                act[0, pos, :] = clean_cache[hook_name][pos, :]
                return act
            with torch.no_grad():
                logits = tl_model.run_with_hooks(
                    corrupted_full_text, fwd_hooks=[(hook_name, patch_fn)])
            ids = tok(corrupted_full_text, return_tensors="pt")["input_ids"][0].tolist()
            probe = _answer_probe_pos(ids, lids)
            logit_patched = float(logits[0, probe, lids[gold_letter]])
            score = (logit_patched - logit_corr) / denom if abs(denom) > 1e-6 else None
            patches.append({"layer": layer, "pos": pos, "score": score})
    scored = [p for p in patches if p["score"] is not None]
    best = max(scored, key=lambda p: p["score"]) if scored else None
    return {"logit_clean": logit_clean, "logit_corrupted": logit_corr,
            "n_patches": len(patches), "patches": patches, "best": best}


def run_patching(tl_model, items, perturbations, traces, prompt_cfg,
                 out_path, only_correct=True, max_positions=6):
    """items: subset records; perturbations/traces matched by id."""
    from .inference import build_prompt
    pb = {r["id"]: r for r in perturbations}
    tr = {r["id"]: r for r in traces}
    done = read_jsonl_ids(out_path)
    todo = [it for it in items if it["id"] not in done and it["id"] in pb and it["id"] in tr]
    if only_correct:
        todo = [it for it in todo if tr[it["id"]].get("correct")]
    print(f"[patching] {len(done)} done, {len(todo)} to run (only_correct={only_correct})")
    results = []
    for item in tqdm(todo, desc="patching"):
        t, p = tr[item["id"]], pb[item["id"]]
        try:
            clean_prompt = build_prompt(tl_model.tokenizer, prompt_cfg,
                                        item["question"], item["options"])
            corr_prompt = build_prompt(tl_model.tokenizer, prompt_cfg,
                                       p["evidence_question"], item["options"])
            clean_full = clean_prompt + t["trace"]
            corr_full = corr_prompt + t["trace"]
            letters = sorted(item["options"].keys())
            positions = (redaction_positions(corr_prompt, tl_model))[:max_positions]
            res = run_patching_single(tl_model, clean_full, corr_full,
                                      item["gold"], letters, positions=positions)
            rec = {"id": item["id"], "gold": item["gold"], **res}
        except Exception as e:
            rec = {"id": item["id"], "error": str(e)}
        append_jsonl(rec, out_path)
        results.append(rec)
    return results
