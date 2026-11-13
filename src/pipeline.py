"""Phase orchestrators used by the Colab notebooks and the CLI.

run_phase1(cfg)            -> Steps 2-5: sample, generate, perturb, score, stats, subset
run_phase2_logit_lens(cfg) -> Step 7
run_phase2_patching(cfg)   -> Step 9
run_agreement(cfg)         -> Step 10 (+ review-case selection for Phase 3)
run_report(cfg)            -> Step 12
"""
import os
import random
from pathlib import Path

from .common import (ROOT, append_jsonl, ensure_dir, load_config, load_json,
                     load_prompt_config, read_jsonl, read_jsonl_ids, read_records,
                     save_json, set_seed)
from . import data_loading, faithfulness, inference, perturbations


def _paths(cfg):
    data_dir = ensure_dir(ROOT / cfg["paths"]["data_dir"])
    out_dir = ensure_dir(ROOT / cfg["paths"]["outputs_dir"])
    return data_dir, out_dir


def _hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return t if t else None


# ---------------- Phase 1 ----------------

def run_phase1(cfg, model_id=None, smoke=False):
    set_seed(cfg.get("seed", 42))
    data_dir, out_dir = _paths(cfg)
    pcfg = load_prompt_config()
    gcfg = cfg["generation"]
    dcfg = cfg["dataset"]

    n = 20 if smoke else dcfg["n_sample"]
    sample_path = data_dir / f"sample_{n}.jsonl"
    if sample_path.exists():
        items = read_jsonl(sample_path)
    else:
        items = data_loading.load_dataset_records({**cfg, "dataset": {**dcfg, "n_sample": n}})
        data_loading.save_sample(items, sample_path)
    print(f"[phase1] {len(items)} questions from {sample_path}")

    model_id = model_id or cfg["model"]["primary"]
    print(f"[phase1] loading model {model_id}")
    from .common import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(model_id, dtype=cfg["model"].get("dtype", "bfloat16"),
                                          hf_token=_hf_token())

    # Step 3: CoT generation (resumable)
    traces_path = out_dir / f"traces_{n}.jsonl"
    inference.run_generation(model, tok, pcfg, items, traces_path,
                             max_new_tokens=gcfg["max_new_tokens"],
                             checkpoint_every=gcfg.get("checkpoint_every", 10))
    traces = read_jsonl(traces_path)
    print(f"[phase1] {len(traces)} traces")

    # Step 4: perturbations (deterministic per-question rng)
    pert_path = out_dir / f"perturbations_{n}.jsonl"
    done = read_jsonl_ids(pert_path)
    for it in items:
        if it["id"] in done:
            continue
        tr = next((t for t in traces if t["id"] == it["id"]), None)
        if tr is None or not tr.get("trace"):
            continue
        rng = random.Random(hash((cfg.get("seed", 42), it["id"])) & 0xffffffff)
        rec = perturbations.build_perturbations(it, tr["trace"], rng=rng)
        append_jsonl(rec, pert_path)
    perturbations_all = read_jsonl(pert_path)
    print(f"[phase1] {len(perturbations_all)} perturbations")

    # Step 4b: score perturbed prompts (evidence + control), resumable
    ev_path = out_dir / f"ev_scores_{n}.jsonl"
    ct_path = out_dir / f"ctrl_scores_{n}.jsonl"
    tr_by_id = {t["id"]: t for t in traces}
    pb_by_id = {p["id"]: p for p in perturbations_all}
    for path, key in ((ev_path, "evidence_question"), (ct_path, "control_question")):
        done = read_jsonl_ids(path)
        for it in items:
            if it["id"] in done or it["id"] not in tr_by_id or it["id"] not in pb_by_id:
                continue
            t, p = tr_by_id[it["id"]], pb_by_id[it["id"]]
            if not t.get("predicted"):
                continue
            try:
                res = inference.score_perturbed_prompt(
                    model, tok, pcfg, it, p[key], t["predicted"],
                    max_new_tokens=gcfg["max_new_tokens"])
                res["id"] = it["id"]
            except Exception as e:
                res = {"id": it["id"], "error": str(e)}
            append_jsonl(res, path)

    # Step 5: join + stats
    ev = read_jsonl(ev_path)
    ct = read_jsonl(ct_path)
    scored = faithfulness.compute_faithfulness(traces, perturbations_all, ev, ct)
    stats_res = faithfulness.run_phase1_stats(scored, out_dir)
    save_json(scored, out_dir / f"scored_{n}.json")
    print("[phase1] stats:", stats_res)

    # Step 2b: mechanistic subset (skip in smoke mode)
    if not smoke and n == dcfg["n_sample"]:
        subset_ids = data_loading.select_mechanistic_subset(
            scored, items, data_dir,
            n_subset=dcfg["n_subset"],
            n_low_f=dcfg.get("low_faithfulness_oversample", 70),
            seed=cfg.get("seed", 42))
        print(f"[phase1] subset of {len(subset_ids)} -> data/subset_100_ids.json")
    return stats_res


# ---------------- Phase 2a ----------------

def run_phase2_logit_lens(cfg, model_id=None):
    data_dir, out_dir = _paths(cfg)
    subset = read_jsonl(data_dir / "subset_100.jsonl")
    traces = read_jsonl(out_dir / f"traces_{cfg['dataset']['n_sample']}.jsonl")
    from .logit_lens import load_tl_model, run_logit_lens
    tl = load_tl_model(model_id or cfg["model"]["primary"], hf_token=_hf_token())
    return run_logit_lens(tl, subset, traces, out_dir / "logit_lens_100.jsonl")


# ---------------- Phase 2b ----------------

def run_phase2_patching(cfg, model_id=None):
    data_dir, out_dir = _paths(cfg)
    subset = read_jsonl(data_dir / "subset_100.jsonl")
    traces = read_jsonl(out_dir / f"traces_{cfg['dataset']['n_sample']}.jsonl")
    perts = read_jsonl(out_dir / f"perturbations_{cfg['dataset']['n_sample']}.jsonl")
    pcfg = load_prompt_config()
    from .logit_lens import load_tl_model
    from .patching import run_patching
    tl = load_tl_model(model_id or cfg["model"]["primary"], hf_token=_hf_token())
    return run_patching(tl, subset, perts, traces, pcfg, out_dir / "patching_100.jsonl")


# ---------------- Agreement + review cases ----------------

def run_agreement(cfg):
    _, out_dir = _paths(cfg)
    n = cfg["dataset"]["n_sample"]
    scored = read_records(out_dir / f"scored_{n}.json")
    ll = read_jsonl(out_dir / "logit_lens_100.jsonl")
    pt = read_jsonl(out_dir / "patching_100.jsonl")
    from .agreement import run_agreement as _ra
    stats_res, rows = _ra(scored, ll, pt, out_dir)
    print("[agreement]", {k: {kk: vv for kk, vv in v.items() if kk != 'off_diagonal_ids'}
                           for k, v in stats_res.items()})
    # select the N=50 review cases for Phase 3
    from .taxonomy import select_review_cases
    off_path = out_dir / "off_diagonal_ids.json"
    off = load_json(off_path) if off_path.exists() else []
    case_ids = select_review_cases(scored, off, n_total=50,
                                   traces=read_jsonl(out_dir / f"traces_{n}.jsonl"))
    save_json(case_ids, out_dir / "review_case_ids.json")
    print(f"[agreement] {len(case_ids)} review cases -> outputs/review_case_ids.json")
    return stats_res, case_ids


# ---------------- Report ----------------

def run_report(cfg):
    from .report import run_report as _rr
    _, out_dir = _paths(cfg)
    return _rr(out_dir, out_dir / "figures")


if __name__ == "__main__":
    cfg = load_config()
    run_phase1(cfg)
