"""Step 2 - Dataset loading and sampling.

Primary: MedQA (USMLE, 4-option, bigbio/med_qa parquet files on HF - ungated,
no dataset script required). Fallbacks: GBaker/MedQA-USMLE-4-options (parquet-style
HF dataset), or a local JSONL file with fields {id, question, options, gold}.

Sampling: fixed-seed sample of N=400 from the MedQA test split; the Phase-2
subset of N=100 is drawn later from Phase-1 results (see pipeline.select_subset).
"""
import random
from pathlib import Path

from .common import ROOT, append_jsonl, ensure_dir, read_jsonl, save_json

BIGBIO_REPO = "bigbio/med_qa"
BIGBIO_TEST_PARQUET = "med_qa_en_4options_source/test-00000-of-00001.parquet"


def _normalize_options(options):
    """bigbio stores options as a list/array of {key, value}; some mirrors use
    dicts {A: text} or lists of {label/text, answer}."""
    if isinstance(options, dict):
        return {str(k).upper(): str(v) for k, v in options.items()}
    out = {}
    for i, opt in enumerate(options):
        if isinstance(opt, dict):
            label = str(opt.get("label", opt.get("key", chr(ord("A") + i)))).upper()
            text = opt.get("text", opt.get("answer", opt.get("value", "")))
            out[label] = str(text)
        else:
            out[chr(ord("A") + i)] = str(opt)
    return out


def _normalize_answer(answer, options):
    """Normalize a gold-answer field (letter, index, or free text) to an option letter."""
    import re
    if answer is None:
        return None
    # bigbio_qa config stores answer as list of dicts with label/text
    if isinstance(answer, list) and answer:
        lab = str(answer[0].get("label", "")).upper()
        if lab in options:
            return lab
        answer = answer[0].get("text", answer[0].get("value", ""))
    a = str(answer).strip()
    au = a.upper()
    if au in options:
        return au
    if au.isdigit():  # 0-based or 1-based index
        idx = int(au)
        keys = sorted(options)
        if 0 <= idx < len(keys):
            return keys[idx]
        if 1 <= idx <= len(keys):
            return keys[idx - 1]
    # free-text answer (e.g. bigbio's "answer" column): match against option text
    a_norm = a.strip().lower()
    for k, v in options.items():
        if str(v).strip().lower() == a_norm:
            return k
    # last resort: a short bare-letter answer like "(A)" or "A." — never scan long
    # free text for this, since e.g. "Tell the attending..." contains stray A-E letters
    if len(au) <= 3:
        m = re.search(r"([A-E])", au)
        if m and m.group(1) in options:
            return m.group(1)
    return None


def _normalize_record(raw, idx):
    options = _normalize_options(raw.get("options", {}))
    # Prefer the dataset's explicit answer-letter column (e.g. bigbio's
    # `answer_idx`) over free-text `answer`, which needs text-matching against
    # the (now-correct) option text and is otherwise unreliable.
    gold = _normalize_answer(raw.get("answer_idx", raw.get("gold")), options)
    if gold is None:
        gold = _normalize_answer(raw.get("answer"), options)
    qid = raw.get("id", raw.get("question_id", f"medqa-{idx}"))
    return {
        "id": str(qid),
        "question": str(raw.get("question", "")),
        "options": options,
        "gold": gold,
        "meta_info": raw.get("meta_info", ""),
    }


def load_medqa_local(path):
    """Load a local JSONL with {id, question, options, gold}."""
    recs = read_jsonl(path)
    return [_normalize_record(r, i) for i, r in enumerate(recs)]


def load_medqa(n=None, seed=42, split="test"):
    """Load MedQA USMLE 4-option test split (canonical bigbio parquet, ungated).

    Returns list of normalized records. Downloads via huggingface_hub (no
    dataset-script dependency, works with modern `datasets` versions).
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    local = hf_hub_download(repo_id=BIGBIO_REPO, repo_type="dataset",
                            filename=BIGBIO_TEST_PARQUET)
    df = pd.read_parquet(local)
    records = [_normalize_record(row, i) for i, row in enumerate(df.to_dict("records"))]
    records = [r for r in records if r["gold"] is not None and len(r["options"]) >= 2]
    return sample_records(records, n=n, seed=seed)


def load_gbaker_medqa(n=None, seed=42):
    """Fallback: GBaker/MedQA-USMLE-4-options (plain HF dataset, parquet-backed)."""
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    records = [_normalize_record(r, i) for i, r in enumerate(ds)]
    records = [r for r in records if r["gold"] is not None]
    return sample_records(records, n=n, seed=seed)


def load_dataset_records(cfg):
    """Dispatch on cfg['dataset']['name']: 'medqa' | 'gbaker' | 'local'."""
    dcfg = cfg["dataset"]
    name = dcfg.get("name", "medqa")
    n = dcfg.get("n_sample")
    seed = cfg.get("seed", 42)
    if name == "local" or dcfg.get("path"):
        return load_medqa_local(dcfg["path"])
    if name == "gbaker":
        return load_gbaker_medqa(n=n, seed=seed)
    try:
        return load_medqa(n=n, seed=seed)
    except Exception as e:
        print(f"[data_loading] bigbio load failed ({e}); trying GBaker mirror")
        return load_gbaker_medqa(n=n, seed=seed)


def sample_records(records, n=None, seed=42):
    rng = random.Random(seed)
    if n is None or n >= len(records):
        return records
    return rng.sample(records, n)


def save_sample(records, path):
    path = ensure_dir(Path(path).parent) / Path(path).name
    path.write_text("")  # truncate
    for r in records:
        append_jsonl(r, path)
    return path


def load_sample(path):
    return read_jsonl(path)


def select_mechanistic_subset(scored_records, all_records, out_dir,
                              n_subset=100, n_low_f=70, seed=42):
    """Step 2b: draw the N=100 mechanistic subset from Phase-1 scored records,
    oversampling low-faithfulness cases (lowest F(x)), rest random.
    Writes subset_100_ids.json and subset_100.jsonl into out_dir."""
    out_dir = ensure_dir(out_dir)
    by_id = {r["id"]: r for r in all_records}
    scored = [r for r in scored_records if r["id"] in by_id]
    scored_sorted = sorted(scored, key=lambda r: r.get("F", 0.0))
    low = scored_sorted[:n_low_f]
    rest = scored_sorted[n_low_f:]
    rng = random.Random(seed)
    fill = rng.sample(rest, min(n_subset - len(low), len(rest)))
    chosen = low + fill
    rng.shuffle(chosen)
    subset_ids = [r["id"] for r in chosen]
    save_json(subset_ids, out_dir / "subset_100_ids.json")
    path = out_dir / "subset_100.jsonl"
    path.write_text("")
    for r in chosen:
        append_jsonl(by_id[r["id"]], path)
    return subset_ids


if __name__ == "__main__":
    from .common import load_config
    cfg = load_config()
    recs = load_dataset_records(cfg)
    out = ensure_dir(ROOT / cfg["paths"]["data_dir"]) / "sample_400.jsonl"
    save_sample(recs, out)
    print(f"Saved {len(recs)} records to {out}")
    print("Example:", {k: (str(v)[:80]) for k, v in recs[0].items()})
