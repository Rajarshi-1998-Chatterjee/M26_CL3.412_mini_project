#!/usr/bin/env python3
"""Step 14 - Lightweight, reusable CLI for trace-perturbation faithfulness
evaluation on MedQA-style datasets. No mechanistic dependencies.

Examples:
  # Full run on MedQA with the primary medical model (GPU recommended)
  python eval_faithfulness.py --model Tianye88/Qwen2.5-1.5B-Instruct-Medical-cpt-sft-v1 \
      --dataset medqa --n 400 --out outputs/run1/

  # Quick smoke test (20 questions)
  python eval_faithfulness.py --smoke-test --out outputs/smoke/

  # Local MedQA-style JSONL ({id, question, options, gold})
  python eval_faithfulness.py --dataset local --dataset-path my_data.jsonl --n 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.common import load_config  # noqa: E402
from src import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="HF model id (default: config primary)")
    ap.add_argument("--dataset", default=None, choices=["medqa", "gbaker", "local"],
                    help="dataset to use (default: config)")
    ap.add_argument("--dataset-path", default=None, help="local JSONL path (dataset=local)")
    ap.add_argument("--n", type=int, default=None, help="sample size (default: config 400)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="output directory (default: config outputs_dir)")
    ap.add_argument("--smoke-test", action="store_true", help="run with N=20")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hf-token", default=None, help="HF token for gated models")
    args = ap.parse_args()

    cfg = load_config()
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.dataset_path:
        cfg["dataset"]["name"] = "local"
        cfg["dataset"]["path"] = args.dataset_path
    if args.n:
        cfg["dataset"]["n_sample"] = args.n
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out:
        cfg["paths"]["outputs_dir"] = args.out
    if args.max_new_tokens:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.hf_token:
        import os
        os.environ["HF_TOKEN"] = args.hf_token

    stats = pipeline.run_phase1(cfg, model_id=args.model, smoke=args.smoke_test)
    print("\n=== Phase 1 results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
