"""Step 3 - Few-shot CoT generation and answer-probability scoring.

Key functions:
- build_prompt: renders the shared few-shot chat template (configs/prompt.yaml)
- generate_trace: greedy CoT generation, returns text + per-step logits
- score_generation: given generated token ids + step logits, compute the
  probability of every option letter at the answer-decision step, the top-1
  predicted option, and the probability of a reference (original) answer.
- run_generation / score_perturbed: resumable loops with JSONL checkpointing.
"""
import os
import random

import torch
from tqdm import tqdm

from .common import (append_jsonl, letter_token_ids, load_prompt_config,
                     parse_answer_letter, read_jsonl_ids)


# ---------- Prompt construction ----------

def format_options(options):
    return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))


def build_messages(pcfg, question, options):
    """Chat messages: system + few-shot pairs + the target question."""
    msgs = [{"role": "system", "content": pcfg["system_prompt"]}]
    for ex in pcfg.get("few_shot", []):
        user = pcfg["user_template"].format(
            question=ex["question"],
            options=format_options(ex["options"]),
        )
        # Strip the trailing "Let's think step by step." for the assistant turn
        user = user.replace("Answer: Let's think step by step.", "Answer:").strip()
        assistant = f"{ex['reasoning'].strip()}\nAnswer: {ex['answer']}"
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    msgs.append({
        "role": "user",
        "content": pcfg["user_template"].format(
            question=question, options=format_options(options)
        ),
    })
    return msgs


def build_prompt(tokenizer, pcfg, question, options):
    msgs = build_messages(pcfg, question, options)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    # Fallback: plain-text rendering
    parts = [pcfg["system_prompt"], ""]
    for m in msgs[1:]:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)


# ---------- Generation ----------

def generate_with_scores(model, tokenizer, prompt, max_new_tokens=512):
    """Greedy generation. Returns (gen_text, gen_token_ids, step_logits, input_len)."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True, pad_token_id=pad_id,
        )
    gen_ids = out.sequences[0, input_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    step_logits = [s[0].float().cpu() for s in out.scores]  # list of [vocab]
    return gen_text, gen_ids, step_logits, input_len


def _answer_step_index(gen_ids, letter_ids, parsed_letter):
    """Index into gen_ids of the token that emits the answer letter."""
    targets = {v for k, v in letter_ids.items() if k == parsed_letter} or set(letter_ids.values())
    for i in range(len(gen_ids) - 1, -1, -1):
        if gen_ids[i] in targets:
            return i
    return None


def score_generation(gen_ids, step_logits, letter_ids, reference_letter=None):
    """At the answer-decision step, compute per-letter probabilities.

    Returns dict with predicted (top-1 letter), probs per letter,
    P_reference (prob of reference_letter, e.g. the clean-run answer),
    and the step index used. Falls back to the final step if no answer
    token is found.
    """
    if not step_logits:
        return None
    parsed = None
    step = _answer_step_index(gen_ids, letter_ids, parsed)
    if step is None:
        step = len(step_logits) - 1
    logits = step_logits[step]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = {L: float(torch.exp(log_probs[tid])) for L, tid in letter_ids.items()}
    predicted = max(probs, key=probs.get)
    return {
        "predicted": predicted,
        "probs": probs,
        "P_reference": float(probs[reference_letter]) if reference_letter in probs else None,
        "answer_step": step,
    }


def generate_and_score(model, tokenizer, pcfg, item, max_new_tokens=512,
                       reference_letter=None):
    """Full clean run for one question: build prompt, generate, parse, score."""
    letters = sorted(item["options"].keys())
    lids = letter_token_ids(tokenizer, letters)
    prompt = build_prompt(tokenizer, pcfg, item["question"], item["options"])
    gen_text, gen_ids, step_logits, input_len = generate_with_scores(
        model, tokenizer, prompt, max_new_tokens)
    parsed = parse_answer_letter(gen_text, letters)
    scored = score_generation(gen_ids, step_logits, lids, reference_letter=reference_letter)
    predicted = parsed or (scored["predicted"] if scored else None)
    p_clean = scored["probs"].get(predicted) if scored else None
    return {
        "id": item["id"],
        "prompt": prompt,
        "trace": gen_text,
        "parsed_answer": parsed,
        "predicted": predicted,
        "P_clean": p_clean,
        "probs": scored["probs"] if scored else {},
        "gold": item["gold"],
        "correct": (predicted == item["gold"]) if predicted else False,
        "parse_ok": parsed is not None,
    }


# ---------- Resumable loops ----------

def run_generation(model, tokenizer, pcfg, items, out_path, max_new_tokens=512,
                   checkpoint_every=10):
    """Step 3 loop over all items with checkpointing + resume."""
    done = read_jsonl_ids(out_path)
    results = []
    todo = [it for it in items if it["id"] not in done]
    print(f"[inference] {len(done)} already done, {len(todo)} to run")
    for i, item in enumerate(tqdm(todo, desc="generate")):
        try:
            rec = generate_and_score(model, tokenizer, pcfg, item, max_new_tokens)
        except Exception as e:
            rec = {"id": item["id"], "error": str(e), "gold": item["gold"]}
        append_jsonl(rec, out_path)
        results.append(rec)
        if (i + 1) % checkpoint_every == 0:
            pass  # append_jsonl already persists each record
    return results


def score_perturbed_prompt(model, tokenizer, pcfg, item, perturbed_question,
                           original_letter, max_new_tokens=512):
    """Step 4 helper: re-run answer selection on a perturbed vignette.

    Returns predicted option, P(original answer), flip indicator, per-letter probs.
    """
    letters = sorted(item["options"].keys())
    lids = letter_token_ids(tokenizer, letters)
    p_item = dict(item, question=perturbed_question)
    prompt = build_prompt(tokenizer, pcfg, p_item["question"], p_item["options"])
    gen_text, gen_ids, step_logits, _ = generate_with_scores(
        model, tokenizer, prompt, max_new_tokens)
    parsed = parse_answer_letter(gen_text, letters)
    scored = score_generation(gen_ids, step_logits, lids, reference_letter=original_letter)
    predicted = parsed or (scored["predicted"] if scored else None)
    p_orig = scored["P_reference"] if scored else None
    return {
        "predicted": predicted,
        "P_original": p_orig,
        "flip": (predicted is not None and predicted != original_letter),
        "probs": scored["probs"] if scored else {},
        "trace": gen_text,
    }
