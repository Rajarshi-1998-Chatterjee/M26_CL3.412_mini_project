"""Step 4 - Counterfactual perturbations of the vignette.

For each question we build TWO modified versions of the question text
(the option list is never changed):
  - evidence-removal: the span the CoT trace cites as decisive is redacted
  - control-removal:  a different, uncited span of comparable length is redacted

The control distinguishes "sensitive to losing the cited evidence" from
"sensitive to losing any information".
"""
import difflib
import json
import random
import re

REDACTED = "[REDACTED]"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _normalize(s):
    # lowercase, strip punctuation (keeps hyphens/word chars), collapse space
    s = re.sub(r"[^\w\s\-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _tokenize(s):
    """Return (normalized_words, original_words) aligned 1:1; punctuation dropped."""
    norm, orig = [], []
    for m in re.finditer(r"[\w\-]+|[^\w\s\-]+|\s+", s):
        t = m.group(0)
        if t.isspace() or not re.fullmatch(r"[\w\-]+", t):
            continue
        norm.append(t.lower())
        orig.append(t)
    return norm, orig


def _extract_cited_candidates(trace):
    """Candidate 'decisive' spans: quoted spans, causal phrases, and full
    trace sentences (matched against the question later)."""
    cands = []
    for m in re.finditer(r'"([^"]{8,120})"', trace):  # quoted spans
        cands.append(m.group(1))
    for m in re.finditer(
            r"\b(?:because|since|given|as|due to|key finding is|shows?|reveals?)\s+(.{8,160}?)[\.\n]",
            trace, re.IGNORECASE):
        cands.append(m.group(1))
    cands.extend(split_sentences(trace))
    return [c for c in cands if 8 <= len(c) <= 300]


def _match_in_question(candidate, question, min_words=3, min_ratio=0.4):
    """Word-level fuzzy match of a trace span into the question.

    Returns the matched question substring (original casing), or None.
    """
    q_words, q_orig = _tokenize(question)
    c_words, _ = _tokenize(candidate)
    if not q_words or not c_words:
        return None
    sm = difflib.SequenceMatcher(None, q_words, c_words, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= min_words]
    if not blocks:
        return None
    best = max(blocks, key=lambda b: b.size)
    ratio = sm.ratio()
    # Accept a solid verbatim phrase (>=4 words), or a shorter one only when
    # the overall candidate is mostly contained in the question.
    if best.size < 4 and not (best.size >= min_words and ratio >= 0.5):
        return None
    start, n = best.a, best.size
    if start + n <= len(q_orig):
        return " ".join(q_orig[start:start + n])
    return None


def find_cited_span(question, trace, llm_extract=None):
    """Find the question span the trace cites as decisive.

    Strategy: (1) quoted/because-phrases in the trace fuzzy-matched into the
    question; (2) optional LLM extraction (pass llm_extract(question, trace)
    -> span string); (3) None -> flagged for manual review.
    Returns (span, method) or (None, 'unresolved').
    """
    for cand in _extract_cited_candidates(trace):
        span = _match_in_question(cand, question)
        if span and _normalize(span) in _normalize(question):
            return span, "trace_match"
    if llm_extract is not None:
        try:
            span = llm_extract(question, trace)
            if span:
                span = str(span).strip().strip('"')
                if _normalize(span) in _normalize(question):
                    return span, "llm_match"
                m = _match_in_question(span, question)
                if m:
                    return m, "llm_match"
        except Exception:
            pass
    return None, "unresolved"


def make_evidence_removal(question, cited_span):
    """Replace the cited span with [REDACTED] (aligned word-level match)."""
    if not cited_span:
        return question
    q_words, q_orig = _tokenize(question)
    s_words, _ = _tokenize(cited_span)
    if not s_words:
        return question
    for i in range(len(q_words) - len(s_words) + 1):
        if q_words[i:i + len(s_words)] == s_words:
            orig_span = " ".join(q_orig[i:i + len(s_words)])
            return question.replace(orig_span, REDACTED, 1)
    return question  # no-op if span not found (flagged upstream)


def make_control_removal(question, cited_span, rng=None):
    """Redact a random UNCITED sentence/span of comparable length."""
    rng = rng or random.Random()
    sents = split_sentences(question)
    cited_norm = _normalize(cited_span or "")
    target_len = len(cited_span or "") or 40
    candidates = []
    for s in sents:
        if cited_norm and _normalize(s) in cited_norm or cited_norm in _normalize(s):
            continue  # skip the cited sentence
        if 0.4 * target_len <= len(s) <= 2.5 * target_len + 20:
            candidates.append(s)
    if not candidates:  # relax constraints
        candidates = [s for s in sents if cited_norm not in _normalize(s)] or sents
    if not candidates:
        return question, None
    span = rng.choice(candidates)
    return question.replace(span, REDACTED, 1), span


def build_perturbations(item, trace, rng=None, llm_extract=None, manual_spans=None):
    """Build evidence- and control-removal vignettes for one question.

    manual_spans: optional dict id -> span, for human-resolved cases.
    Returns dict with both perturbed questions, the spans used, and flags.
    """
    rng = rng or random.Random()
    q = item["question"]
    manual = (manual_spans or {}).get(item["id"])
    if manual:
        cited, method = manual, "manual"
    else:
        cited, method = find_cited_span(q, trace, llm_extract=llm_extract)
    if cited is None:
        # fallback: redact a random sentence as 'evidence' and flag it
        sents = split_sentences(q)
        cited = rng.choice(sents) if sents else q[:50]
        method = "fallback_random"
    ev_q = make_evidence_removal(q, cited)
    ctrl_q, ctrl_span = make_control_removal(q, cited, rng)
    return {
        "id": item["id"],
        "cited_span": cited,
        "cited_method": method,
        "control_span": ctrl_span,
        "evidence_question": ev_q,
        "control_question": ctrl_q,
        "options_unchanged": True,
        "len_ratio_ev": len(ev_q) / max(len(q), 1),
        "len_ratio_ctrl": len(ctrl_q) / max(len(q), 1),
    }


def llm_span_extractor(model, tokenizer, prompt_cfg):
    """Optional LLM-based decisive-span extractor (used only if regex fails).
    Returns a callable(question, trace) -> span string."""
    def extract(question, trace):
        from .inference import generate_with_scores
        prompt = (
            "Given the clinical vignette and the model's reasoning trace, output "
            "ONLY the exact verbatim span of the vignette that the trace relies on "
            "as decisive evidence, as JSON: {\"span\": \"...\"}.\n\n"
            f"Vignette:\n{question}\n\nTrace:\n{trace[-2000:]}\n\nJSON:"
        )
        text, _, _, _ = generate_with_scores(model, tokenizer, prompt, max_new_tokens=128)
        m = re.search(r'\{[^{}]*"span"[^{}]*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0)).get("span", "")
        return ""
    return extract
