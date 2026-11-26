#!/usr/bin/env python3
"""Judge helpers: extract 最终答案 + Chinese-aware F1 / keyword-recall scoring."""

from __future__ import annotations

import re
import string

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ANSWER_TAGS = ("最终答案：", "最终答案:", "答案：", "答案:")
# Emitted once ignore_eos + skip_special_tokens=false let the model run past its
# own stop token; everything from here on is padding, not an answer.
# ===ANSWER_END=== is emitted by the model itself (chat.PAD_DELIM) when a prompt
# instruction asks it to mark the end of the answer and keep writing filler, so
# fixed-length replies work on gateways that drop ignore_eos.
EOS_MARKS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "===ANSWER_END===")


def cut_at_eos(text: str) -> str:
    """Keep only what the model produced before its first natural stop."""
    body = text or ""
    cuts = [body.find(m) for m in EOS_MARKS]
    cuts = [c for c in cuts if c >= 0]
    return body[:min(cuts)] if cuts else body


def strip_think(text: str) -> str:
    return THINK_RE.sub("", cut_at_eos(text))


def extract_final_answer(text: str) -> str:
    """Parse the first tagged answer line; fallback to last non-empty line."""
    body = strip_think(text).strip()
    if not body:
        return ""
    for line in body.splitlines():
        stripped = line.strip()
        for tag in ANSWER_TAGS:
            if stripped.startswith(tag):
                rest = stripped[len(tag):].strip()
                if rest:
                    return rest
    for tag in ANSWER_TAGS:
        if tag in body:
            tail = body.split(tag, 1)[-1]
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line
            return tail.strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else body


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def tokenize(text: str) -> list[str]:
    """Mixed CJK char + latin word tokenizer (better F1 for Chinese answers)."""
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) and ch not in "，。！？；：、""''（）【】《》…—·")
    toks: list[str] = []
    buf = []
    for ch in text:
        if _is_cjk(ch):
            if buf:
                toks.append("".join(buf))
                buf = []
            if not ch.isspace():
                toks.append(ch)
        elif ch.isspace():
            if buf:
                toks.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        toks.append("".join(buf))
    return [t for t in toks if t]


def f1(pred: str, gold: str) -> float:
    pt, gt = tokenize(pred), tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common: dict[str, int] = {}
    for t in pt:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gt:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, gold_ans: str | None) -> float:
    if not gold_ans:
        return 1.0
    g = tokenize(str(gold_ans))
    if not g:
        return 1.0
    pset = set(tokenize(pred))
    return sum(1 for t in g if t in pset) / len(g)


def score_one(
    pred_text: str,
    answers: list | None,
    gold_ans: str | None = None,
    recall_threshold: float = 0.2,
    extract_answer: bool = True,
) -> dict:
    """Return score dict. Default recall_threshold=0.2 matches LV-Eval Chinese."""
    extracted = extract_final_answer(pred_text) if extract_answer else (pred_text or "")
    gate = keyword_recall(extracted, gold_ans)
    if gate < recall_threshold:
        return {
            "score": 0.0,
            "extracted_answer": extracted,
            "keyword_recall": gate,
            "gated": True,
        }
    refs = list(answers or [])
    if gold_ans and gold_ans not in refs:
        refs.append(gold_ans)
    if not refs:
        s = 0.0
    else:
        s = max(f1(extracted, str(a)) for a in refs)
    return {
        "score": s,
        "extracted_answer": extracted,
        "keyword_recall": gate,
        "gated": False,
    }
