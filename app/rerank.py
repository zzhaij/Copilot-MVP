"""基于交叉编码器的 rerank 模块。"""
from __future__ import annotations

from functools import lru_cache
from math import exp
from typing import Iterable
import re

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from app.config import get_settings
from app.logger import logger


@lru_cache
def _get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    model_name = settings.RERANK_MODEL
    if not model_name:
        raise RuntimeError("RERANK_MODEL 未配置")
    logger.info(f"加载 rerank 模型: {model_name}")
    return CrossEncoder(model_name)


@lru_cache
def _get_qa_components():
    settings = get_settings()
    model_name = settings.QA_MODEL
    if not model_name:
        raise RuntimeError("QA_MODEL 未配置")
    logger.info(f"加载 QA 抽取模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _extract_answer(question: str, context: str, *, settings, tokenizer, model) -> dict:
    if not context:
        return {"score": 0.0, "na_prob": 1.0, "answer": "", "start": None, "end": None, "has_answer": False}

    encoded = tokenizer(
        question,
        context,
        max_length=settings.QA_MAX_LENGTH,
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop("offset_mapping")[0]
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)

    seq_len = start_probs.size(0)
    max_len = min(settings.QA_MAX_ANSWER_LEN, seq_len)
    best_score = 0.0
    best_start = 0
    best_end = 0

    for i in range(1, seq_len):
        max_j = min(i + max_len, seq_len)
        for j in range(i, max_j):
            score = float((start_probs[i] * end_probs[j]).item())
            if score > best_score:
                best_score = score
                best_start = i
                best_end = j

    na_prob = float((start_probs[0] * end_probs[0]).item())

    if best_score <= na_prob or best_start == 0:
        return {"score": best_score, "na_prob": na_prob, "answer": "", "start": None, "end": None, "has_answer": False}

    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[best_start:best_end + 1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    start_char = end_char = None
    if best_end < len(offset_mapping):
        start_char = int(offset_mapping[best_start][0])
        end_char = int(offset_mapping[best_end][1])

    has_answer = bool(answer_text) and best_score >= settings.QA_SCORE_THRESHOLD

    return {
        "score": best_score,
        "na_prob": na_prob,
        "answer": answer_text,
        "start": start_char,
        "end": end_char,
        "has_answer": has_answer,
    }


_STOP_WORDS = {"是", "的", "了", "吗", "呢", "吧", "有", "什么", "哪些", "怎么", "如何", "谁", "哪", "多少", "请问", "请"}


def _question_keywords(question: str) -> list[str]:
    coarse = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{2,}|\d+", question)
    keywords: list[str] = []
    for token in coarse:
        if token in _STOP_WORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            length = len(token)
            if length >= 2:
                if token not in keywords:
                    keywords.append(token)
                for n in range(2, min(4, length) + 1):
                    for i in range(0, length - n + 1):
                        sub = token[i:i + n]
                        if sub not in _STOP_WORDS and sub not in keywords:
                            keywords.append(sub)
            else:
                keywords.append(token)
        else:
            keywords.append(token)
    return keywords


def _keyword_hit_ratio(text: str, keywords: list[str]) -> float:
    if not text or not keywords:
        return 0.0
    matched = 0
    for kw in keywords:
        if kw in text:
            matched += 1
    return matched / len(keywords)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def rerank_passages(question: str, hits: Iterable[dict]) -> list[dict]:
    """对候选片段进行 rerank，并返回新的降序列表。"""
    hits = list(hits)
    if len(hits) <= 1:
        return hits

    settings = get_settings()
    if not settings.RERANK_MODEL:
        return hits

    model = _get_cross_encoder()
    tokenizer, qa_model = _get_qa_components()
    keywords = _question_keywords(question)
    pairs = [(question, h.get("text", "")) for h in hits]
    scores = model.predict(pairs)

    enriched: list[dict] = []
    for hit, score in zip(hits, scores):
        new_hit = {**hit}
        rel_score = float(score)
        rel_prob = _sigmoid(rel_score)
        qa_result = _extract_answer(
            question,
            hit.get("text", ""),
            settings=settings,
            tokenizer=tokenizer,
            model=qa_model,
        )
        qa_score = float(qa_result.get("score", 0.0))
        answer_text = qa_result.get("answer", "")
        has_answer = bool(qa_result.get("has_answer"))
        kw_ratio = _keyword_hit_ratio(hit.get("text", ""), keywords)

        combined = (
            rel_prob * settings.RERANK_REL_WEIGHT
            + qa_score * settings.RERANK_QA_WEIGHT
            + (settings.RERANK_HAS_ANSWER_BONUS if has_answer else 0.0)
        )

        if keywords and not has_answer:
            penalty = (1 - kw_ratio) * settings.RERANK_KEYWORD_PENALTY
            combined = max(0.0, combined - penalty)
        else:
            penalty = 0.0

        new_hit["rerank_score"] = round(rel_score, 4)
        new_hit["rel_prob"] = round(rel_prob, 4)
        new_hit["qa_score"] = round(qa_score, 4)
        new_hit["qa_na_prob"] = round(float(qa_result.get("na_prob", 0.0)), 4)
        new_hit["qa_answer"] = answer_text
        new_hit["qa_start"] = qa_result.get("start")
        new_hit["qa_end"] = qa_result.get("end")
        new_hit["has_answer"] = has_answer
        new_hit["keyword_ratio"] = round(kw_ratio, 4)
        new_hit["keyword_penalty"] = round(penalty, 4)
        new_hit["final_score"] = round(combined, 4)
        enriched.append(new_hit)

    enriched.sort(key=lambda h: h["final_score"], reverse=True)

    for idx, hit in enumerate(enriched, start=1):
        hit["rank"] = idx
        hit["score"] = hit["final_score"]
    return enriched
