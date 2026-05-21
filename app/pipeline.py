"""高层链路：把 retrieve + prompt + llm 串起来，并落 trace。"""
from __future__ import annotations

import json

from app.llm import chat_complete
from app.logger import logger
from app.prompts import build_qa_messages, build_summary_messages
from app.retriever import retrieve
from app.tracing import Trace


def answer_question(question: str, *, top_k: int | None = None, trace: Trace | None = None) -> dict:
    hits = retrieve(question, top_k=top_k)
    if trace:
        trace.log("retrieved", count=len(hits),
                  preview=[{"rank": h["rank"], "doc": h["doc_name"], "score": h["score"]} for h in hits])

    messages = build_qa_messages(question, hits)
    if trace:
        trace.set(question=question, retrieved=hits, messages=messages)

    answer = chat_complete(messages, temperature=0.2)
    if trace:
        trace.log("llm_done", chars=len(answer))
        trace.set(answer=answer)

    return {
        "question": question,
        "answer": answer,
        "citations": hits,
        "trace_id": trace.id if trace else None,
    }


def summarize_doc(doc_name: str | None = None, *, top_k: int = 12, trace: Trace | None = None) -> dict:
    """对某文档（按 doc_name 过滤）的若干 chunk 做结构化摘要。

    简化策略：用文件名当 query 检索 top_k，再交给 LLM 输出 JSON。
    """
    query = doc_name or "整体概要"
    hits = retrieve(query, top_k=top_k)
    if doc_name:
        hits = [h for h in hits if h["doc_name"] == doc_name] or hits

    if trace:
        trace.log("retrieved", count=len(hits))

    messages = build_summary_messages(hits)
    if trace:
        trace.set(target=doc_name, retrieved=hits, messages=messages)

    raw = chat_complete(messages, json_mode=True, temperature=0.1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"LLM 返回非 JSON，原文前 200 字: {raw[:200]}")
        data = {"_raw": raw, "_error": "LLM 未返回合法 JSON"}

    if trace:
        trace.set(summary=data)
        trace.log("llm_done", ok=isinstance(data, dict) and "_error" not in data)

    return {"target": doc_name, "summary": data, "citations": hits, "trace_id": trace.id if trace else None}
