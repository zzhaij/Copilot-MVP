"""Prompt 模板：QA 与结构化摘要。"""
from __future__ import annotations

QA_SYSTEM = """你是企业资料问答助手。回答必须严格基于提供的【参考资料】。

规则：
1. 答案中的每个事实点都要在末尾用 [序号] 标注引用，序号对应【参考资料】的编号。
2. 若【参考资料】不足以回答，请直接回答"根据现有资料无法确定"，不要编造。
3. 回答简洁，使用中文，必要时使用要点列表。
"""

QA_USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}

请基于以上资料作答，并标注引用序号。"""


def build_qa_messages(question: str, hits: list[dict]) -> list[dict]:
    if hits:
        ctx_lines = []
        for h in hits:
            tag = f"[{h['rank']}] {h['doc_name']}"
            if h.get("page"):
                tag += f" 第{h['page']}页"
            ctx_lines.append(f"{tag}\n{h['text']}")
        context = "\n\n---\n\n".join(ctx_lines)
    else:
        context = "(无)"
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QA_USER_TEMPLATE.format(context=context, question=question)},
    ]


SUMMARY_SYSTEM = """你是企业资料结构化摘要助手。你会收到若干文档片段，请输出 JSON。

输出 JSON Schema：
{
  "title": "整体主题（一句话）",
  "key_points": ["要点1", "要点2", ...],   // 3-7 条
  "entities": ["关键实体/人名/产品/组织", ...],
  "action_items": ["可执行事项", ...],       // 若无可为空数组
  "risks": ["潜在风险或不确定点", ...]       // 若无可为空数组
}

只输出 JSON，不要任何额外文字。所有字段使用中文。"""

SUMMARY_USER_TEMPLATE = """【文档片段】
{context}

请按 schema 输出 JSON 摘要。"""


def build_summary_messages(hits: list[dict]) -> list[dict]:
    ctx = "\n\n---\n\n".join(f"[{h['rank']}] {h['doc_name']}\n{h['text']}" for h in hits) or "(无)"
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(context=ctx)},
    ]
