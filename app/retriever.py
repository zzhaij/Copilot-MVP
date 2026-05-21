"""混合检索：向量检索 + 关键词匹配，合并排序。"""
from __future__ import annotations

import re

from app.config import get_settings
from app.llm import embed_texts
from app.rerank import rerank_passages
from app.logger import logger
from app.store import get_collection


def _extract_keywords(query: str) -> list[str]:
    """从查询中提取有意义的关键词（结合中文分词和 n-gram)。"""
    # 抹掉常见停用词/语气词
    stop = {"是", "的", "了", "吗", "呢", "吧", "有", "什么", "哪些", "怎么", "如何", "谁", "哪", "多少", "请问", "请"}

    # 1) 先按中文/英文/数字粗分
    coarse = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}|\d+', query)

    # 2) 对中文 token 生成 2~4 字的 n-gram，保留原 token
    keywords: list[str] = []
    for token in coarse:
        if token in stop:
            continue
        if re.fullmatch(r'[\u4e00-\u9fff]+', token):
            length = len(token)
            if length >= 2:
                # 添加原 token
                if token not in keywords:
                    keywords.append(token)
                # 生成滑动窗口 n-gram（2~4字）
                for n in range(2, min(4, length) + 1):
                    for i in range(0, length - n + 1):
                        sub = token[i:i + n]
                        if sub not in stop and sub not in keywords:
                            keywords.append(sub)
            else:
                keywords.append(token)
        else:
            keywords.append(token)

    return keywords


def _keyword_search(coll, keywords: list[str], limit: int) -> dict:
    """使用 ChromaDB 的 where_document 做关键词包含匹配。"""
    if not keywords:
        return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    # 尝试用最长关键词进行匹配
    all_ids, all_docs, all_metas = [], [], []
    seen_ids: set[str] = set()

    for kw in sorted(keywords, key=len, reverse=True):
        try:
            res = coll.get(
                where_document={"$contains": kw},
                include=["documents", "metadatas"],
                limit=limit,
            )
            for rid, doc, meta in zip(res.get("ids") or [], res.get("documents") or [], res.get("metadatas") or []):
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_ids.append(rid)
                    all_docs.append(doc)
                    all_metas.append(meta)
        except Exception:
            continue

    return {"ids": all_ids, "documents": all_docs, "metadatas": all_metas}


def _detect_entity_target(query: str) -> str | None:
    """检测诸如“甲方是谁”“X是什么”类型问题中的主体关键字。"""
    # 匹配“xxx是谁/是哪个/是什么/为谁”等模式
    m = re.search(r'([\u4e00-\u9fff]{1,6})(?:是(?:谁|哪|什么)|为谁|为哪|有哪些)', query)
    if m:
        return m.group(1)
    # 匹配“谁是xxx”
    m = re.search(r'(?:谁|哪位)(?:是|为)([\u4e00-\u9fff]{1,6})', query)
    if m:
        return m.group(1)
    return None


def _compute_keyword_score(text: str, keywords: list[str], target: str | None, query: str) -> float:
    """根据关键词命中数量和频率计算一个 0~1 的关键词得分，并对实体问题做额外加权。"""
    if not keywords or not text:
        return 0.0
    total_hits = 0
    matched_kw = 0
    has_long_match = False
    for kw in keywords:
        count = text.count(kw)
        if count > 0:
            matched_kw += 1
            total_hits += count
            if len(kw) >= 3:
                has_long_match = True
    if total_hits == 0:
        base = 0.0
    else:
        kw_ratio = matched_kw / len(keywords)
        import math
        freq_score = min(1.0, math.log2(total_hits + 1) / 3)
        base = kw_ratio * 0.7 + freq_score * 0.3
        if has_long_match:
            base = min(1.0, base + 0.1)

    # 实体问题加权：如果文本中出现 “甲方：” “甲方为” 等模式，则提高得分
    boost = 0.0
    if target:
        if re.search(fr'{target}(?:[:：=]|\s?为)', text):
            boost = 0.25
        elif re.search(fr'(?:是|为){target}', text):
            boost = 0.15

    # 时间/日期类问题：出现“年/月/日”数字时额外加权
    if any(tok in query for tok in ("日期", "时间", "时候", "截止")):
        if re.search(r'\d{4}\s*年\s*\d{1,2}\s*月', text) or re.search(r'\d{1,2}\s*月\s*\d{1,2}\s*日', text):
            boost = max(boost, 0.2)

    return round(min(1.0, base + boost), 4)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    k = top_k or settings.TOP_K
    coll = get_collection()
    if coll.count() == 0:
        logger.warning("collection 为空，先调用 /api/upload 上传文档")
        return []

    keywords = _extract_keywords(query)
    target = _detect_entity_target(query)
    logger.debug(f"retrieve: query='{query}', keywords={keywords}, target={target}")

    # ---- 路径 1：向量检索 ----
    qvec = embed_texts([query])[0]
    vec_res = coll.query(query_embeddings=[qvec], n_results=k,
                         include=["documents", "metadatas", "distances"])

    # 构建候选池（id → info）
    candidates: dict[str, dict] = {}

    vec_docs = vec_res.get("documents", [[]])[0]
    vec_metas = vec_res.get("metadatas", [[]])[0]
    vec_dists = vec_res.get("distances", [[]])[0]
    vec_ids = vec_res.get("ids", [[]])[0]

    for rid, d, m, dist in zip(vec_ids, vec_docs, vec_metas, vec_dists):
        vec_score = round(1 - float(dist), 4)
        candidates[rid] = {
            "text": d,
            "doc_name": m.get("doc_name"),
            "page": m.get("page"),
            "chunk_index": m.get("chunk_index"),
            "vec_score": vec_score,
            "kw_score": 0.0,
        }

    # ---- 路径 2：关键词检索 ----
    if keywords:
        kw_res = _keyword_search(coll, keywords, limit=k)
        for rid, d, m in zip(kw_res["ids"], kw_res["documents"], kw_res["metadatas"]):
            if rid not in candidates:
                candidates[rid] = {
                    "text": d,
                    "doc_name": m.get("doc_name"),
                    "page": m.get("page"),
                    "chunk_index": m.get("chunk_index"),
                    "vec_score": 0.0,
                    "kw_score": 0.0,
                }

    # ---- 计算关键词得分 ----
    for rid, info in candidates.items():
        info["kw_score"] = _compute_keyword_score(info["text"], keywords, target, query)

    # ---- 混合排序：hybrid_score = 0.4 * vec_score + 0.6 * kw_score ----
    # 对于短查询（实体类），关键词权重更高；对于长查询（语义类），向量权重更高
    vec_weight = 0.7 if len(query) > 10 else (0.3 if target else 0.4)
    kw_weight = 1.0 - vec_weight

    for info in candidates.values():
        info["hybrid_score"] = round(info["vec_score"] * vec_weight + info["kw_score"] * kw_weight, 4)

    # 按 hybrid_score 降序排列
    sorted_candidates = sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)

    # 过滤 + 限制数量
    threshold = settings.SCORE_THRESHOLD
    hits: list[dict] = []
    for i, info in enumerate(sorted_candidates[:k], start=1):
        if info["hybrid_score"] < threshold and info["vec_score"] < threshold and info["kw_score"] < threshold:
            logger.debug(f"skip (hybrid={info['hybrid_score']}, vec={info['vec_score']}, kw={info['kw_score']}): {info['text'][:60]}")
            continue
        hits.append({
            "rank": i,
            "text": info["text"],
            "doc_name": info["doc_name"],
            "page": info["page"],
            "chunk_index": info["chunk_index"],
            "score": info["hybrid_score"],
            "hybrid_score": info["hybrid_score"],
            "vec_score": info["vec_score"],
            "kw_score": info["kw_score"],
        })

    if not hits:
        logger.warning(f"所有检索结果低于阈值 {threshold}，可能文档不包含相关内容")

    if hits:
        hits = rerank_passages(query, hits)

    return hits
