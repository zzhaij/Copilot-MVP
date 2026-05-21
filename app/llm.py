"""LLM 与 Embedding 客户端封装（OpenAI 兼容协议）。"""
from __future__ import annotations

from typing import Iterable

import httpx
from openai import OpenAI

from app.config import get_settings
from app.logger import logger

_settings = get_settings()

_llm_client: OpenAI | None = None
_embed_client: OpenAI | None = None


def _make_http_client() -> httpx.Client | None:
    """若配置了 HTTP_PROXY，则创建带代理的 httpx 客户端。"""
    proxy = _settings.HTTP_PROXY
    if not proxy:
        return None
    logger.info(f"使用代理: {proxy}")
    return httpx.Client(proxy=proxy, verify=True)


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        if not _settings.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY 未配置，请检查 .env")
        kwargs: dict = {"base_url": _settings.LLM_BASE_URL, "api_key": _settings.LLM_API_KEY}
        hc = _make_http_client()
        if hc:
            kwargs["http_client"] = hc
        _llm_client = OpenAI(**kwargs)
    return _llm_client


def get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        if not _settings.EMBED_API_KEY:
            raise RuntimeError("EMBED_API_KEY 未配置，请检查 .env")
        kwargs: dict = {"base_url": _settings.EMBED_BASE_URL, "api_key": _settings.EMBED_API_KEY}
        hc = _make_http_client()
        if hc:
            kwargs["http_client"] = hc
        _embed_client = OpenAI(**kwargs)
    return _embed_client


_local_ef = None


def _get_local_ef():
    """懒加载本地 embedding function（多语言模型，支持中文，384 维）。"""
    global _local_ef
    if _local_ef is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _local_ef = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        logger.info("本地 Embedding 模型已加载 (paraphrase-multilingual-MiniLM-L12-v2)")
    return _local_ef


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """向量化入口。根据 EMBED_PROVIDER 决定使用本地模型或 API。"""
    texts = list(texts)
    if _settings.EMBED_PROVIDER == "local":
        ef = _get_local_ef()
        out = ef(texts)
        logger.debug(f"embed_texts[local]: {len(texts)} 条 -> dim={len(out[0]) if out else 0}")
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in out]

    # API 模式
    client = get_embed_client()
    out: list[list[float]] = []
    batch = 64
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=_settings.EMBED_MODEL, input=chunk)
        out.extend([d.embedding for d in resp.data])
    logger.debug(f"embed_texts[api]: {len(texts)} 条 -> dim={len(out[0]) if out else 0}")
    return out


def chat_complete(messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2) -> str:
    """非流式调用。json_mode=True 时要求模型输出 JSON。"""
    client = get_llm_client()
    kwargs: dict = {
        "model": _settings.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage:
        logger.debug(f"llm usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    return content
