"""ChromaDB 持久化向量库封装。"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import CHROMA_DIR, get_settings


@lru_cache
def get_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    # 我们自己提供 embedding，所以 embedding_function=None
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """测试使用：清空集合。"""
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass
    get_collection.cache_clear()
    get_collection()
