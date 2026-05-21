"""集中式配置：从环境变量读取，启动时校验一次。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "uploads"
CHROMA_DIR = ROOT_DIR / "chroma_db"
TRACE_DIR = ROOT_DIR / "traces"
LOG_DIR = ROOT_DIR / "logs"

for _d in (DATA_DIR, CHROMA_DIR, TRACE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"

    # Embedding 提供方："local"=本地模型(无需网络), "api"=调用外部 API
    EMBED_PROVIDER: str = "local"
    EMBED_BASE_URL: str = "https://api.openai.com/v1"
    EMBED_API_KEY: str = ""
    EMBED_MODEL: str = "text-embedding-3-small"

    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    SCORE_THRESHOLD: float = 0.25

    COLLECTION_NAME: str = "enterprise_docs"

    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    QA_MODEL: str = "uer/roberta-base-chinese-extractive-qa"
    QA_SCORE_THRESHOLD: float = 0.35
    QA_MAX_LENGTH: int = 384
    QA_MAX_ANSWER_LEN: int = 48
    QA_TOP_K: int = 4
    RERANK_REL_WEIGHT: float = 0.55
    RERANK_QA_WEIGHT: float = 0.45
    RERANK_HAS_ANSWER_BONUS: float = 0.05
    RERANK_KEYWORD_PENALTY: float = 0.3

    # 网络代理（留空则不使用）
    HTTP_PROXY: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
