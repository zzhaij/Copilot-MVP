"""统一日志：控制台 + 文件落盘。"""
from __future__ import annotations

import sys

from loguru import logger

from app.config import LOG_DIR

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}")
logger.add(LOG_DIR / "run.log", level="DEBUG", rotation="10 MB",
           encoding="utf-8", enqueue=True,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} | {message}")

__all__ = ["logger"]
