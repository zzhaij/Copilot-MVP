"""请求级 trace：每次链路调用落盘成 JSON，便于评审查阅。"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import TRACE_DIR
from app.logger import logger


class Trace:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.t0 = time.time()
        self.events: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = {}

    def log(self, event: str, **fields: Any) -> None:
        item = {"event": event, "ts": round(time.time() - self.t0, 4), **fields}
        self.events.append(item)
        logger.debug(f"[trace {self.id}] {event} {fields}")

    def set(self, **fields: Any) -> None:
        self.payload.update(fields)

    def dump(self) -> Path:
        path = TRACE_DIR / f"{self.kind}_{self.id}.json"
        data = {
            "id": self.id,
            "kind": self.kind,
            "elapsed_sec": round(time.time() - self.t0, 4),
            "payload": self.payload,
            "events": self.events,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


@contextmanager
def new_trace(kind: str):
    t = Trace(kind)
    try:
        yield t
    finally:
        try:
            p = t.dump()
            logger.info(f"trace saved -> {p.name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"trace dump failed: {e}")
