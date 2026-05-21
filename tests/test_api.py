"""端到端冒烟测试：验证 API 三条主链路正常。

使用方式：
  pytest tests/test_api.py -v

前提：
  1. .env 正确配置 LLM_API_KEY、EMBED_API_KEY
  2. samples/ 下至少有一份测试文件
"""
from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = ROOT / "samples" / "sample_contract.md"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_and_ask(client):
    # upload
    with open(SAMPLE_FILE, "rb") as f:
        r = await client.post("/api/upload", files={"file": ("sample_contract.md", f, "text/markdown")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["chunks"] > 0

    # ask
    r = await client.post("/api/ask", json={"question": "合同的违约责任有哪些？"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["citations"]


@pytest.mark.asyncio
async def test_summarize(client):
    r = await client.post("/api/summarize", json={"doc_name": "sample_contract.md", "top_k": 6})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "summary" in data
    summary = data["summary"]
    assert "key_points" in summary or "_raw" in summary
