"""FastAPI 主入口。"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT_DIR, TRACE_DIR, get_settings
from app.ingest import SUPPORTED_EXTS, ingest_file, list_documents, save_upload
from app.logger import logger
from app.pipeline import answer_question, summarize_doc
from app.tracing import new_trace

app = FastAPI(title="Enterprise Doc Copilot MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------- Schema ----------

class AskReq(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


class SummaryReq(BaseModel):
    doc_name: str | None = None
    top_k: int = 12


# ---------- 路由 ----------

@app.get("/api/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "llm_model": s.LLM_MODEL,
        "embed_model": s.EMBED_MODEL,
        "llm_configured": bool(s.LLM_API_KEY),
        "embed_configured": bool(s.EMBED_API_KEY),
    }


@app.get("/api/documents")
def documents():
    return {"documents": list_documents()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400, f"不支持的文件类型: {ext}; 支持: {sorted(SUPPORTED_EXTS)}")

    content = await file.read()
    path = save_upload(file.filename, content)
    logger.info(f"upload received: {path.name} ({len(content)} bytes)")

    with new_trace("ingest") as t:
        t.set(filename=file.filename, size=len(content))
        try:
            stats = ingest_file(path, trace=t)
        except Exception as e:  # noqa: BLE001
            t.log("error", msg=str(e))
            logger.exception("ingest failed")
            raise HTTPException(500, f"入库失败: {e}")
        t.set(stats=stats)
        return {"ok": True, "trace_id": t.id, **stats}


@app.post("/api/ask")
def ask(req: AskReq):
    with new_trace("ask") as t:
        try:
            result = answer_question(req.question, top_k=req.top_k, trace=t)
        except Exception as e:  # noqa: BLE001
            t.log("error", msg=str(e))
            logger.exception("ask failed")
            raise HTTPException(500, f"问答失败: {e}")
        return result


@app.post("/api/summarize")
def summarize(req: SummaryReq):
    with new_trace("summarize") as t:
        try:
            result = summarize_doc(req.doc_name, top_k=req.top_k, trace=t)
        except Exception as e:  # noqa: BLE001
            t.log("error", msg=str(e))
            logger.exception("summarize failed")
            raise HTTPException(500, f"摘要失败: {e}")
        return result


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str):
    # trace 文件名形如 {kind}_{id}.json
    matches = list(TRACE_DIR.glob(f"*_{trace_id}.json"))
    if not matches:
        raise HTTPException(404, "trace 不存在")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/api/traces")
def list_traces(limit: int = 30):
    files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    items = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            items.append({
                "id": d.get("id"),
                "kind": d.get("kind"),
                "elapsed_sec": d.get("elapsed_sec"),
                "file": f.name,
            })
        except Exception:
            continue
    return {"traces": items}


# ---------- 静态前端 ----------

WEB_DIR = ROOT_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.APP_HOST, port=s.APP_PORT, reload=False)
