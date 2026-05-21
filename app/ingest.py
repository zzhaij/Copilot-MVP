"""资料解析 → 切分 → 向量化 → 入库。

设计取舍：
- 仅支持文字版 PDF / DOCX / MD / TXT；扫描件 OCR 不在 MVP 范围。
- 切分器用 langchain 的 RecursiveCharacterTextSplitter，对中英文混排表现稳定。
- 每个 chunk 携带 metadata: doc_id / doc_name / chunk_index / page（PDF 才有）。
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DATA_DIR, get_settings
from app.llm import embed_texts
from app.logger import logger
from app.store import get_collection

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


# ---------- 解析 ----------

def _parse_pdf(path: Path) -> list[tuple[int, str]]:
    """返回 [(page_no, text)]，page_no 从 1 起。"""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _parse_docx(path: Path) -> list[tuple[int, str]]:
    from docx import Document
    doc = Document(str(path))
    full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(0, full)] if full.strip() else []


def _parse_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(0, text)] if text.strip() else []


def parse_file(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in {".md", ".markdown", ".txt"}:
        return _parse_text(path)
    raise ValueError(f"unsupported extension: {ext}")


# ---------- 切分 + 入库 ----------

def _split(text: str) -> list[str]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _doc_id(path: Path) -> str:
    h = hashlib.md5(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem}-{h}"


def ingest_file(path: Path, *, trace=None) -> dict:
    """解析+入库；返回统计信息。"""
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    doc_id = _doc_id(path)
    pages = parse_file(path)
    if not pages:
        raise ValueError("文件解析为空，可能是扫描件或不支持的编码")

    if trace:
        trace.log("parsed", doc_id=doc_id, pages=len(pages))

    # 切分（保留页码信息）
    chunk_texts: list[str] = []
    metadatas: list[dict] = []
    for page_no, text in pages:
        for i, chunk in enumerate(_split(text)):
            chunk_texts.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "doc_name": path.name,
                "page": page_no,
                "chunk_index": i,
            })

    if trace:
        trace.log("chunked", chunks=len(chunk_texts))

    if not chunk_texts:
        raise ValueError("切分结果为空")

    embeddings = embed_texts(chunk_texts)
    ids = [f"{doc_id}::{uuid.uuid4().hex[:8]}" for _ in chunk_texts]

    coll = get_collection()
    # 去重：如果同一 doc_id 已入库，先删除旧数据
    existing = coll.get(where={"doc_id": doc_id}, include=[])
    if existing and existing.get("ids"):
        coll.delete(ids=existing["ids"])
        logger.info(f"ingest: 已删除旧版本 {doc_id} ({len(existing['ids'])} chunks)")

    coll.add(ids=ids, documents=chunk_texts, metadatas=metadatas, embeddings=embeddings)
    logger.info(f"ingest: {path.name} -> {len(chunk_texts)} chunks")

    if trace:
        trace.log("indexed", count=len(chunk_texts))

    return {"doc_id": doc_id, "doc_name": path.name, "chunks": len(chunk_texts), "pages": len(pages)}


def save_upload(filename: str, content: bytes) -> Path:
    """保存上传文件到 data/uploads/。"""
    safe = Path(filename).name
    target = DATA_DIR / safe
    target.write_bytes(content)
    return target


def list_documents() -> list[dict]:
    """从 collection 元数据聚合已入库文档。"""
    coll = get_collection()
    # peek 前 N 条用于预览；列出全部用 get
    res = coll.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for md in res.get("metadatas", []) or []:
        did = md.get("doc_id")
        if did and did not in seen:
            seen[did] = {"doc_id": did, "doc_name": md.get("doc_name")}
    return list(seen.values())
