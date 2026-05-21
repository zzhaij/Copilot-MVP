"""命令行入口：python -m app.cli ingest|ask|summarize|db"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingest import ingest_file
from app.pipeline import answer_question, summarize_doc
from app.tracing import new_trace


def cmd_ingest(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"文件不存在: {path}")
    with new_trace("ingest") as t:
        t.set(filename=path.name)
        stats = ingest_file(path, trace=t)
    print(json.dumps({"trace_id": t.id, **stats}, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    with new_trace("ask") as t:
        result = answer_question(args.question, top_k=args.top_k, trace=t)
    print("=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  [{c['rank']}] {c['doc_name']} p{c['page']} score={c['score']}")
        print(f"{c['text']}")
    print(f"\ntrace_id = {result['trace_id']}")


def cmd_summarize(args: argparse.Namespace) -> None:
    with new_trace("summarize") as t:
        result = summarize_doc(args.doc_name, top_k=args.top_k, trace=t)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\ntrace_id = {result['trace_id']}")


def cmd_db(args: argparse.Namespace) -> None:
    from app.store import get_collection, reset_collection
    coll = get_collection()
    action = args.action

    if action == "stats":
        count = coll.count()
        print(f"Collection: {coll.name}")
        print(f"Total chunks: {count}")
        # 按文档聚合
        if count > 0:
            res = coll.get(include=["metadatas"])
            docs: dict[str, int] = {}
            for md in res.get("metadatas") or []:
                name = md.get("doc_name", "unknown")
                docs[name] = docs.get(name, 0) + 1
            print(f"Documents: {len(docs)}")
            for name, cnt in docs.items():
                print(f"  - {name}: {cnt} chunks")

    elif action == "list":
        res = coll.get(include=["metadatas", "documents"])
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        docs_text = res.get("documents") or []
        for i, (rid, md) in enumerate(zip(ids, metas)):
            text = docs_text[i] if i < len(docs_text) else ""
            preview = (text[:80] + "...") if len(text) > 80 else text
            print(f"[{i}] id={rid}  doc={md.get('doc_name')}  page={md.get('page')}  chunk={md.get('chunk_index')}")
            print(f"    {preview}")

    elif action == "peek":
        n = args.n or 5
        res = coll.peek(n)
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        docs_text = res.get("documents") or []
        for i, (rid, md) in enumerate(zip(ids, metas)):
            text = docs_text[i] if i < len(docs_text) else ""
            preview = (text[:120] + "...") if len(text) > 120 else text
            print(f"--- [{i}] id={rid} ---")
            print(f"    meta: {json.dumps(md, ensure_ascii=False)}")
            print(f"    text: {preview}\n")

    elif action == "delete":
        doc_name = args.doc_name
        if not doc_name:
            raise SystemExit("请指定要删除的文档名，如: db delete sample_contract.md")
        res = coll.get(where={"doc_name": doc_name}, include=["metadatas"])
        ids = res.get("ids") or []
        if not ids:
            print(f"未找到文档: {doc_name}")
            return
        coll.delete(ids=ids)
        print(f"已删除 {doc_name}: {len(ids)} chunks")

    elif action == "reset":
        reset_collection()
        print("已清空全部数据，collection 已重建。")

    else:
        print(f"未知操作: {action}，可选: stats / list / peek / delete / reset")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="导入一个文档")
    p_ing.add_argument("path")
    p_ing.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="问答")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_sum = sub.add_parser("summarize", help="结构化摘要")
    p_sum.add_argument("doc_name", nargs="?", default=None)
    p_sum.add_argument("--top-k", type=int, default=12)
    p_sum.set_defaults(func=cmd_summarize)

    p_db = sub.add_parser("db", help="查看/管理 ChromaDB")
    p_db.add_argument("action", choices=["stats", "list", "peek", "delete", "reset"],
                      help="stats=统计 | list=列出全部 | peek=预览 | delete=删除文档 | reset=清空")
    p_db.add_argument("doc_name", nargs="?", default=None, help="delete 时指定文档名")
    p_db.add_argument("-n", type=int, default=5, help="peek 时的条数")
    p_db.set_defaults(func=cmd_db)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
