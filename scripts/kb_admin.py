#!/usr/bin/env python3
"""知识库管理后台 CLI 演示（功能四 · LangChain RAG 配套管理后台）。

覆盖 6 个子功能的命令行入口：
  1) 文档上传与增量入库（支持 PDF/DOCX/MD/TXT，医疗分隔符分块）
  2) 批量导入 / 向量化
  3) 分类目录管理
  4) 文档版本管理（归档 / 失效 / 恢复，检索默认过滤失效）
  5) 权限隔离（public / department / confidential）
  6) 检索测试面板（召回 → 过滤 → 重排，调 chunk/top-k/rerank）

用法示例:
  # 上传单篇（自动归档同名旧版为 archived）
  python scripts/kb_admin.py upload data/raw/hypertension.md --category 临床指南 --title 高血压指南 --version v2.0

  # 涉密文档：仅指定账号可检索
  python scripts/kb_admin.py upload data/raw/secret.md --visibility confidential --allow alice bob

  # 批量导入目录下全部文档
  python scripts/kb_admin.py batch data/raw --category 内科

  # 分类目录
  python scripts/kb_admin.py categories
  python scripts/kb_admin.py add-category 感染控制
  python scripts/kb_admin.py stats

  # 版本管理
  python scripts/kb_admin.py versions --title 高血压指南
  python scripts/kb_admin.py invalidate doc-xxxxxxxx
  python scripts/kb_admin.py archive doc-xxxxxxxx
  python scripts/kb_admin.py reactivate doc-xxxxxxxx

  # 检索测试面板（含 ACL / 失效过滤 / 重排 / 参数调优）
  python scripts/kb_admin.py debug "高血压的诊断标准" --top-k 3 --recall-k 10 --account alice --rerank

  # 带权限的问答
  python scripts/kb_admin.py query "高血压的诊断标准" --account alice --department 心内科

  # 总览 / 重建索引
  python scripts/kb_admin.py overview
  python scripts/kb_admin.py reindex

  无参数进入交互菜单。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_system import MedRAGSystem  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="MedRAG 知识库管理后台 CLI（功能四）")
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("upload", help="上传单篇文档并增量入库")
    up.add_argument("file")
    up.add_argument("--category", default="临床指南")
    up.add_argument("--title", default=None)
    up.add_argument("--version", default="v1.0")
    up.add_argument("--visibility", default="public", choices=["public", "department", "confidential"])
    up.add_argument("--allow", nargs="*", default=[], help="允许的账号（confidential）或科室标识（department）")
    up.add_argument("--separators", nargs="*", default=None, help="自定义切分分隔符，如 第 条 章节")
    up.add_argument("--chunk-size", type=int, default=None)
    up.add_argument("--no-replace", action="store_true", help="不自动归档同名旧版")

    ba = sub.add_parser("batch", help="批量导入目录下全部文档")
    ba.add_argument("directory")
    ba.add_argument("--category", default="临床指南")
    ba.add_argument("--visibility", default="public", choices=["public", "department", "confidential"])
    ba.add_argument("--allow", nargs="*", default=[])

    sub.add_parser("categories", help="列出分类目录")
    ac = sub.add_parser("add-category", help="新增分类")
    ac.add_argument("name")
    sub.add_parser("stats", help="分类统计")

    ve = sub.add_parser("versions", help="查看版本历史")
    ve.add_argument("--title", default=None)
    ve.add_argument("--doc-id", default=None)

    for cmd, help_txt in [
        ("invalidate", "标记失效（检索默认过滤）"),
        ("archive", "归档旧版"),
        ("reactivate", "恢复为 active"),
    ]:
        sp = sub.add_parser(cmd, help=help_txt)
        sp.add_argument("doc_id")

    db = sub.add_parser("debug", help="检索测试面板")
    db.add_argument("question")
    db.add_argument("--top-k", type=int, default=None)
    db.add_argument("--recall-k", type=int, default=None)
    db.add_argument("--chunk-size", type=int, default=None)
    db.add_argument("--account", default=None)
    db.add_argument("--department", default=None)
    db.add_argument("--category", default=None)
    db.add_argument("--rerank", action="store_true", default=None)
    db.add_argument("--include-invalid", action="store_true")

    qy = sub.add_parser("query", help="带权限的知识库问答")
    qy.add_argument("question")
    qy.add_argument("--account", default=None)
    qy.add_argument("--department", default=None)
    qy.add_argument("--category", default=None)
    qy.add_argument("--top-k", type=int, default=5)

    sub.add_parser("overview", help="知识库总览")
    sub.add_parser("reindex", help="从 registry 全量重建向量索引")
    return p


def dump(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    args = build_parser().parse_args(argv)
    system = MedRAGSystem()
    kb = system.kb

    if args.cmd == "upload":
        rec = kb.upload_file(
            args.file,
            category=args.category,
            title=args.title,
            version=args.version,
            visibility=args.visibility,
            allowed_accounts=args.allow or None,
            separators=args.separators,
            chunk_size=args.chunk_size,
            replace_latest=not args.no_replace,
        )
        dump(rec)

    elif args.cmd == "batch":
        files = [
            f for f in Path(args.directory).rglob("*")
            if f.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
        ]
        items = [(f, {
            "category": args.category,
            "visibility": args.visibility,
            "allowed_accounts": args.allow or None,
        }) for f in files]
        dump(kb.upload_batch(items))

    elif args.cmd == "categories":
        dump(kb.list_categories())

    elif args.cmd == "add-category":
        ok = kb.add_category(args.name)
        dump({"added": ok, "categories": kb.list_categories()})

    elif args.cmd == "stats":
        dump(kb.category_stats())

    elif args.cmd == "versions":
        dump(kb.list_versions(title=args.title, doc_id=args.doc_id))

    elif args.cmd in {"invalidate", "archive", "reactivate"}:
        dump(getattr(kb, args.cmd)(args.doc_id))

    elif args.cmd == "debug":
        dump(kb.debug_retrieval(
            args.question,
            account=args.account,
            department=args.department,
            category=args.category,
            top_k=args.top_k,
            recall_k=args.recall_k,
            chunk_size=args.chunk_size,
            use_rerank=args.rerank,
            include_invalid=args.include_invalid,
        ))

    elif args.cmd == "query":
        dump(system.kb_query(
            args.question,
            account=args.account,
            department=args.department,
            category=args.category,
            top_k=args.top_k,
        ))

    elif args.cmd == "overview":
        dump(kb.overview())

    elif args.cmd == "reindex":
        # 用所有 active 版本文档的 source_path 重新走一遍入库流程，重建向量索引
        from pathlib import Path as _Path
        items = []
        for rec in kb.documents.values():
            if rec.get("status") == "active" and _Path(rec["source_path"]).exists():
                items.append((rec["source_path"], {
                    "category": rec["category"],
                    "title": rec["title"],
                    "version": rec["version"],
                    "visibility": rec["visibility"],
                    "allowed_accounts": rec.get("allowed_accounts") or None,
                    "replace_latest": False,
                }))
        dump({"rebuilt": len(kb.upload_batch(items)), "total": len(kb.documents)})

    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
