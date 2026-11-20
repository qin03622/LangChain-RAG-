#!/usr/bin/env python3
"""把 data/raw 下的文档构建为向量索引。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_system import MedRAGSystem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="构建 MedRAG 向量索引")
    parser.add_argument("--dir", type=str, default=None, help="原始文档目录（默认取配置中的 DATA_DIR）")
    parser.add_argument("--no-rebuild", action="store_true", help="增量添加而非重建索引")
    args = parser.parse_args()

    system = MedRAGSystem()
    count = system.ingest(args.dir, rebuild=not args.no_rebuild)
    print(f"✅ 索引构建完成，共 {count} 个文本块。")


if __name__ == "__main__":
    main()
