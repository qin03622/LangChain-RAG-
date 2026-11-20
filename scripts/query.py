#!/usr/bin/env python3
"""命令行问答入口。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_system import MedRAGSystem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="MedRAG 医疗问答")
    parser.add_argument("question", nargs="?", help="要询问的问题；不提供则进入交互模式")
    parser.add_argument("--no-sources", action="store_true", help="不打印参考来源")
    args = parser.parse_args()

    system = MedRAGSystem()

    if args.question:
        result = system.query(args.question)
        if args.no_sources:
            print(result["answer"])
        else:
            print(system.format_result(result))
        return

    print("进入 MedRAG 交互问答（输入 exit / quit 退出）")
    while True:
        q = input("\n🧑 您: ").strip()
        if q.lower() in {"exit", "quit", "退出"}:
            print("再见，请注意：AI 回答不能替代医生诊断。")
            break
        if not q:
            continue
        try:
            result = system.query(q)
            print(system.format_result(result))
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 出错: {exc}")


if __name__ == "__main__":
    main()
