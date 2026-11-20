#!/usr/bin/env python3
"""临床增强问答演示 CLI：双模式 + 多轮记忆 + 引用溯源 + 反馈。

用法:
    python scripts/clinical.py --mode precise "高血压的诊断标准？"
    python scripts/clinical.py --mode comprehensive "老年糖尿病合并肌酐升高如何选药？"
    python scripts/clinical.py            # 交互多轮对话（自动记忆上下文）

交互模式命令:
    feedback correct|hallucination|outdated|incomplete  提交上一条回答的反馈
    session   查看当前会话 ID
    new       开启新会话
    exit      退出
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_system import MedRAGSystem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="MedRAG 临床增强问答")
    parser.add_argument("question", nargs="?", help="问题；不提供则进入交互模式")
    parser.add_argument(
        "--mode", choices=["precise", "comprehensive"], default=None, help="检索/推理模式"
    )
    args = parser.parse_args()

    system = MedRAGSystem()
    session_id = system.new_session()
    last_result = {}

    if args.question:
        result = system.clinical_query(args.question, mode=args.mode, session_id=session_id)
        print(system.clinical.format_result(result))
        return

    print("进入 MedRAG 临床问答（多轮记忆已开启，输入 feedback correct 等可提交反馈）")
    while True:
        q = input("\n🧑 医生: ").strip()
        if q.lower() in {"exit", "quit", "退出"}:
            print("再见，请注意：AI 回答不能替代医生诊断。")
            break
        if not q:
            continue
        if q == "session":
            print(f"当前会话: {session_id}")
            continue
        if q == "new":
            session_id = system.new_session()
            print(f"已开启新会话: {session_id}")
            continue
        if q.startswith("feedback "):
            label = q.split(maxsplit=1)[1].strip()
            if not last_result:
                print("还没有可反馈的回答。")
                continue
            fb = system.submit_feedback(
                last_result["question"], last_result["answer"], label, result=last_result
            )
            print(f"✅ 反馈已记录 ({fb.get('status', 'ok')}): {label}")
            continue

        try:
            result = system.clinical_query(q, mode=args.mode, session_id=session_id)
            last_result = {"question": q, **result}
            print(system.clinical.format_result(result))
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 出错: {exc}")


if __name__ == "__main__":
    main()
