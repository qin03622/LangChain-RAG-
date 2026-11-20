#!/usr/bin/env python3
"""后台运维 & 统计管理 CLI（功能五）。

覆盖四组子命令：
  1) 问答统计：stats
  2) 模型切换 / RAG 开关：models / model / switch-model / rag / compare
  3) 参数可视化配置：params / set-param / reset-params
  4) 会话管理：sessions / session / delete-session / purge-sensitive

用法示例:
  python scripts/ops_admin.py stats --top-n 5
  python scripts/ops_admin.py models
  python scripts/ops_admin.py switch-model --provider fake --model fake
  python scripts/ops_admin.py rag --off              # 关闭 RAG，进入纯 LLM 模式
  python scripts/ops_admin.py rag --on               # 开启 RAG
  python scripts/ops_admin.py set-param --chunk-size 300 --top-k 5 --temperature 0.3
  python scripts/ops_admin.py params
  python scripts/ops_admin.py compare "高血压的诊断标准是什么？"
  python scripts/ops_admin.py sessions
  python scripts/ops_admin.py session <session_id>
  python scripts/ops_admin.py delete-session <session_id>
  python scripts/ops_admin.py purge-sensitive

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
    p = argparse.ArgumentParser(description="MedRAG 后台运维 CLI（功能五）")
    sub = p.add_subparsers(dest="cmd")

    st = sub.add_parser("stats", help="问答统计：高频病种/问题/反馈")
    st.add_argument("--top-n", type=int, default=10)
    st.add_argument("--keyword", default=None, help="仅统计含该关键词的问题")

    sub.add_parser("models", help="列出可切换的模型")
    sub.add_parser("model", help="查看当前模型与 RAG 状态")

    sw = sub.add_parser("switch-model", help="热切换 LLM")
    sw.add_argument("--provider", default=None)
    sw.add_argument("--model", default=None)
    sw.add_argument("--temperature", type=float, default=None)

    rag = sub.add_parser("rag", help="开关 RAG（纯 LLM vs RAG）")
    g = rag.add_mutually_exclusive_group(required=True)
    g.add_argument("--on", action="store_true", dest="enabled", help="开启 RAG")
    g.add_argument("--off", action="store_true", dest="disabled", help="关闭 RAG（纯 LLM）")

    sub.add_parser("params", help="查看当前可调参数")
    sp = sub.add_parser("set-param", help="设置运行时参数")
    sp.add_argument("--chunk-size", type=int, default=None)
    sp.add_argument("--chunk-overlap", type=int, default=None)
    sp.add_argument("--top-k", type=int, default=None)
    sp.add_argument("--rerank-threshold", type=float, default=None)
    sp.add_argument("--use-hyde", action="store_true", default=None, dest="use_hyde_true")
    sp.add_argument("--no-use-hyde", action="store_true", default=None, dest="use_hyde_false")
    sp.add_argument("--temperature", type=float, default=None)
    sub.add_parser("reset-params", help="重置所有运行时参数")

    cp = sub.add_parser("compare", help="纯 LLM vs RAG A/B 对比")
    cp.add_argument("question")
    cp.add_argument("--mode", default=None, choices=["precise", "comprehensive"])
    cp.add_argument("--session-id", default=None)

    ls = sub.add_parser("sessions", help="查看历史会话列表")
    ls.add_argument("--include-expired", action="store_true")

    sg = sub.add_parser("session", help="查看单个会话对话记录")
    sg.add_argument("session_id")

    de = sub.add_parser("delete-session", help="手动删除会话")
    de.add_argument("session_id")

    sub.add_parser("purge-sensitive", help="批量清理敏感会话")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    system = MedRAGSystem()

    if args.cmd is None:
        _interactive(system)
        return

    ops = system.ops
    if args.cmd == "stats":
        out = ops.stats(top_n=args.top_n, keyword=args.keyword)
    elif args.cmd == "models":
        out = ops.list_models()
    elif args.cmd == "model":
        out = ops.current_model()
    elif args.cmd == "switch-model":
        out = ops.switch_model(
            args.provider, args.model, temperature=args.temperature
        )
    elif args.cmd == "rag":
        out = ops.set_rag(not args.disabled)
    elif args.cmd == "params":
        out = ops.get_params()
    elif args.cmd == "set-param":
        params = {}
        for k in ("chunk_size", "chunk_overlap", "top_k", "rerank_threshold", "temperature"):
            v = getattr(args, k, None)
            if v is not None:
                params[k] = v
        if args.use_hyde_true:
            params["use_hyde"] = True
        if args.use_hyde_false:
            params["use_hyde"] = False
        if not params:
            parser.error("至少提供一个参数")
        out = ops.set_params(**params)
    elif args.cmd == "reset-params":
        out = ops.reset_params()
    elif args.cmd == "compare":
        out = ops.compare(args.question, mode=args.mode, session_id=args.session_id)
    elif args.cmd == "sessions":
        out = ops.list_sessions(include_expired=args.include_expired)
    elif args.cmd == "session":
        out = ops.get_session(args.session_id)
    elif args.cmd == "delete-session":
        out = ops.delete_session(args.session_id)
    elif args.cmd == "purge-sensitive":
        out = ops.purge_sensitive_sessions()
    else:  # pragma: no cover
        parser.error("未知命令")
        return
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def _interactive(system):
    """无参数时的极简交互菜单。"""
    ops = system.ops
    menu = """
===== MedRAG 后台运维（功能五）=====
 1  问答统计
 2  模型列表 / 当前状态
 3  切换模型
 4  开关 RAG
 5  查看参数
 6  设置参数
 7  纯 LLM vs RAG 对比
 8  会话列表
 9  删除会话
 10 清理敏感会话
  0  退出
"""
    while True:
        print(menu)
        choice = input("请选择: ").strip()
        try:
            if choice == "1":
                print(json.dumps(ops.stats(), ensure_ascii=False, indent=2, default=str))
            elif choice == "2":
                print(json.dumps({"models": ops.list_models(), "current": ops.current_model()},
                                 ensure_ascii=False, indent=2, default=str))
            elif choice == "3":
                provider = input("provider(openai/ollama/fake，回车跳过): ").strip() or None
                model = input("model(回车跳过): ").strip() or None
                print(json.dumps(ops.switch_model(provider, model), ensure_ascii=False, indent=2, default=str))
            elif choice == "4":
                val = input("开启 RAG? (y/n): ").strip().lower() == "y"
                print(json.dumps(ops.set_rag(val), ensure_ascii=False, indent=2))
            elif choice == "5":
                print(json.dumps(ops.get_params(), ensure_ascii=False, indent=2, default=str))
            elif choice == "6":
                raw = input("参数 JSON(如 {\"top_k\": 5}): ").strip()
                print(json.dumps(ops.set_params(**json.loads(raw)), ensure_ascii=False, indent=2))
            elif choice == "7":
                q = input("问题: ").strip()
                print(json.dumps(ops.compare(q), ensure_ascii=False, indent=2, default=str))
            elif choice == "8":
                print(json.dumps(ops.list_sessions(), ensure_ascii=False, indent=2, default=str))
            elif choice == "9":
                print(json.dumps(ops.delete_session(input("会话 ID: ").strip()),
                                 ensure_ascii=False, indent=2))
            elif choice == "10":
                print(json.dumps(ops.purge_sensitive_sessions(), ensure_ascii=False, indent=2))
            elif choice == "0":
                break
            else:
                print("无效选择")
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] {exc}")


if __name__ == "__main__":
    main()
