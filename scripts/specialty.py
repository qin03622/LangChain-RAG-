#!/usr/bin/env python3
"""面向临床医护的专项辅助功能演示 CLI（功能二）。

用法示例:
  python scripts/specialty.py differential --presentation "发热伴干咳3天" --history "类风湿，长期用激素"
  python scripts/specialty.py lab --items "白细胞计数" "CRP"
  python scripts/specialty.py drug --drug 二甲双胍
  python scripts/specialty.py interactions --drugs 依那普利 氢氯噻嗪
  python scripts/specialty.py guidelines --disease 高血压
  python scripts/specialty.py summarize --text "患者女，45岁，发热..."
  python scripts/specialty.py template --disease 高血压
  python scripts/specialty.py pathway --subject 高血压
  python scripts/specialty.py allergy --drugs 美托洛尔 --allergies 哮喘 --population 老年 肾功能

  无参数进入交互菜单，按提示选择功能。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_system import MedRAGSystem  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="MedRAG 临床医护专项辅助功能")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("differential", aliases=["diff"], help="1. 症状鉴别辅助")
    d.add_argument("--presentation", required=True, help="主诉+体征")
    d.add_argument("--history", default="", help="基础病史")

    l = sub.add_parser("lab", aliases=["lab_interpret"], help="2. 检验/检查报告解读")
    l.add_argument("--items", nargs="+", required=True, help="指标/术语列表")

    di = sub.add_parser("drug", aliases=["drug_info"], help="3a. 单药药学信息")
    di.add_argument("--drug", required=True)
    di.add_argument("--question", default="")

    ix = sub.add_parser("interactions", aliases=["drug_interactions"], help="3b. 联用风险筛查")
    ix.add_argument("--drugs", nargs="+", required=True)

    g = sub.add_parser("guidelines", aliases=["guideline_search"], help="4. 指南共识检索")
    g.add_argument("--disease", required=True)
    g.add_argument("--version", default="latest", help="latest 最新版 | all 全部版本")

    s = sub.add_parser("summarize", aliases=["document_summarize"], help="5a. 病例要点提炼")
    s.add_argument("--text", required=True)

    t = sub.add_parser("template", aliases=["document_template"], help="5b. 书写要点模板")
    t.add_argument("--disease", required=True)

    pw = sub.add_parser("pathway", help="6. 临床路径/操作规范")
    pw.add_argument("--subject", required=True)
    pw.add_argument("--aspect", default="全部")

    a = sub.add_parser("allergy", aliases=["allergy_risk"], help="7. 过敏/特殊人群风险")
    a.add_argument("--drugs", nargs="*", default=[])
    a.add_argument("--allergies", nargs="*", default=[])
    a.add_argument("--population", nargs="*", default=[])

    return p


def dispatch(system, args):
    c = args.cmd
    op = {
        "differential": "differential", "diff": "differential",
        "lab": "lab_interpret", "lab_interpret": "lab_interpret",
        "drug": "drug_info", "drug_info": "drug_info",
        "interactions": "drug_interactions", "drug_interactions": "drug_interactions",
        "guidelines": "guideline_search", "guideline_search": "guideline_search",
        "summarize": "document_summarize", "document_summarize": "document_summarize",
        "template": "document_template", "document_template": "document_template",
        "pathway": "pathway",
        "allergy": "allergy_risk", "allergy_risk": "allergy_risk",
    }[c]
    params = vars(args).copy()
    params.pop("cmd", None)
    return system.specialty_run(op, **params)


def interactive(system):
    menu = {
        "1": ("differential", "症状鉴别辅助"),
        "2": ("lab_interpret", "检验/检查报告解读"),
        "3": ("drug_info", "单药药学信息"),
        "4": ("drug_interactions", "联用风险筛查"),
        "5": ("guideline_search", "指南共识检索"),
        "6": ("document_summarize", "病例要点提炼"),
        "7": ("document_template", "书写要点模板"),
        "8": ("pathway", "临床路径/操作规范"),
        "9": ("allergy_risk", "过敏/特殊人群风险"),
    }
    prompts = {
        "differential": [("presentation", "主诉+体征"), ("history", "基础病史(可空)")],
        "lab_interpret": [("items", "指标(逗号分隔)")],
        "drug_info": [("drug", "药品名称"), ("question", "关注点(可空)")],
        "drug_interactions": [("drugs", "药品列表(逗号分隔，至少2种)")],
        "guideline_search": [("disease", "疾病/主题"), ("version", "版本(默认 latest)")],
        "document_summarize": [("text", "病程记录原文")],
        "document_template": [("disease", "疾病名称")],
        "pathway": [("subject", "疾病/操作"), ("aspect", "关注环节(默认全部)")],
        "allergy_risk": [("drugs", "用药(逗号分隔)"), ("allergies", "过敏史(逗号分隔)"),
                         ("population", "特殊人群(逗号分隔)")],
    }
    list_fields = {"items", "drugs", "allergies", "population"}
    while True:
        print("\n" + "=" * 50)
        for k, (_, name) in menu.items():
            print(f"  {k}. {name}")
        print("  0. 退出")
        choice = input("\n请选择功能: ").strip()
        if choice == "0":
            print("再见，请注意 AI 输出不能替代医生诊断。")
            break
        if choice not in menu:
            print("无效选择。")
            continue
        op = menu[choice][0]
        params = {}
        for field, label in prompts[op]:
            value = input(f"  {label}: ").strip()
            if field in list_fields:
                params[field] = [x.strip() for x in value.replace("，", ",").split(",") if x.strip()]
            else:
                params[field] = value
        if op == "guideline_search" and not params.get("version"):
            params["version"] = "latest"
        try:
            result = system.specialty_run(op, **params)
            print(system.format_specialty_result(result))
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 出错: {exc}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    system = MedRAGSystem()

    if not args.cmd:
        interactive(system)
        return

    try:
        result = dispatch(system, args)
        print(system.format_specialty_result(result))
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 出错: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
