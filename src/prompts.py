"""Prompt 模板：中文医疗临床场景，支持精准检索/综合推理双模式、多轮上下文、反馈调优。"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

SYSTEM_PROMPT = """你是一名严谨的医疗临床智能助手，服务于医生的临床教育性问答。
请严格遵守以下规则：
1. 只依据【参考资料】回答。如果参考资料不足以回答，明确说"根据现有资料无法回答"，不要编造。
2. 区分"资料中明确记载的事实"与"你的推断"，推断需标注。
3. 涉及用药剂量、禁忌症、急症处理时，提醒用户咨询执业医师，不做最终处方决定。
4. 用简体中文回答，条理清晰，必要时分点列出。
5. 在回答末尾用 [来源1] [来源2] 标注引用了哪些资料编号。
"""

PRECISE_EXTRA = """【精准检索模式】优先逐字引用指南/规范原文，给出明确出处；不要做跨文档推断。
"""

COMPREHENSIVE_EXTRA = """【综合推理模式】请综合多份资料交叉分析，可给出鉴别诊断思路与推理过程，
但每一步推断都要标明依据的资料编号；结论需区分"证据支持"与"临床判断"。
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "{mode_instruction}" + "{tuning_hint}"),
        (
            "human",
            """{history_block}
【参考资料】
{context}

【问题】
{question}

请基于参考资料回答问题：""",
        ),
    ]
)

# 旧版简单模板，保留兼容
BASIC_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """【参考资料】
{context}

【问题】
{question}

请基于参考资料回答问题：""",
        ),
    ]
)

# 纯 LLM 模式（RAG 关闭，功能五 A/B 对比用）：不注入任何检索证据，
# 明确要求模型在缺少权威依据时不要编造，便于与 RAG 模式做对照测试。
PURE_LLM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名医疗临床教育助手。当前处于【纯 LLM 模式】，没有检索任何知识库资料。"
            "请仅依据通用医学知识谨慎作答；当缺乏可靠依据或涉及具体剂量/禁忌/急救处置时，"
            "明确说明无法给出确定结论并建议咨询执业医师。禁止虚构文献、指南出处或页码。"
            "用简体中文回答，条理清晰。\n" + "{tuning_hint}",
        ),
        ("human", "【问题】\n{question}\n\n请回答："),
    ]
)


CONTEXT_TEMPLATE = PromptTemplate.from_template(
    """[{index}] 来源: {source}
{content}
"""
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "将用户的后续问题改写为一个可独立检索的完整问题。若已是独立问题，原样输出。只输出改写后的问题。",
        ),
        ("human", "对话历史:\n{history}\n\n后续问题: {question}"),
    ]
)


def format_context(documents) -> str:
    """把 Document 列表格式化为带编号的上下文字符串（含章节/页码）。"""
    parts = []
    for i, doc in enumerate(documents, start=1):
        md = doc.metadata
        source = md.get("doc_title") or md.get("filename") or md.get("source") or "未知来源"
        section = md.get("section", "")
        page = md.get("page", -1)
        if section:
            source += f" / {section}"
        if page and page > 0:
            source += f" / 第{page}页"
        parts.append(
            CONTEXT_TEMPLATE.format(index=i, source=source, content=doc.page_content.strip())
        )
    return "\n---\n".join(parts)
