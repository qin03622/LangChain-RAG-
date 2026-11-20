"""双角色问答口径 + 幻觉抑制提示词（Feature 3.2 / 3.5）。

- 医护端（clinician）：偏专业术语、指南原文、鉴别思路；
- 患者端（patient）：通俗科普、严格限制、不做诊断、引导线下就医。

两套口径共享同一条"幻觉抑制"铁律：知识库没有的内容直接说暂无权威资料，
禁止编造；回答必须带 [来源N] 引用，引用编号必须存在于给定上下文中。
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ---------- 幻觉抑制铁律（两套口径共享） ----------
_ANTI_HALLUCINATION = """\
【证据铁律 · 必须遵守】
1. 只能依据下面【参考资料】中给出的内容回答。参考资料未涵盖的内容，直接回答"暂无相关权威资料，建议咨询执业医师"，禁止凭记忆或常识编造任何医学内容、数字、剂量或结论。
2. 不得虚构文献、指南名称、页码或 [来源N] 编号；引用编号必须在参考资料中真实存在。
3. 区分"资料原文"与"你的推断"：逐字引用用引号标注；推断必须写明"基于资料推断"，且不得升级为确诊结论。
4. 若资料之间互相矛盾，如实指出矛盾，不要擅自择一。
5. 不做确诊、不开处方、不推荐具体用药剂量；涉及用药与急症，一律提示线下就医。
"""

# ---------- 医护端口径 ----------
_CLINICIAN_ROLE = """\
你是一名面向临床医护的专业助手，回答用于临床教育与病例讨论。
口径要求：
- 使用规范医学术语，优先逐字引用指南/共识原文并标注出处与版本；
- 可提供鉴别诊断思路、检查选择逻辑，但每一步须标明依据的资料编号，且明确标注为"思路"而非确诊；
- 结论需区分"证据支持"与"临床判断"，不替代主治医师决策。
"""

# ---------- 患者端口径 ----------
_PATIENT_ROLE = """\
你是一名面向患者的健康科普助手，回答必须用通俗易懂的语言。
口径要求：
- 避免专业术语；必须用术语时，先用大白话解释；
- 只做健康科普与就医引导，绝不做诊断、不下结论、不给用药剂量、不推荐自行购药；
- 涉及症状、用药、检查结果时，明确建议"请到正规医疗机构由医生面诊判断"；
- 语气温暖、耐心，但不给出任何可能被理解为治疗方案的指令。
"""


def build_role_prompt(role: str = "patient", with_history: bool = True) -> ChatPromptTemplate:
    """按角色构造问答 Prompt。

    Args:
        role: ``clinician`` 或 ``patient``；未知值按患者端处理（更安全）。
        with_history: 是否在 human 消息中包含对话历史块。

    Returns:
        可直接用于 LCEL 的 :class:`ChatPromptTemplate`，
        变量包含 ``context`` / ``question`` / ``disclaimer``，
        ``with_history=True`` 时额外包含 ``history_block``。
    """
    role_block = _CLINICIAN_ROLE if role == "clinician" else _PATIENT_ROLE
    history_placeholder = "{history_block}\n" if with_history else ""

    return ChatPromptTemplate.from_messages(
        [
            ("system", role_block + "\n" + _ANTI_HALLUCINATION + "\n{disclaimer}"),
            (
                "human",
                history_placeholder
                + """【参考资料】
{context}

【问题】
{question}

请严格依据参考资料，按上述口径回答问题：""",
            ),
        ]
    )


# 预设实例，便于直接引用
CLINICIAN_QA_PROMPT = build_role_prompt("clinician", with_history=True)
PATIENT_QA_PROMPT = build_role_prompt("patient", with_history=True)


__all__ = [
    "build_role_prompt",
    "CLINICIAN_QA_PROMPT",
    "PATIENT_QA_PROMPT",
]
