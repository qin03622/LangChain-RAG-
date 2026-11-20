"""前置内容过滤 + 医疗风控（Feature 3.1）。

在问题进入检索/LLM 之前做意图分类与违规拦截：
- 拒绝开处方、拒绝给出确诊结论、拒绝推荐剂量用于自行购药（患者端严格）；
- 拒绝自杀、自残、堕胎、违禁药等高危问题；
- 医护端允许专业术语与鉴别思路，但仍不下确诊结论、不开处方。

分类基于"关键词 + 语义模式"的确定性规则（无额外依赖、可离线测试），
高风险类别直接拦截并返回统一的安全回应；低风险命中仅附加风险提示。

LangChain 集成：:func:`build_content_filter_chain` 返回一个 Runnable，
输入 ``str`` 问题，输出 :class:`FilterResult`；违规时抛 :class:`ContentBlockedError`
（也可通过 ``raise_on_block=False`` 返回带 ``blocked=True`` 的结果，由上层决定）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda


class ContentBlockedError(Exception):
    """违规内容被前置拦截。"""

    def __init__(self, result: "FilterResult"):
        self.result = result
        super().__init__(result.reason or "内容已被安全策略拦截")


@dataclass
class FilterResult:
    """内容过滤结果。"""

    allowed: bool = True
    blocked: bool = False
    category: str = ""                 # 命中的违规类别
    matched: list[str] = field(default_factory=list)  # 命中的关键词/模式
    reason: str = ""                   # 拦截原因（合规话术）
    severity: str = "none"             # none | warn | block
    risk_hint: str = ""                # 非拦截类的风险提示

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked": self.blocked,
            "category": self.category,
            "matched": self.matched,
            "reason": self.reason,
            "severity": self.severity,
            "risk_hint": self.risk_hint,
        }


# ---------- 违规类别规则 ----------
# 高危：一律拦截，不区分角色
_BLOCK_RULES: dict[str, list[str]] = {
    "自杀/自残": [
        r"自杀", r"自残", r"伤害自己", r"结束生命", r"不想活", r"跳楼", r"割腕", r"上吊",
        r"安眠药.{0,8}(自杀|过量)", r"怎么(死|弄死自己)",
    ],
    "堕胎/终止妊娠": [
        r"(自行|自己|在家).{0,6}(堕胎|流产|打胎|药流)",
        r"(药流|米非司酮|米索前列醇).{0,10}(怎么吃|剂量|购买|自行)",
        r"堕胎药.{0,8}(哪|买|剂量|怎么)",
    ],
    "违禁药/毒品": [
        r"(海洛因|冰毒|摇头丸|大麻|可卡因|鸦片|吗啡).{0,10}(怎么|购买|剂量|吸|注射)",
        r"(制毒|吸毒|毒品).{0,8}(方法|怎么|教程|购买)",
    ],
}

# 患者端严格拦截：开处方 / 确诊结论 / 自行用药剂量
_PATIENT_BLOCK_RULES: dict[str, list[str]] = {
    "处方/自行购药": [
        r"(帮我|给我|请).{0,4}(开|写).{0,4}处方",
        r"处方药.{0,10}(剂量|怎么吃|怎么服用|买|购买)",
        r"(自行|自己|擅自).{0,6}(购药|买药|加量|减量|调整剂量|停药|换药)",
        r"(头孢|抗生素|激素|降压药|降糖药|二甲双胍|阿莫西林|布洛芬).{0,10}(剂量|怎么吃|吃几片|一天几次|几片)",
        r"一天.{0,4}(吃|服用).{0,4}(几片|多少|几粒)",
    ],
    "确诊结论": [
        r"(帮我|给我|请).{0,4}(确诊|诊断).{0,6}(是什么|是不是|结果)",
        r"我是不是(得|患).{0,8}(癌|瘤|糖尿病|高血压|心脏病)",
        r"(确诊|诊断)结果.{0,6}(是什么|帮我看)",
    ],
}

# 医护端警示（不拦截）：可作为鉴别思路，但须提示不能替代确诊
_WARN_RULES: dict[str, list[str]] = {
    "用药剂量": [r"剂量", r"用法用量", r"几片", r"每日几次"],
    "确诊式表述": [r"确诊", r"诊断标准", r"是不是.{0,6}症"],
}


_BLOCK_REASON = {
    "自杀/自残": "检测到涉及自伤/自杀的高危内容。请立即联系身边亲友或拨打全国心理援助热线（如 12356），或直接前往急诊。本系统不能提供此类帮助。",
    "堕胎/终止妊娠": "涉及终止妊娠的用药与操作属于高危医疗行为，必须线下由正规医疗机构评估，本系统不予提供相关指导。",
    "违禁药/毒品": "涉及违禁药品/毒品的内容已被拒绝，本系统不提供任何相关信息。",
    "处方/自行购药": "本系统不能开具处方，也不提供自行购药、自行调整剂量的建议。请携带检查结果到正规医疗机构，由执业医师诊断并开具处方。",
    "确诊结论": "本系统不能给出确诊结论。是否有疾病需由执业医师结合检查当面诊断，请线下就医。",
}


def _match_any(text: str, patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            hits.append(m.group(0))
    return hits


def classify_intent(
    question: str,
    role: str = "patient",
) -> FilterResult:
    """对问题做意图分类与违规判定。

    Args:
        question: 用户问题原文（建议先脱敏）。
        role: ``patient``（患者端，严格）或 ``clinician``（医护端，专业口径）。

    Returns:
        :class:`FilterResult`。``blocked=True`` 时应短路，不再进入检索/LLM。
    """
    text = (question or "").strip()
    if not text:
        return FilterResult(allowed=False, blocked=True, category="空问题", reason="问题不能为空。", severity="block")

    # 1) 所有角色一律拦截的高危类别
    for category, patterns in _BLOCK_RULES.items():
        hits = _match_any(text, patterns)
        if hits:
            return FilterResult(
                allowed=False,
                blocked=True,
                category=category,
                matched=hits,
                reason=_BLOCK_REASON[category],
                severity="block",
            )

    # 2) 患者端严格拦截
    if role == "patient":
        for category, patterns in _PATIENT_BLOCK_RULES.items():
            hits = _match_any(text, patterns)
            if hits:
                return FilterResult(
                    allowed=False,
                    blocked=True,
                    category=category,
                    matched=hits,
                    reason=_BLOCK_REASON[category],
                    severity="block",
                )

    # 3) 医护端 / 低危：附加警示，不拦截
    hints: list[str] = []
    categories: list[str] = []
    for category, patterns in _WARN_RULES.items():
        if _match_any(text, patterns):
            categories.append(category)
    if categories:
        hints.append("涉及用药或诊断表述时，仅作鉴别思路参考，不能替代执业医师的确诊与处方。")
    if role == "patient":
        hints.append("以下内容为健康科普，不构成诊断或治疗建议，请以下线就医结论为准。")

    return FilterResult(
        allowed=True,
        blocked=False,
        category=",".join(categories) if categories else "normal",
        severity="warn" if categories else "none",
        risk_hint=" ".join(hints),
    )


def build_content_filter_chain(
    role: str = "patient",
    raise_on_block: bool = False,
) -> Runnable:
    """构造 LCEL 前置过滤链：``str`` → :class:`FilterResult`。

    ``raise_on_block=True`` 时违规直接抛 :class:`ContentBlockedError`。
    """
    def _filter(question: str) -> FilterResult:
        result = classify_intent(question, role=role)
        if result.blocked and raise_on_block:
            raise ContentBlockedError(result)
        return result

    return RunnableLambda(_filter)


__all__ = [
    "FilterResult",
    "ContentBlockedError",
    "classify_intent",
    "build_content_filter_chain",
]
