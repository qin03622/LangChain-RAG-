"""隐私保护：前置脱敏预处理链（Feature 3.4）。

设计目标：
- 自动识别并脱敏患者隐私信息：姓名、手机号、住址、身份证号、病历号等；
- 不自动"采集"患者身份信息——脱敏在入检索/入模型/入日志之前完成；
- 命中即替换为掩码（默认 ``*``），同时返回命中类型清单，便于审计统计。

脱敏分两类规则：
1. 高置信度结构化规则（身份证 / 手机号 / 邮箱 / 银行卡 / 地址 / 日期身份证等）：
   正则直接替换，误伤率低；
2. 上下文姓名规则（如"姓名：张三""患者李四""我叫王五"）：
   依靠称谓词 + 中文姓名模式定位，避免把"高血压"之类的普通词当作人名。

对外主入口 :func:`desensitize` 返回 :class:`DesensitizeResult`，
其中 ``text`` 为脱敏后文本，``hits`` 为命中类型列表，``details`` 为
``(类型, 原文片段, 掩码)`` 明细（供本地审计，不写日志原文）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda


@dataclass
class DesensitizeResult:
    """脱敏结果。"""

    text: str
    hits: list[str] = field(default_factory=list)
    details: list[tuple[str, str, str]] = field(default_factory=list)
    changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "hits": self.hits,
            "changed": self.changed,
            # details 含隐私原文片段，默认只在本地对象保留，不写入日志/API
            "mask_count": len(self.details),
        }


# ---------- 结构化正则规则（顺序：先长后短，先具体后泛化） ----------
# 18 位身份证（含末位 X），支持 15 位旧证
_ID_CARD = r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)"
# 中国大陆手机号：1[3-9] 开头 11 位
_PHONE = r"(?<!\d)1[3-9]\d{9}(?!\d)"
# 邮箱
_EMAIL = r"[\w.+-]+@[\w-]+\.[\w.-]+"
# 银行卡 / 社保卡号（16-19 位连续数字；在手机号、身份证之后匹配）
_BANK_CARD = r"(?<!\d)\d{16,19}(?!\d)"
# 详细地址：省/市/区 + 路/街/号/室 等门牌信息
_ADDRESS = (
    r"[\u4e00-\u9fa5]{2,8}(?:省|自治区|市|县|区)"
    r"[\u4e00-\u9fa5\d]{0,20}?(?:路|街|巷|村|小区|花园|大厦)"
    r"[\u4e00-\u9fa5\d]{0,20}?(?:号|栋|单元|室|楼层)?"
)
# 病历号 / 住院号 / 门诊号（中文标签 + 字母数字）
_MEDICAL_ID = r"(?:病历号|住院号|门诊号|就诊卡号|医保卡号)\s*[:：]?\s*[A-Za-z0-9\-]{4,20}"
# 姓名上下文：称谓词 + 2~4 位中文姓名
_NAME_CONTEXT = (
    r"(?:患者|病人|姓名|患者姓名|病人姓名|我叫|本人| called\s*name)"
    r"\s*[:：为是]?\s*([\u4e00-\u9fa5]{2,4})(?![\u4e00-\u9fa5])"
)
# 日期型出生日期（上下文）
_BIRTH_DATE = r"(?:出生日期|生日)\s*[:：]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"

_STRUCTURED_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("身份证号", re.compile(_ID_CARD), "*" * 18),
    ("手机号", re.compile(_PHONE), "***********"),
    ("邮箱", re.compile(_EMAIL), "***@***.***"),
    ("银行卡号", re.compile(_BANK_CARD), "****************"),
    ("住址", re.compile(_ADDRESS), "********"),
    ("病历号", re.compile(_MEDICAL_ID), "********"),
    ("出生日期", re.compile(_BIRTH_DATE), "****-**-**"),
]

# 姓名出现在称谓/句读之后，要求首字为常见姓氏，降低"高血压"等医学词误伤
_COMMON_SURNAMES = (
    "王李张刘陈杨黄赵周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾"
    "肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏"
    "韦付方白邹孟熊秦邱江尹薛阎段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴"
    "莫孔向汤"
)
_NAME_BARE = (
    r"(?:^|[，,。;\s])([" + _COMMON_SURNAMES + r"][\u4e00-\u9fa5]{1,3})"
    r"(?=(?:请|联系|您好|医生|大夫|就诊|看病|随诊|，|,))"
)

_CONTEXT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("姓名", re.compile(_NAME_CONTEXT)),
    ("姓名", re.compile(_NAME_BARE)),
]


def desensitize(
    text: str,
    mask: str = "*",
    extra_patterns: list[tuple[str, str]] | None = None,
) -> DesensitizeResult:
    """对文本执行脱敏，返回 :class:`DesensitizeResult`。

    Args:
        text: 待脱敏的原始文本（用户提问 / 粘贴的病历）。
        mask: 用于替换命名实体（姓名/地址）的掩码字符。
        extra_patterns: 额外的 ``(标签, 正则)`` 规则，便于业务侧扩展。

    Notes:
        结构化数字类（身份证/手机号）采用定长掩码，避免暴露尾号；
        地址/姓名等非定长实体用 ``mask * len(matched)`` 等长替换，保持可读性。
    """
    if not text:
        return DesensitizeResult(text=text or "")

    hits: list[str] = []
    details: list[tuple[str, str, str]] = []
    out = text

    def _replace(label: str, matched: str, replacement: str) -> str:
        if matched:
            hits.append(label)
            details.append((label, matched, replacement))
        return replacement

    # 1) 上下文规则（姓名 / 出生日期）——先处理，保留"姓名："标签
    for label, pattern in _CONTEXT_RULES:
        def _ctx_sub(m: re.Match[str], _label=label, _mask=mask) -> str:
            captured = m.group(1)
            repl = _mask * len(captured)
            hits.append(_label)
            details.append((_label, captured, repl))
            return m.group(0).replace(captured, repl, 1)

        out = pattern.sub(_ctx_sub, out)

    # 2) 结构化规则
    for label, pattern, default_repl in _STRUCTURED_RULES:
        def _struct_sub(m: re.Match[str], _label=label, _default=default_repl, _mask=mask) -> str:
            matched = m.group(0)
            # 带捕获组的规则（出生日期）替换整个匹配
            replacement = _default if _default else _mask * len(matched)
            return _replace(_label, matched, replacement)

        out = pattern.sub(_struct_sub, out)

    # 3) 扩展规则
    if extra_patterns:
        for label, pattern in extra_patterns:
            compiled = re.compile(pattern)

            def _extra_sub(m: re.Match[str], _label=label, _mask=mask) -> str:
                matched = m.group(0)
                return _replace(_label, matched, _mask * len(matched))

            out = compiled.sub(_extra_sub, out)

    return DesensitizeResult(
        text=out,
        hits=hits,
        details=details,
        changed=(out != text),
    )


def contains_sensitive(text: str) -> bool:
    """快速判断文本是否含敏感信息（不返回脱敏文本）。"""
    return desensitize(text).changed


def build_desensitize_chain(
    mask: str = "*",
    extra_patterns: list[tuple[str, str]] | None = None,
) -> Runnable:
    """构造 LCEL 前置脱敏链：输入字符串 → 输出脱敏后字符串。

    用法::

        chain = build_desensitize_chain() | retriever
        chain.invoke("我叫张三，身份证 110101199001011234，高血压怎么治？")

    若需要保留命中类型，可直接使用 :func:`desensitize` 函数。
    """
    return RunnableLambda(lambda text: desensitize(text, mask=mask, extra_patterns=extra_patterns).text)


__all__ = [
    "DesensitizeResult",
    "desensitize",
    "contains_sensitive",
    "build_desensitize_chain",
]
