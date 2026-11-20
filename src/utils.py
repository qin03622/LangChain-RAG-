"""通用工具：日志、文本清洗、安全护栏。"""
from __future__ import annotations

import logging
import re
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """获取统一格式的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/emoji 乱码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def clean_text(text: str) -> str:
    """规范空白字符。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 常见需要拒绝的高危场景关键词（仅用于提示风险，不做诊断）
_HIGH_RISK_PATTERNS = [
    r"自行(停[用药]|换药|加量|减量|调整剂量)",
    r"处方药.{0,6}(剂量|怎么吃|怎么服用)",
    r"(自杀|自残|伤害自己)",
    r"拒绝(就医|治疗)",
]


def safety_check(question: str, answer: str) -> tuple[bool, str]:
    """简单的安全护栏：检测高危意图并追加提醒。

    返回 (是否通过, 附加提示)。不完全拦截，而是给出风险提示，
    避免假阴性导致用户得不到任何帮助。
    """
    combined = question + " " + answer
    hits = [p for p in _HIGH_RISK_PATTERNS if re.search(p, combined)]
    if not hits:
        return True, ""
    return False, "⚠️ 检测到高风险内容：请立即联系专业医生或拨打急救电话，切勿依据 AI 回答自行处理。"
