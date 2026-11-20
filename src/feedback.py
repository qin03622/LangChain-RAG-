"""问答反馈机制：记录 正确 / 幻觉 / 过时 / 不完整，并用于检索调优。

反馈以 JSONL 持久化；统计每个 chunk 的负面反馈率，
检索时按质量分对文档加权，实现“反馈回传 → 检索调优”。
"""
from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"correct", "hallucination", "outdated", "incomplete"}
NEGATIVE_LABELS = {"hallucination", "outdated", "incomplete"}


@dataclass
class Feedback:
    id: str
    question: str
    answer: str
    label: str
    chunk_ids: list[str] = field(default_factory=list)
    mode: str = ""
    session_id: str = ""
    note: str = ""
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """反馈的读写与统计。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.feedback_enabled
        self.tuning_enabled = settings.feedback_tuning_enabled
        self.path: Path = settings.feedback_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        question: str,
        answer: str,
        label: str,
        chunk_ids: list[str] | None = None,
        mode: str = "",
        session_id: str = "",
        note: str = "",
    ) -> Feedback | None:
        if not self.enabled:
            return None
        if label not in VALID_LABELS:
            raise ValueError(f"未知反馈标签: {label}，合法值: {VALID_LABELS}")
        fb = Feedback(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            label=label,
            chunk_ids=chunk_ids or [],
            mode=mode,
            session_id=session_id,
            note=note,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fb.as_dict(), ensure_ascii=False) + "\n")
        logger.info("记录反馈 %s: %s", label, fb.id)
        return fb

    def all(self) -> list[Feedback]:
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Feedback(**json.loads(line)))
            except Exception:  # noqa: BLE001
                continue
        return out

    def stats(self) -> dict[str, Any]:
        fbs = self.all()
        labels = Counter(f.label for f in fbs)
        return {
            "total": len(fbs),
            "by_label": dict(labels),
            "negative_rate": (
                sum(labels[l] for l in NEGATIVE_LABELS) / len(fbs) if fbs else 0.0
            ),
        }

    def chunk_quality(self) -> dict[str, float]:
        """计算每个 chunk 的质量分（1.0 = 无负面反馈）。"""
        pos: Counter[str] = Counter()
        neg: Counter[str] = Counter()
        for fb in self.all():
            for cid in fb.chunk_ids:
                if fb.label in NEGATIVE_LABELS:
                    neg[cid] += 1
                else:
                    pos[cid] += 1
        quality: dict[str, float] = {}
        for cid in set(pos) | set(neg):
            p, n = pos[cid], neg[cid]
            quality[cid] = (p + 1.0) / (p + n + 2.0)  # 贝叶斯平滑
        return quality

    def rerank_by_quality(self, docs: Iterable[Document]) -> list[Document]:
        """按反馈质量分重排文档（反馈调优入口）。"""
        if not self.tuning_enabled:
            return list(docs)
        quality = self.chunk_quality()
        if not quality:
            return list(docs)
        return sorted(
            docs,
            key=lambda d: quality.get(d.metadata.get("chunk_id", ""), 1.0),
            reverse=True,
        )

    def tuning_hint(self) -> str:
        """生成给提示词的优化提示（例如幻觉率高时强调严格引用）。"""
        s = self.stats()
        if s["total"] == 0:
            return ""
        hints = []
        by_label = s["by_label"]
        if by_label.get("hallucination", 0) > 0:
            hints.append("历史反馈显示曾出现幻觉，请严格逐字依据资料作答，不要推断未记载的内容。")
        if by_label.get("outdated", 0) > 0:
            hints.append("历史反馈提示部分内容可能过时，请注明资料版本/日期并建议核对最新指南。")
        if by_label.get("incomplete", 0) > 0:
            hints.append("历史反馈提示回答不够完整，请尽量覆盖禁忌症、并发症与随访要点。")
        return " ".join(hints)
