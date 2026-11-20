"""审计留痕（Feature 3.3）：全量记录问答，满足医疗审计与溯源要求。

每条审计记录包含：
    - 提问内容（脱敏后）
    - 回答内容（脱敏后）
    - 提问人 / 角色（user_id, role）
    - 时间（UTC ISO-8601 + Unix 时间戳）
    - 召回文档（chunk_id / 文档名 / 章节 / 页码）
    - 安全过滤结果（拦截类别 / 风险提示）

存储：JSONL 追加写（只增不改），按 session_id 可选关联多轮。
"双保险"：即便上层忘记脱敏，审计模块写入前会再次调用 :func:`desensitize`
对问题和答案做脱敏，确保原始隐私绝不落盘。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .privacy import desensitize
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class AuditRecord:
    """单条问答审计记录。"""

    audit_id: str
    timestamp: str
    timestamp_epoch: float
    question: str
    answer: str
    user_id: str
    role: str
    session_id: str = ""
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    filter_result: dict[str, Any] = field(default_factory=dict)
    privacy_hits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> tuple[str, float]:
    epoch = time.time()
    iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    return iso, epoch


class AuditLogger:
    """基于 JSONL 的只追加审计日志。"""

    def __init__(self, settings: Settings, store_path: str | Path | None = None):
        self.settings = settings
        p = Path(store_path) if store_path else Path(settings.audit_store_path)
        self.path = p if p.is_absolute() else Path(settings.audit_store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 写入 ----------
    def record(
        self,
        question: str,
        answer: str,
        user_id: str = "anonymous",
        role: str = "patient",
        session_id: str = "",
        retrieved_documents: Iterable[dict[str, Any]] | None = None,
        filter_result: dict[str, Any] | None = None,
        desensitize_text: bool = True,
    ) -> AuditRecord:
        """追加一条审计记录，返回写入的记录。

        Args:
            question: 原始提问（会被再次脱敏）。
            answer: 原始回答（会被再次脱敏）。
            user_id: 提问人标识。
            role: 提问人角色（patient / clinician）。
            session_id: 多轮会话 ID。
            retrieved_documents: 召回文档元数据列表。
            filter_result: 内容过滤结果（FilterResult.as_dict()）。
            desensitize_text: 是否再次脱敏（默认 True，双保险）。
        """
        privacy_hits: list[str] = []
        if desensitize_text:
            q = desensitize(question or "")
            a = desensitize(answer or "")
            privacy_hits = sorted(set(q.hits + a.hits))
            question, answer = q.text, a.text

        iso, epoch = _utc_now()
        rec = AuditRecord(
            audit_id=str(uuid.uuid4()),
            timestamp=iso,
            timestamp_epoch=epoch,
            question=question,
            answer=answer,
            user_id=user_id or "anonymous",
            role=role,
            session_id=session_id,
            retrieved_documents=list(retrieved_documents or []),
            filter_result=dict(filter_result or {}),
            privacy_hits=privacy_hits,
        )
        self._append(rec)
        return rec

    def _append(self, rec: AuditRecord) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - 审计失败不能中断问答主流程
            logger.warning("审计日志写入失败: %s", exc)

    # ---------- 查询 ----------
    def entries(
        self,
        user_id: str | None = None,
        role: str | None = None,
        session_id: str | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按条件查询审计记录（返回最新的 limit 条）。"""
        results: list[dict[str, Any]] = []
        if not self.path.exists():
            return results
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if user_id is not None and rec.get("user_id") != user_id:
                    continue
                if role is not None and rec.get("role") != role:
                    continue
                if session_id is not None and rec.get("session_id") != session_id:
                    continue
                ts = rec.get("timestamp_epoch", 0)
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    continue
                if keyword and keyword not in rec.get("question", "") and keyword not in rec.get("answer", ""):
                    continue
                results.append(rec)
        return results[-limit:]

    def stats(self) -> dict[str, Any]:
        """审计统计：总数、按角色/拦截类别分布。"""
        total = 0
        by_role: dict[str, int] = {}
        by_category: dict[str, int] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    by_role[rec.get("role", "unknown")] = by_role.get(rec.get("role", "unknown"), 0) + 1
                    fr = rec.get("filter_result") or {}
                    if fr.get("blocked"):
                        cat = fr.get("category", "unknown") or "unknown"
                        by_category[cat] = by_category.get(cat, 0) + 1
        return {"total": total, "by_role": by_role, "blocked_by_category": by_category}


__all__ = ["AuditRecord", "AuditLogger"]
