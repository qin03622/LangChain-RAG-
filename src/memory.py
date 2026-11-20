"""多轮对话记忆：会话级上下文、TTL 过期、摘要压缩、敏感会话自动清理。

兼容 LangChain 的 ConversationBufferMemory / ConversationSummaryMemory 思路，
用自实现存储以便支持 TTL 与敏感清理，并可选择持久化到磁盘。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)

# 敏感信息指示词：命中后在会话结束时自动清理（不在日志/落盘中留存原文）
_SENSITIVE_PATTERNS = [
    r"身份证", r"手机号", r"住址", r"病历号", r"医保卡",
    r"真实姓名", r"病人姓名", r"出生日期", r"电话号码", r"\d{11}",
]


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sensitive: bool = False

    def is_expired(self, ttl_seconds: int) -> bool:
        return time.time() - self.updated_at > ttl_seconds

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionMemory:
    """带 TTL 与敏感清理的会话记忆。"""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None):
        self.settings = settings
        self.llm = llm
        self.ttl = settings.memory_ttl_seconds
        self.max_turns = settings.memory_max_turns
        self.enabled = settings.memory_enabled
        self.auto_purge = settings.memory_auto_purge_sensitive
        self.store_path: Path = settings.session_store_path
        self._sessions: dict[str, Session] = {}
        if self.enabled and self.store_path:
            self.store_path.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---------- 会话生命周期 ----------
    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(session_id=sid)
        return sid

    def get(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s and s.is_expired(self.ttl):
            logger.info("会话 %s 已过期，自动清理", session_id)
            self.clear(session_id)
            return None
        return s

    def touch(self, session_id: str) -> None:
        s = self.get(session_id)
        if s:
            s.updated_at = time.time()

    def clear(self, session_id: str) -> None:
        """删除会话（敏感会话直接丢弃，不落盘）。"""
        self._sessions.pop(session_id, None)
        self._delete_from_disk(session_id)

    def list_sessions(self, include_expired: bool = False) -> list[dict]:
        """(功能五) 列出会话元数据，供运维后台查看/管理历史对话。

        不返回完整消息正文，仅返回摘要预览，避免敏感内容在管理端裸露。
        """
        items: list[dict] = []
        for s in list(self._sessions.values()):
            expired = s.is_expired(self.ttl)
            if expired and not include_expired:
                continue
            items.append(
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "message_count": len(s.messages),
                    "turn_count": len(s.messages) // 2,
                    "sensitive": s.sensitive,
                    "summary": s.summary[:120] if s.summary else "",
                    "expired": expired,
                }
            )
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items

    # ---------- 读写 ----------
    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        if not self.enabled:
            return
        s = self.get(session_id) or Session(session_id=session_id)
        if self.auto_purge and any(re.search(p, question) for p in _SENSITIVE_PATTERNS):
            s.sensitive = True

        s.messages.append({"role": "user", "content": question})
        s.messages.append({"role": "assistant", "content": answer})
        s.updated_at = time.time()

        # 超过轮数阈值则压缩为摘要（ConversationSummaryMemory 思路）
        if len(s.messages) > self.max_turns * 2:
            self._summarize(s)

        self._sessions[session_id] = s
        if not s.sensitive:
            self._save(s)

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        s = self.get(session_id)
        if not s:
            return []
        history: list[dict[str, str]] = []
        if s.summary:
            history.append({"role": "system", "content": f"此前对话摘要：{s.summary}"})
        history.extend(s.messages)
        return history

    def get_context_text(self, session_id: str) -> str:
        """把历史拼成给 LLM 的对话上下文文本。"""
        if not self.enabled:
            return ""
        lines = []
        for m in self.get_history(session_id):
            role = "医生/用户" if m["role"] == "user" else "助手"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    # ---------- 摘要压缩 ----------
    def _summarize(self, session: Session) -> None:
        to_summarize = session.messages[: -4]  # 保留最近 2 轮原文
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
        if self.llm is not None:
            try:
                prompt = (
                    "把以下医患对话压缩成一段简短摘要，保留关键临床背景（年龄、病史、用药等），"
                    f"用于后续问答。只输出摘要：\n{text}"
                )
                session.summary = str(self.llm.invoke(prompt).content).strip()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("摘要生成失败(%s)，使用截断摘要。", exc)
                session.summary = text[:500]
        else:
            session.summary = text[:500]
        session.messages = session.messages[-4:]

    # ---------- 持久化 ----------
    def _path(self, session_id: str) -> Path:
        return self.store_path / f"{session_id}.json"

    def _save(self, session: Session) -> None:
        if not self.store_path:
            return
        try:
            self._path(session.session_id).write_text(
                json.dumps(session.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话落盘失败: %s", exc)

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        for f in self.store_path.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                s = Session(**data)
                if not s.is_expired(self.ttl):
                    self._sessions[s.session_id] = s
                else:
                    f.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("会话加载失败 %s: %s", f, exc)

    def _delete_from_disk(self, session_id: str) -> None:
        if self.store_path:
            self._path(session_id).unlink(missing_ok=True)

    def purge_expired(self) -> int:
        """清理所有过期会话，返回清理数量。"""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(self.ttl)]
        for sid in expired:
            self.clear(sid)
        return len(expired)
