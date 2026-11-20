"""临床问答链：整合 HyDE/MultiQuery/Rerank、多轮记忆、引用溯源、反馈调优。"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .citation import Citation, build_citations, render_citations_markdown
from .clinical_retriever import ClinicalRetriever
from .config import Settings
from .feedback import FeedbackStore
from .memory import SessionMemory
from .prompts import COMPREHENSIVE_EXTRA, PRECISE_EXTRA, QA_PROMPT
from .utils import safety_check
from .vectorstore import VectorStoreManager


class ClinicalQAService:
    """临床问答服务：把各增强模块组合成统一调用入口。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        vsm: VectorStoreManager,
        memory: SessionMemory,
        feedback: FeedbackStore,
    ):
        self.settings = settings
        self.llm = llm
        self.vsm = vsm
        self.memory = memory
        self.feedback = feedback
        self.retriever = ClinicalRetriever(settings, llm, vsm)
        self.chain = QA_PROMPT | llm | StrOutputParser()

    def query(
        self,
        question: str,
        mode: str | None = None,
        session_id: str | None = None,
        use_history: bool = True,
    ) -> dict[str, Any]:
        mode = (mode or self.settings.clinical_mode).lower()
        session_id = session_id or (self.memory.new_session() if self.settings.memory_enabled else "")

        # 1) 检索（含 HyDE / MultiQuery / Rerank）
        docs: list[Document] = self.retriever.retrieve(question, mode=mode)

        # 2) 反馈调优：按历史反馈质量重排
        docs = self.feedback.rerank_by_quality(docs)

        # 3) 构造引用与上下文
        citations: list[Citation] = build_citations(docs)
        from .citation import citations_to_context
        context = citations_to_context(citations)

        # 4) 多轮历史
        history_text = self.memory.get_context_text(session_id) if (use_history and session_id) else ""
        history_block = f"【对话历史（病例背景）】\n{history_text}\n" if history_text else ""

        # 5) Prompt 组装
        mode_instruction = COMPREHENSIVE_EXTRA if mode == "comprehensive" else PRECISE_EXTRA
        tuning_hint = self.feedback.tuning_hint() if self.settings.feedback_tuning_enabled else ""

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
                "history_block": history_block,
                "mode_instruction": mode_instruction,
                "tuning_hint": tuning_hint,
            }
        )

        # 6) 安全护栏
        passed, warning = (True, "")
        if self.settings.enable_safety_guard:
            passed, warning = safety_check(question, str(answer))

        # 7) 写入会话记忆
        if session_id:
            self.memory.add_turn(session_id, question, str(answer))

        return {
            "answer": str(answer),
            "mode": mode,
            "session_id": session_id,
            "source_documents": docs,
            "citations": [c.as_dict() for c in citations],
            "citations_markdown": render_citations_markdown(citations),
            "safety_passed": passed,
            "safety_warning": warning,
            "disclaimer": self.settings.disclaimer,
        }

    def submit_feedback(
        self,
        question: str,
        answer: str,
        label: str,
        result: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        chunk_ids = [c["chunk_id"] for c in result.get("citations", [])] if result else []
        fb = self.feedback.add(
            question=question,
            answer=answer,
            label=label,
            chunk_ids=chunk_ids,
            mode=result.get("mode", "") if result else "",
            session_id=result.get("session_id", "") if result else "",
            note=note,
        )
        return fb.as_dict() if fb else {"status": "disabled"}

    def get_original(self, result: dict[str, Any], chunk_id: str) -> dict[str, Any] | None:
        """按 chunk_id 跳查看完整原文片段。"""
        from .citation import get_original_fragment

        return get_original_fragment(result.get("source_documents", []), chunk_id)

    def format_result(self, result: dict[str, Any]) -> str:
        """输出带引用溯源的完整文本。"""
        lines = [
            "=" * 60,
            f"【回答 · {result['mode']}模式】",
            result["answer"],
            "",
        ]
        if result.get("safety_warning"):
            lines += [result["safety_warning"], ""]
        lines.append("【引用来源（文档 / 章节 / 页码）】")
        for c in result.get("citations", []):
            loc = []
            if c.get("section"):
                loc.append(f"章节: {c['section']}")
            if c.get("page", -1) > 0:
                loc.append(f"第{c['page']}页")
            loc_str = "，".join(loc) if loc else c.get("source", "")
            lines.append(f"  [{c['index']}] {c['doc_title']}（{loc_str}） chunk={c['chunk_id']}")
        lines += ["", result.get("disclaimer", ""), "=" * 60]
        return "\n".join(lines)
