"""安全问答编排层（Feature 3 总装）：前置过滤 → 脱敏 → 检索 → 角色 Prompt → 审计。

工作流（LCEL 前置链 + 显式服务类）：
    用户问题
      → 1. 内容过滤（意图分类 / 违规拦截，按角色区分口径）
      → 2. 隐私脱敏（姓名/手机/身份证/住址等）
      → 3. 检索（复用 ClinicalRetriever）
      → 4. 角色 Prompt 生成（医护专业 / 患者科普 + 幻觉抑制）
      → 5. 安全后处理（免责声明 / 风险提示 / 确诊式表述警示）
      → 6. 审计留痕（全量记录，含召回文档）

违规问题在步骤 1 即短路，不检索、不调用 LLM，但仍写入审计（记录拦截事实）。
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .audit import AuditLogger
from .citation import build_citations, render_citations_markdown
from .clinical_retriever import ClinicalRetriever
from .config import Settings
from .content_filter import FilterResult, classify_intent
from .privacy import desensitize
from .role_prompts import build_role_prompt
from .vectorstore import VectorStoreManager


class SafetyGuardService:
    """统一安全问答服务：角色区分 + 过滤 + 脱敏 + 引用 + 审计。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        vsm: VectorStoreManager,
        audit: AuditLogger,
    ):
        self.settings = settings
        self.llm = llm
        self.vsm = vsm
        self.audit = audit
        self.retriever = ClinicalRetriever(settings, llm, vsm)
        self._prompt_cache: dict[tuple[str, bool], Any] = {}

    def _prompt(self, role: str, with_history: bool):
        key = (role, with_history)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = build_role_prompt(role, with_history=with_history)
        return self._prompt_cache[key]

    def query(
        self,
        question: str,
        role: str = "patient",
        user_id: str = "anonymous",
        session_id: str = "",
        history_text: str = "",
    ) -> dict[str, Any]:
        """执行一次安全问答，返回结果 dict（违规时直接返回拦截结果）。"""
        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        role = role if role in ("clinician", "patient") else "patient"

        # 1) 前置内容过滤（按角色）
        filter_result: FilterResult = classify_intent(question, role=role)
        if filter_result.blocked:
            # 拦截也留痕：记录拦截事实与原因（问题原文经脱敏后写入）
            dq = desensitize(question)
            self.audit.record(
                question=dq.text,
                answer=filter_result.reason,
                user_id=user_id,
                role=role,
                session_id=session_id,
                retrieved_documents=[],
                filter_result=filter_result.as_dict(),
                desensitize_text=False,  # 已脱敏
            )
            return {
                "answer": filter_result.reason,
                "blocked": True,
                "filter": filter_result.as_dict(),
                "privacy_hits": dq.hits,
                "source_documents": [],
                "citations": [],
                "citations_markdown": "",
                "safety_warning": filter_result.reason,
                "disclaimer": self.settings.disclaimer,
                "role": role,
            }

        # 2) 隐私脱敏（入检索/入模型前）
        dq = desensitize(question)
        safe_question = dq.text

        # 3) 检索
        docs: list[Document] = self.retriever.retrieve(safe_question, mode="precise")

        # 4) 引用与上下文
        citations = build_citations(docs)
        from .citation import citations_to_context
        context = citations_to_context(citations)

        # 5) 角色 Prompt 生成
        history_block = f"【对话历史】\n{history_text}\n" if history_text else ""
        chain: Runnable = self._prompt(role, True) | self.llm | StrOutputParser()
        answer = str(
            chain.invoke(
                {
                    "context": context,
                    "question": safe_question,
                    "history_block": history_block,
                    "disclaimer": self.settings.disclaimer,
                }
            )
        )

        # 6) 安全后处理：免责声明 + 风险提示 + 确诊式表述警示（回答也脱敏）
        da = desensitize(answer)
        answer = da.text
        safety_parts: list[str] = []
        if filter_result.risk_hint:
            safety_parts.append(filter_result.risk_hint)
        if self.settings.specialty_warn_diagnosis_assertion:
            import re
            if re.search(r"(确诊|就是.{0,6}症|肯定是.{0,6}病)", answer):
                safety_parts.append("⚠️ 回答含确定性表述，仅供参考，不能作为确诊依据，务必线下就医核实。")

        # 7) 审计留痕（全量，含召回文档）
        rec = self.audit.record(
            question=safe_question,
            answer=answer,
            user_id=user_id,
            role=role,
            session_id=session_id,
            retrieved_documents=[
                {
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "source": c.source,
                    "section": c.section,
                    "page": c.page,
                }
                for c in citations
            ],
            filter_result=filter_result.as_dict(),
            desensitize_text=False,  # 已脱敏
        )

        return {
            "answer": answer,
            "blocked": False,
            "filter": filter_result.as_dict(),
            "privacy_hits": sorted(set(dq.hits + da.hits)),
            "audit_id": rec.audit_id,
            "source_documents": docs,
            "citations": [c.as_dict() for c in citations],
            "citations_markdown": render_citations_markdown(citations),
            "safety_warning": " ".join(safety_parts),
            "disclaimer": self.settings.disclaimer,
            "role": role,
        }

    @staticmethod
    def format_result(result: dict[str, Any]) -> str:
        """格式化为可读文本（含角色、风险提示、引用、免责声明）。"""
        lines = [
            "=" * 60,
            f"【安全问答 · {result.get('role', 'patient')}端" + (" · 已拦截" if result.get("blocked") else "") + "】",
            result.get("answer", ""),
            "",
        ]
        if result.get("privacy_hits"):
            lines.append(f"【隐私保护】已脱敏字段：{', '.join(result['privacy_hits'])}")
            lines.append("")
        if result.get("safety_warning"):
            lines += [result["safety_warning"], ""]
        citations = result.get("citations") or []
        if citations:
            lines.append("【引用来源】")
            for c in citations:
                loc = []
                if c.get("section"):
                    loc.append(f"章节: {c['section']}")
                if c.get("page", -1) > 0:
                    loc.append(f"第{c['page']}页")
                loc_str = "，".join(loc) if loc else c.get("source", "")
                lines.append(f"  [{c['index']}] {c['doc_title']}（{loc_str}）")
            lines.append("")
        lines += [result.get("disclaimer", ""), "=" * 60]
        return "\n".join(lines)


def build_safety_prefilter_chain(role: str = "patient") -> Runnable:
    """构造可独立使用的 LCEL 前置链：问题字符串 → (脱敏文本, FilterResult)。

    便于在其它 RAG 链（如基础链、专项链）前复用同一套过滤与脱敏逻辑。
    """
    from langchain_core.runnables import RunnableLambda

    def _prefilter(question: str) -> tuple[str, dict[str, Any]]:
        fr = classify_intent(question, role=role)
        dq = desensitize(question)
        return dq.text, fr.as_dict()

    return RunnableLambda(_prefilter)


__all__ = ["SafetyGuardService", "build_safety_prefilter_chain"]
