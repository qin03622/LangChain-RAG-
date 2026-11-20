"""MedRAG 顶层系统：统一入口，封装构建索引与问答。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from .audit import AuditLogger
from .chain import build_rag_chain
from .clinical_chain import ClinicalQAService
from .config import Settings, get_settings
from .safety_chain import SafetyGuardService
from .document_processor import DocumentProcessor
from .embeddings import build_embeddings
from .feedback import FeedbackStore
from .knowledge_base import KnowledgeBaseManager
from .llm import build_llm
from .memory import SessionMemory
from .ops import OpsManager
from .retriever import build_retriever
from .specialty_assistant import ClinicalSpecialtyAssistant
from .utils import get_logger
from .vectorstore import VectorStoreManager

logger = get_logger(__name__)


class MedRAGSystem:
    """医疗 RAG 问答系统主类。

    典型用法::

        system = MedRAGSystem()
        system.ingest("./data/raw")     # 首次构建索引
        result = system.query("高血压的用药原则？")
        print(result["answer"])
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embeddings = build_embeddings(self.settings)
        self.llm = build_llm(self.settings)
        self.vsm = VectorStoreManager(self.settings, self.embeddings)
        self.processor = DocumentProcessor(self.settings)
        # 新增：会话记忆 / 反馈 / 临床问答服务
        self.memory = SessionMemory(self.settings, self.llm)
        self.feedback = FeedbackStore(self.settings)
        self.clinical = ClinicalQAService(
            self.settings, self.llm, self.vsm, self.memory, self.feedback
        )
        # 功能二：面向临床医护的专项辅助功能
        self.specialty = ClinicalSpecialtyAssistant(self.settings, self.llm, self.vsm)
        # 功能三：安全 / 合规 / 风控（前置过滤 + 脱敏 + 审计）
        self.audit = AuditLogger(self.settings)
        self.safety = SafetyGuardService(self.settings, self.llm, self.vsm, self.audit)
        # 功能四：知识库管理后台（登记库 / 分类 / 版本 / 权限 / 调试）
        self.kb = KnowledgeBaseManager(self.settings, self.vsm, self.processor)
        # 功能五：后台运维 & 统计（统计 / 模型切换 / 参数配置 / 会话管理）
        self.ops = OpsManager(self)
        # 检索器/链延迟构建：构造时索引可能尚未建立
        self._retriever = None
        self._chain: Optional[Runnable] = None

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = build_retriever(self.settings, self.vsm)
        return self._retriever

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = build_rag_chain(self.llm, self.retriever, self.settings)
        return self._chain

    # ---------- 索引构建 ----------
    def ingest(self, directory: Optional[Path | str] = None, rebuild: bool = True) -> int:
        """加载目录文档、切分并构建向量索引，返回 chunk 数。"""
        path = Path(directory) if directory else self.settings.data_path
        chunks = self.processor.process_directory(path)
        if not chunks:
            logger.warning("未从 %s 读取到任何文档。", path)
            return 0

        if rebuild or not self.vsm.exists():
            self.vsm.build(chunks)
        else:
            self.vsm.add_documents(chunks)
        self.vsm.save()
        self._invalidate_chain()
        logger.info("索引构建完成，共 %d 个 chunk。", len(chunks))
        return len(chunks)

    def ingest_documents(self, documents: Iterable[Document]) -> int:
        """直接从 Document 列表构建索引（便于测试）。"""
        chunks = self.processor.split_documents(documents)
        if not chunks:
            return 0
        if self.vsm.exists() and self.vsm._store is None:
            self.vsm.load()
            self.vsm.add_documents(chunks)
        else:
            self.vsm.build(chunks)
        self.vsm.save()
        self._invalidate_chain()
        return len(chunks)

    def _invalidate_chain(self) -> None:
        """索引更新后重建检索器/链（retriever 绑定了旧 store 引用）。"""
        self._retriever = None
        self._chain = None

    def rebind_llm(self, llm) -> None:
        """(功能五) 模型热切换后，把所有持有 LLM 引用的组件重新绑定。

        LCEL 链在构造时即绑定了具体的 llm 实例，因此必须重建；
        仅重设属性而不重建链会导致切换不生效。
        """
        self.llm = llm
        self.memory.llm = llm
        self.clinical.llm = llm
        self.clinical.retriever.llm = llm
        from langchain_core.output_parsers import StrOutputParser

        from .prompts import QA_PROMPT

        # QA_PROMPT 链在构造时绑定 llm，必须重建才会生效
        self.clinical.chain = QA_PROMPT | llm | StrOutputParser()
        self.specialty.llm = llm
        self.safety.llm = llm
        self._invalidate_chain()

    # ---------- 问答 ----------
    def query(self, question: str, **kwargs) -> dict[str, Any]:
        """同步问答，返回结果 dict。"""
        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        if not self.vsm.exists() and self.vsm._store is None:
            raise FileNotFoundError("向量索引不存在，请先执行 ingest() 构建索引。")
        result = self.chain.invoke(question.strip())
        # (功能五) 问答统计：记录每次问答（含模式/模型/RAG 开关/命中 chunk）
        try:
            docs = result.get("source_documents") or result.get("docs") or []
            self.ops._log_query(
                question.strip(),
                result.get("answer", ""),
                mode="rag" if getattr(self.settings, "rag_enabled", True) else "pure_llm",
                rag_enabled=getattr(self.settings, "rag_enabled", True),
                model=self.ops.current_model(),
                session_id=kwargs.get("session_id"),
                chunk_ids=[str(d.metadata.get("chunk_id", "")) for d in docs],
            )
        except Exception as exc:  # noqa: BLE001 —— 日志失败不应影响问答
            logger.warning("问答统计日志写入失败: %s", exc)
        return result

    def query_simple(self, question: str) -> str:
        """只返回答案文本。"""
        return self.query(question)["answer"]

    # ---------- 临床增强问答 ----------
    def clinical_query(
        self,
        question: str,
        mode: str | None = None,
        session_id: str | None = None,
        use_history: bool = True,
    ) -> dict[str, Any]:
        """临床问答：双模式 + HyDE/MultiQuery/Rerank + 记忆 + 引用溯源 + 反馈调优。"""
        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        if not self.vsm.exists() and self.vsm._store is None:
            raise FileNotFoundError("向量索引不存在，请先执行 ingest() 构建索引。")
        return self.clinical.query(question.strip(), mode=mode, session_id=session_id, use_history=use_history)

    def submit_feedback(
        self,
        question: str,
        answer: str,
        label: str,
        result: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """提交医生反馈：correct / hallucination / outdated / incomplete。"""
        return self.clinical.submit_feedback(question, answer, label, result, note)

    def get_original(self, result: dict[str, Any], chunk_id: str) -> dict[str, Any] | None:
        """点击引用跳转到完整原文片段。"""
        return self.clinical.get_original(result, chunk_id)

    # ---------- 功能二：临床医护专项辅助 ----------
    def specialty_run(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """调用专项辅助功能（7 大功能 / 9 个操作）。

        operation 可选：
          differential, lab_interpret, drug_info, drug_interactions,
          guideline_search, document_summarize, document_template,
          pathway, allergy_risk
        """
        if not self.settings.specialty_enabled:
            raise RuntimeError("专项辅助功能已在配置中禁用（SPECIALTY_ENABLED=false）。")
        return self.specialty.run(operation, **kwargs)

    @staticmethod
    def format_specialty_result(result: dict[str, Any]) -> str:
        return ClinicalSpecialtyAssistant.format_result(result)

    # ---------- 功能四：知识库管理 ----------
    def kb_query(
        self,
        question: str,
        *,
        account: str | None = None,
        department: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """带版本失效过滤 + 权限隔离的知识库问答。"""
        from langchain_core.output_parsers import StrOutputParser

        from .prompts import BASIC_QA_PROMPT, format_context

        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        docs = self.kb.filtered_similarity_search(
            question, k=top_k, account=account, department=department, category=category
        )
        if not docs:
            return {
                "answer": "当前知识库中没有可访问的资料来回答该问题。",
                "source_documents": [],
                "safety_passed": True,
                "safety_warning": "",
                "disclaimer": self.settings.disclaimer,
            }
        context = format_context(docs)
        chain = BASIC_QA_PROMPT | self.llm | StrOutputParser()
        answer = str(chain.invoke({"context": context, "question": question.strip()}))
        passed, warning = (True, "")
        if self.settings.enable_safety_guard:
            from .utils import safety_check

            passed, warning = safety_check(question, answer)
        return {
            "answer": answer,
            "source_documents": docs,
            "safety_passed": passed,
            "safety_warning": warning,
            "disclaimer": self.settings.disclaimer,
        }

    # ---------- 功能三：安全 / 合规 / 风控 ----------
    def safe_query(
        self,
        question: str,
        role: str | None = None,
        user_id: str = "anonymous",
        session_id: str = "",
        history_text: str = "",
    ) -> dict[str, Any]:
        """安全问答：前置过滤 + 隐私脱敏 + 角色口径 + 幻觉抑制 + 审计留痕。

        Args:
            question: 用户问题。
            role: ``patient``（患者端，严格）或 ``clinician``（医护端，专业）。
            user_id: 提问人标识（写入审计）。
            session_id: 多轮会话 ID。
            history_text: 对话历史文本（可选）。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        if not self.vsm.exists() and self.vsm._store is None:
            raise FileNotFoundError("向量索引不存在，请先执行 ingest() 构建索引。")
        role = role or self.settings.safety_default_role
        return self.safety.query(
            question.strip(), role=role, user_id=user_id,
            session_id=session_id, history_text=history_text,
        )

    def audit_entries(self, **kwargs: Any) -> list[dict[str, Any]]:
        """查询审计日志（支持 user_id / role / session_id / keyword 等过滤）。"""
        return self.audit.entries(**kwargs)

    def audit_stats(self) -> dict[str, Any]:
        """审计统计。"""
        return self.audit.stats()

    @staticmethod
    def format_safe_result(result: dict[str, Any]) -> str:
        return SafetyGuardService.format_result(result)

    def feedback_stats(self) -> dict[str, Any]:
        return self.feedback.stats()

    def new_session(self) -> str:
        return self.memory.new_session()

    def purge_expired_sessions(self) -> int:
        return self.memory.purge_expired()

    # ---------- 功能五：后台运维 & 统计 ----------
    def ops_stats(self, top_n: int = 10, keyword: str | None = None) -> dict[str, Any]:
        """问答统计：高频病种 / 高频问题 / 用户反馈（幻觉数）。"""
        return self.ops.stats(top_n=top_n, keyword=keyword)

    def ops_list_models(self) -> list[dict]:
        return self.ops.list_models()

    def ops_switch_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return self.ops.switch_model(provider, model, temperature=temperature)

    def ops_current_model(self) -> dict[str, Any]:
        return self.ops.current_model()

    def ops_set_rag(self, enabled: bool) -> dict[str, Any]:
        return self.ops.set_rag(enabled)

    def ops_get_params(self) -> dict[str, Any]:
        return self.ops.get_params()

    def ops_set_params(self, **params: Any) -> dict[str, Any]:
        return self.ops.set_params(**params)

    def ops_reset_params(self) -> dict[str, Any]:
        return self.ops.reset_params()

    def ops_compare(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """纯 LLM vs RAG 对比测试。"""
        return self.ops.compare(question, **kwargs)

    def ops_list_sessions(self, include_expired: bool = False) -> list[dict[str, Any]]:
        return self.ops.list_sessions(include_expired=include_expired)

    def ops_get_session(self, session_id: str) -> dict[str, Any]:
        return self.ops.get_session(session_id)

    def ops_delete_session(self, session_id: str) -> dict[str, Any]:
        return self.ops.delete_session(session_id)

    def ops_purge_sensitive_sessions(self) -> dict[str, Any]:
        return self.ops.purge_sensitive_sessions()

    @staticmethod
    def format_result(result: dict[str, Any]) -> str:
        """把结果格式化为可读文本。"""
        lines = ["=" * 60, "【回答】", result.get("answer", ""), ""]
        if result.get("safety_warning"):
            lines += [result["safety_warning"], ""]
        docs = result.get("source_documents") or result.get("docs") or []
        if docs:
            lines.append("【参考来源】")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("filename") or doc.metadata.get("source", "未知")
                snippet = doc.page_content[:80].replace("\n", " ")
                lines.append(f"  [{i}] {src}: {snippet}...")
        lines += ["", result.get("disclaimer", ""), "=" * 60]
        return "\n".join(lines)
