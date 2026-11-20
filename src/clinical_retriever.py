"""临床检索增强：HyDE 改写、MultiQuery 多查询、Rerank 重排、双模式检索。

- precise（精准检索）：优先指南/规范原文，top_k 较小，引用集中。
- comprehensive（综合推理）：多文档交叉，top_k 较大，配合 MultiQuery 与 Rerank。
"""
from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .utils import get_logger
from .vectorstore import VectorStoreManager

logger = get_logger(__name__)

_GUIDELINE_HINTS = ("指南", "规范", "共识", "标准", "指导原则", "guideline", "consensus")


class ClinicalRetriever:
    """封装临床场景的检索流程。"""

    def __init__(self, settings: Settings, llm: BaseChatModel, vsm: VectorStoreManager):
        self.settings = settings
        self.llm = llm
        self.vsm = vsm
        self._hyde_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是临床专家。请针对问题写一段简短的假设性答案（含可能的诊疗要点），"
                    "用于辅助检索医学文献。只输出假设答案本身，不要解释。",
                ),
                ("human", "{question}"),
            ]
        )
        self._multiquery_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是医学检索专家。把用户问题改写成 {n} 个不同角度的检索问题，"
                    "覆盖症状、用药、指南、并发症、禁忌症等维度。每行一个，只输出问题。",
                ),
                ("human", "{question}"),
            ]
        )

    # ---------- HyDE ----------
    def _hyde(self, question: str) -> str:
        hypothetical = (self._hyde_prompt | self.llm | StrOutputParser()).invoke(
            {"question": question}
        )
        logger.info("HyDE 假设答案: %s", str(hypothetical)[:80])
        return str(hypothetical).strip()

    # ---------- MultiQuery ----------
    def _multi_queries(self, question: str) -> list[str]:
        raw = (self._multiquery_prompt | self.llm | StrOutputParser()).invoke(
            {"question": question, "n": self.settings.multi_query_count}
        )
        queries = [line.strip("- ·*0123456789. ") for line in str(raw).splitlines() if line.strip()]
        return queries[: self.settings.multi_query_count] or [question]

    # ---------- Rerank ----------
    def _finalize_ranked(
        self, ranked: list[tuple[Document, float]], top_k: int
    ) -> list[Document]:
        """(功能五) 对 (文档, 得分) 列表应用 rerank 阈值并截断 top_k。

        不同打分器量纲不同（CrossEncoder logits / LLM 0-10 / 启发式计数），
        故先按批内 min-max 归一化到 0-1 再与阈值比较。默认阈值 0 时行为与之前完全一致。
        """
        threshold = float(getattr(self.settings, "rerank_threshold", 0.0) or 0.0)
        if threshold > 0 and ranked:
            scores = [float(s) for _, s in ranked]
            lo, hi = min(scores), max(scores)
            span = hi - lo
            if span > 1e-9:
                ranked = [
                    (d, s) for d, s in ranked if (float(s) - lo) / span >= threshold
                ]
            else:
                # 得分完全相同，无法区分：阈值 <=1 时全部保留
                ranked = ranked if threshold <= 1.0 else []
        return [d for d, _ in ranked[:top_k]]

    def _rerank(self, question: str, docs: list[Document], top_k: int) -> list[Document]:
        """优先用 CrossEncoder；不可用时用 LLM 打分；再失败按指南权重排序。"""
        if not docs:
            return []
        mode = self.settings.rerank_model or ""

        # 离线 fake 模式：直接用启发式排序，避免无意义的重排报错噪音
        if self.settings.llm_provider.lower() in {"fake", "test"}:
            def _fake_score(d: Document) -> float:
                text = (d.page_content + " " + str(d.metadata.get("section", ""))).lower()
                s = sum(1 for h in _GUIDELINE_HINTS if h in text) * self.settings.guideline_boost
                s += sum(1 for term in re.findall(r"[\u4e00-\u9fa5]{2,}", question) if term in text)
                return s
            ranked = [(d, _fake_score(d)) for d in docs]
            ranked.sort(key=lambda x: x[1], reverse=True)
            return self._finalize_ranked(ranked, top_k)

        # 1) CrossEncoder（需安装 sentence-transformers）
        if mode and mode != "llm":
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                model = CrossEncoder(mode)
                pairs = [[question, d.page_content] for d in docs]
                scores = model.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return self._finalize_ranked(ranked, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CrossEncoder 重排不可用(%s)，尝试 LLM 重排。", exc)

        # 2) LLM 打分重排（离线 fake 模式下会退化为原顺序）
        try:
            from langchain_core.output_parsers import JsonOutputParser

            score_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "给每个片段与问题的相关性打 0-10 分，返回 JSON 数组，如 [0, 8, 3]，只输出 JSON。",
                    ),
                    ("human", "问题: {question}\n片段:\n{docs}"),
                ]
            )
            docs_text = "\n".join(f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs))
            scores = (score_prompt | self.llm | JsonOutputParser()).invoke(
                {"question": question, "docs": docs_text}
            )
            if isinstance(scores, list) and len(scores) == len(docs):
                ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
                return self._finalize_ranked(ranked, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 重排不可用(%s)，使用关键词启发式排序。", exc)

        # 3) 启发式：指南优先 + 关键词命中数
        def score(d: Document) -> float:
            text = (d.page_content + " " + str(d.metadata.get("section", ""))).lower()
            s = sum(1 for h in _GUIDELINE_HINTS if h in text) * self.settings.guideline_boost
            s += sum(1 for term in re.findall(r"[\u4e00-\u9fa5]{2,}", question) if term in text)
            return s

        ranked = sorted([(d, score(d)) for d in docs], key=lambda x: x[1], reverse=True)
        return self._finalize_ranked(ranked, top_k)

    # ---------- 去重 ----------
    @staticmethod
    def _dedup(docs: Iterable[Document]) -> list[Document]:
        seen: set[str] = set()
        out: list[Document] = []
        for d in docs:
            key = d.metadata.get("chunk_id") or d.page_content[:80]
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    # ---------- 主入口 ----------
    def retrieve(self, question: str, mode: str | None = None) -> list[Document]:
        mode = (mode or self.settings.clinical_mode).lower()
        if mode not in {"precise", "comprehensive"}:
            raise ValueError(f"未知临床检索模式: {mode}")

        top_k = (
            self.settings.comprehensive_top_k
            if mode == "comprehensive"
            else self.settings.precise_top_k
        )
        queries = [question]

        if mode == "comprehensive":
            if self.settings.use_hyde:
                queries.append(self._hyde(question))
            if self.settings.use_multi_query:
                queries.extend(self._multi_queries(question))
        # 精准模式：也可选 HyDE，但默认直接原文检索以优先引用指南

        candidates: list[Document] = []
        store = self.vsm.store
        for q in queries:
            candidates.extend(store.similarity_search(q, k=top_k * 2))

        # 功能四：默认过滤已归档/失效文档，防止过时知识被引用
        if getattr(self.settings, "kb_filter_invalid_default", True):
            candidates = [
                d for d in candidates if d.metadata.get("doc_status", "active") == "active"
            ]

        candidates = self._dedup(candidates)

        if mode == "precise":
            # 精准模式：指南类文档加权提前
            candidates = self._rerank(question, candidates, top_k)
        else:
            if self.settings.use_rerank:
                candidates = self._rerank(question, candidates, top_k)
            else:
                candidates = candidates[:top_k]

        logger.info("临床检索模式=%s，查询数=%d，返回 %d 个文档", mode, len(queries), len(candidates))
        return candidates
