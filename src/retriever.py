"""检索器：相似度 / MMR / 混合检索 + 可选重排序。"""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import ConfigDict

from .config import Settings
from .utils import get_logger
from .vectorstore import VectorStoreManager

logger = get_logger(__name__)


def _filter_invalid(docs: list[Document], settings: Settings) -> list[Document]:
    """过滤已归档/失效文档 chunk，防止过时知识被检索到（功能四·版本管理）。

    默认开启；chunk 无 doc_status 标签时视为有效（兼容旧索引）。
    """
    if not getattr(settings, "kb_filter_invalid_default", True):
        return docs
    return [d for d in docs if d.metadata.get("doc_status", "active") == "active"]


class _FilteredRetriever(BaseRetriever):
    """装饰任意检索器：检索后过滤失效文档。"""

    inner: BaseRetriever
    settings: Settings

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        docs = self.inner.invoke(query)
        return _filter_invalid(list(docs), self.settings)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        docs = await self.inner.ainvoke(query)
        return _filter_invalid(list(docs), self.settings)


def build_retriever(
    settings: Settings,
    vsm: VectorStoreManager,
    strategy: Optional[str] = None,
) -> BaseRetriever:
    """按配置构建检索器。"""
    strategy = (strategy or settings.retrieval_strategy).lower()
    search_kwargs: dict = {"k": settings.top_k}

    if strategy == "similarity":
        search_type = "similarity"
    elif strategy == "mmr":
        search_type = "mmr"
        search_kwargs["fetch_k"] = settings.fetch_k
        search_kwargs["lambda_mult"] = settings.mmr_lambda
    elif strategy == "similarity_score_threshold":
        search_type = "similarity_score_threshold"
        search_kwargs["score_threshold"] = settings.score_threshold
    elif strategy == "hybrid":
        # 混合检索：向量 + BM25 关键词（需 rank_bm25）
        logger.info("使用混合检索（向量 + BM25）")
        return HybridRetriever(settings, vsm)
    else:
        raise ValueError(f"未知检索策略: {strategy}")

    logger.info("检索策略=%s, top_k=%d", search_type, settings.top_k)
    base = vsm.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return _FilteredRetriever(inner=base, settings=settings)


class HybridRetriever(BaseRetriever):
    """向量检索 + BM25 关键词检索的 RRF 融合（无需额外服务）。"""

    settings: Settings
    vsm: VectorStoreManager

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        top_k = self.settings.top_k

        # 1) 向量检索（多取一些候选）
        vector_docs = self.vsm.store.similarity_search(query, k=top_k * 3)

        # 2) BM25 关键词检索
        bm25_docs: list[Document] = []
        try:
            from rank_bm25 import BM25Okapi

            all_docs = list(self.vsm.store.docstore._dict.values())  # type: ignore[attr-defined]
            corpus = [d.page_content.lower().split() for d in all_docs]
            if corpus:
                bm25 = BM25Okapi(corpus)
                tokens = query.lower().split()
                scores = bm25.get_scores(tokens)
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                bm25_docs = [all_docs[i] for i in ranked[: top_k * 3]]
        except ImportError:
            logger.warning("未安装 rank_bm25，混合检索退化为纯向量检索。pip install rank-bm25")
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 检索失败，退化处理: %s", exc)

        # 3) RRF (Reciprocal Rank Fusion) 融合
        fused = self._reciprocal_rank_fusion([vector_docs, bm25_docs])
        return _filter_invalid(fused[:top_k], self.settings)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

    @staticmethod
    def _reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                key = doc.metadata.get("chunk_id") or doc.page_content[:100]
                doc_map[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        ordered = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return [doc_map[key] for key in ordered]
