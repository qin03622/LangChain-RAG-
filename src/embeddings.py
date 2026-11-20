"""Embedding 工厂：OpenAI 兼容接口 / Ollama 本地模型。"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)


class FakeEmbeddings(Embeddings):
    """确定性哈希向量，仅用于离线测试 / 演示，不用于生产。"""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        import hashlib

        vec = [0.0] * self.dimension
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dimension
            vec[idx] += 1.0
        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    """根据配置构造 Embedding 实例。"""
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("请安装 langchain-openai: pip install langchain-openai") from exc
        logger.info("使用 OpenAI 兼容 Embedding: %s", settings.embedding_model)
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key or settings.llm_api_key or "EMPTY",
            base_url=settings.embedding_base_url or settings.llm_base_url,
            check_embedding_ctx_length=False,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("请安装 langchain-ollama: pip install langchain-ollama") from exc
        logger.info("使用 Ollama Embedding: %s", settings.embedding_model)
        return OllamaEmbeddings(model=settings.embedding_model)

    if provider in {"fake", "test", "local-hash"}:
        logger.warning("使用 FakeEmbeddings（仅用于测试，结果无语义）")
        return FakeEmbeddings()

    raise ValueError(f"不支持的 Embedding 提供方: {provider}")
