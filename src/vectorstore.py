"""向量库封装：FAISS（默认）/ Chroma，支持持久化与增量添加。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """统一管理向量索引的构建、保存、加载与检索。"""

    def __init__(self, settings: Settings, embeddings: Embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self.index_path = settings.vector_store_path / settings.collection_name
        self._store: Optional[FAISS] = None

    def build(self, chunks: list[Document]) -> FAISS:
        """从 chunk 构建索引。"""
        logger.info("构建向量索引，共 %d 个 chunk ...", len(chunks))
        self._store = FAISS.from_documents(chunks, self.embeddings)
        return self._store

    def add_documents(self, chunks: list[Document]) -> None:
        """增量添加文档到已有索引（若未加载则自动加载）。"""
        if not chunks:
            return
        if self._store is None:
            if self.index_path.exists():
                self.load()
            else:
                self.build(chunks)
                return
        self._store.add_documents(chunks)  # type: ignore[union-attr]

    def save(self) -> None:
        if self._store is None:
            raise RuntimeError("向量索引尚未构建，无法保存。")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_path))
        logger.info("向量索引已保存至 %s", self.index_path)

    def load(self) -> FAISS:
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引不存在: {self.index_path}，请先执行 ingest。")
        # FAISS 反序列化需要允许 dangerous deserialization（本地可信索引）
        self._store = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("已加载向量索引: %s", self.index_path)
        return self._store

    def exists(self) -> bool:
        return (self.index_path / "index.faiss").exists()

    def as_retriever(self, **kwargs):
        if self._store is None:
            self.load()
        return self._store.as_retriever(**kwargs)  # type: ignore[union-attr]

    @property
    def store(self) -> FAISS:
        if self._store is None:
            self.load()
        return self._store  # type: ignore[return-value]
