"""MedRAG: 基于 LangChain 的医疗临床智能诊疗 RAG 问答系统。

主要导出:
    MedRAGSystem      顶层系统类
    DocumentProcessor 文档加载与切分
    VectorStoreManager 向量索引管理
    build_embeddings / build_llm / build_retriever / build_rag_chain
"""

__version__ = "1.0.0"

from .config import Settings, get_settings
from .rag_system import MedRAGSystem

__all__ = [
    "MedRAGSystem",
    "Settings",
    "get_settings",
    "__version__",
]
