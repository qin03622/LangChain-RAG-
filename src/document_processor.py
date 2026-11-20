"""文档加载与切分：支持 PDF / DOCX / TXT / MD / CSV。

医疗文档通常带有结构化标题，切分时优先按标题层级切分，
再对过长 chunk 做递归字符切分，尽量保留临床语义完整。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)

# 尝试导入可选文档加载器（缺失时给出友好提示）
try:
    from langchain_community.document_loaders import (
        CSVLoader,
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    _LOADERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LOADERS_AVAILABLE = False
    logger.warning("langchain_community 文档加载器不可用，仅支持 .txt/.md 纯文本。")


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".csv"}


class DocumentProcessor:
    """负责把原始医疗文档转换为带元数据的 Document chunk 列表。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

    def reconfigure(self, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
        """(功能五) 运行时更新默认切分器参数，供运维参数可视化配置使用。

        仅改变后续切分行为，不重建任何已有向量索引；返回更新后的生效值。
        """
        size = int(chunk_size) if chunk_size is not None else self.settings.chunk_size
        overlap = int(chunk_overlap) if chunk_overlap is not None else self.settings.chunk_overlap
        if size < 1:
            raise ValueError("chunk_size 必须为正整数")
        if overlap < 0 or overlap >= size:
            raise ValueError("chunk_overlap 必须 >= 0 且小于 chunk_size")
        self.settings.chunk_size = size
        self.settings.chunk_overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        return {"chunk_size": size, "chunk_overlap": overlap}

    def load_file(self, path: Path) -> list[Document]:
        """加载单个文件为 Document 列表。"""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {suffix}（支持: {SUPPORTED_EXTENSIONS}）")

        if not _LOADERS_AVAILABLE:
            if suffix in {".txt", ".md", ".markdown"}:
                return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]
            raise RuntimeError("缺少 langchain_community，无法加载该类型文件。")

        if suffix == ".pdf":
            docs = PyPDFLoader(str(path)).load()
        elif suffix in {".docx", ".doc"}:
            docs = Docx2txtLoader(str(path)).load()
        elif suffix == ".csv":
            docs = CSVLoader(file_path=str(path), encoding="utf-8").load()
        else:
            # .txt / .md / .markdown 用纯文本加载器，避免额外依赖 unstructured
            docs = TextLoader(str(path), encoding="utf-8").load()

        for d in docs:
            d.metadata.setdefault("source", str(path))
            d.metadata["filename"] = path.name
            # doc_title：去掉扩展名的文件名，作为引用时展示的文档标题
            d.metadata.setdefault("doc_title", path.stem)
            d.metadata["doc_type"] = suffix.lstrip(".")
            # page：PDF 等带页码的加载器会提供 page；缺失时用 -1 表示未知
            if "page" not in d.metadata:
                d.metadata["page"] = -1
        return docs

    def load_directory(self, directory: Optional[Path] = None) -> list[Document]:
        """递归加载目录下所有支持的文件。"""
        directory = Path(directory or self.settings.data_path)
        if not directory.exists():
            logger.warning("数据目录不存在: %s", directory)
            return []

        documents: list[Document] = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    documents.extend(self.load_file(path))
                    logger.info("加载文档: %s", path)
                except Exception as exc:  # noqa: BLE001
                    logger.error("加载失败 %s: %s", path, exc)
        return documents

    def split_documents(
        self,
        documents: Iterable[Document],
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ) -> list[Document]:
        """对 Document 列表做语义感知切分。

        Markdown 文件优先按标题切分，再对每段做字符级切分。
        可通过 chunk_size / chunk_overlap / separators 临时覆盖全局切分配置
        （知识库管理后台上传文档时可自定义医疗分隔符，如章节、条款）。
        """
        size = chunk_size or self.settings.chunk_size
        overlap = self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        # 自定义较小的 chunk_size 时，重叠必须小于块大小，否则切分器报错
        if overlap >= size:
            overlap = max(0, size - 1)
        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=separators,
                length_function=len,
            )
            if (separators is not None or size != self.settings.chunk_size or overlap != self.settings.chunk_overlap)
            else self.splitter
        )

        final_chunks: list[Document] = []
        for doc in documents:
            if doc.metadata.get("doc_type") in {"md", "markdown"}:
                sections = self.md_header_splitter.split_text(doc.page_content)
                for sec in sections:
                    sec.metadata.update(doc.metadata)
                    # 用 Markdown 标题拼出章节路径，供引用溯源展示
                    headers = [sec.metadata[k] for k in ("h1", "h2", "h3") if k in sec.metadata]
                    sec.metadata["section"] = " / ".join(headers)
                    for sub in splitter.split_documents([sec]):
                        sub.metadata["section"] = sec.metadata["section"]
                        final_chunks.append(sub)
            else:
                for sub in splitter.split_documents([doc]):
                    sub.metadata.setdefault("section", "")
                    final_chunks.append(sub)

        # 过滤空 chunk 并补充 chunk_id
        result: list[Document] = []
        for i, chunk in enumerate(final_chunks):
            if not chunk.page_content.strip():
                continue
            chunk.metadata["chunk_id"] = f"chunk-{i:06d}"
            # 保存原始完整文本，便于“点击跳转查看原文片段”
            chunk.metadata.setdefault("full_content", chunk.page_content)
            result.append(chunk)
        logger.info("切分完成: %d 个 chunk", len(result))
        return result

    def process_directory(self, directory: Optional[Path] = None) -> list[Document]:
        """加载 + 切分一步到位。"""
        return self.split_documents(self.load_directory(directory))

    @staticmethod
    def build_csv_from_qa(rows: list[tuple[str, str]], path: Path) -> None:
        """工具方法：把 (问题, 答案) 对写成 CSV，便于构造 FAQ 语料。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerows(rows)
