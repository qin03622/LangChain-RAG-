"""引用溯源：把检索到的 Document 转成结构化引用，并支持查看原文片段。

每条引用包含：文档名称、章节、页码、chunk_id、原文片段，
回答中的 [来源N] 可据此跳转核对，防止幻觉。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from langchain_core.documents import Document

from .prompts import format_context


@dataclass
class Citation:
    index: int                 # 对应回答里的 [来源N]
    doc_title: str            # 文档标题
    source: str               # 文件路径
    section: str              # 章节路径
    page: int                 # 页码，-1 表示未知
    chunk_id: str
    content: str              # 原文片段

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_citations(documents: Iterable[Document]) -> list[Citation]:
    """从检索结果构造引用列表。"""
    citations: list[Citation] = []
    for i, doc in enumerate(documents, start=1):
        md = doc.metadata
        citations.append(
            Citation(
                index=i,
                doc_title=md.get("doc_title") or md.get("filename") or md.get("source", "未知文档"),
                source=md.get("source", ""),
                section=md.get("section", ""),
                page=int(md.get("page", -1)),
                chunk_id=md.get("chunk_id", ""),
                content=doc.page_content.strip(),
            )
        )
    return citations


def format_citation_line(c: Citation) -> str:
    """人类可读的一行引用，含文档名/章节/页码。"""
    loc = []
    if c.section:
        loc.append(f"章节: {c.section}")
    if c.page and c.page > 0:
        loc.append(f"第 {c.page} 页")
    loc_str = "，".join(loc) if loc else c.source
    return f"[{c.index}] {c.doc_title}（{loc_str}）"


def render_citations_markdown(citations: list[Citation]) -> str:
    """渲染成可折叠的“点击查看原文”Markdown（前端/API 可直接展示）。"""
    lines = ["### 引用来源（点击查看原文片段）"]
    for c in citations:
        lines.append(f"\n<details><summary>{format_citation_line(c)}</summary>\n")
        lines.append(f"```text\n{c.content}\n```\n")
        lines.append(f"</details>")
    return "\n".join(lines)


def get_original_fragment(documents: Iterable[Document], chunk_id: str) -> dict[str, Any] | None:
    """按 chunk_id 查出完整原文片段，供“原文跳转”接口使用。"""
    for doc in documents:
        if doc.metadata.get("chunk_id") == chunk_id:
            md = doc.metadata
            return {
                "chunk_id": chunk_id,
                "doc_title": md.get("doc_title") or md.get("filename", ""),
                "source": md.get("source", ""),
                "section": md.get("section", ""),
                "page": md.get("page", -1),
                "full_content": md.get("full_content") or doc.page_content,
            }
    return None


def citations_to_context(citations: list[Citation]) -> str:
    """把引用重新拼成带编号上下文（供 LLM Prompt 使用）。"""
    parts = []
    for c in citations:
        header = f"[{c.index}] 来源: {c.doc_title}"
        if c.section:
            header += f" / {c.section}"
        if c.page and c.page > 0:
            header += f" / 第{c.page}页"
        parts.append(f"{header}\n{c.content}")
    return "\n---\n".join(parts)
