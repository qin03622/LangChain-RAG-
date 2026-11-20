"""知识库管理后台（功能四）：LangChain RAG 配套管理能力。

覆盖 6 项能力：
  1. 文档上传与增量入库：PDF/Word/Markdown/txt，按章节/条款等医疗分隔符分块
  2. 文档向量化与批量导入：入库时打标签，向量化后增量加入 FAISS
  3. 知识库分类目录：临床指南、药学、检验、内科、外科、急诊、儿科、护理、感染控制
  4. 文档版本管理：新版入库自动归档旧版、标记失效；检索默认过滤失效文档
  5. 知识库权限隔离：public / department / confidential 三级可见性 + 账号白名单
  6. 检索测试面板：召回片段、重排结果、chunk 大小 / top-k / rerank 参数调试

设计原则：
  - 登记库（registry.json）只存元数据，向量仍存 FAISS；通过 chunk 的 doc_id 标签
    关联回文档记录，实现“失效/权限”过滤——无需重建索引。
  - 不再修改 FAISS 中已入库的 chunk；归档/失效只是状态变更，安全且可回滚（重新激活）。
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document

from .config import Settings
from .document_processor import DocumentProcessor
from .utils import get_logger
from .vectorstore import VectorStoreManager

logger = get_logger(__name__)

# ---------- 常量与校验 ----------
DEFAULT_CATEGORIES = [
    "临床指南", "药学", "检验", "内科", "外科", "急诊", "儿科", "护理", "感染控制",
]
VALID_STATUSES = {"active", "archived", "invalid"}
VALID_VISIBILITIES = {"public", "department", "confidential"}
# 医疗文档常用分隔符：章节标题、条款编号、段落、句子（递归字符切分器按顺序尝试）
MEDICAL_SEPARATORS = [
    "\n第", "\n第", "\n## ", "\n# ", "\n### ", "\n一、", "\n二、", "\n三、",
    "\n\n", "\n", "。", "；", "，", " ", "",
]
# 去重后保留
MEDICAL_SEPARATORS = list(dict.fromkeys(MEDICAL_SEPARATORS))


def _now() -> str:
    # 微秒精度：同一秒内多次入库也能正确排序版本
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class KnowledgeBaseManager:
    """知识库管理后台主类。线程安全（文件写入 + 状态变更加锁）。"""

    def __init__(
        self,
        settings: Settings,
        vsm: VectorStoreManager,
        processor: Optional[DocumentProcessor] = None,
    ):
        self.settings = settings
        self.vsm = vsm
        self.processor = processor or DocumentProcessor(settings)
        self.registry_path = settings.kb_registry_file
        self.upload_dir = settings.kb_upload_path
        self._lock = threading.RLock()
        self._registry: dict[str, Any] = self._load_registry()
        # 分类目录持久化（允许管理员扩充）
        self._registry.setdefault("categories", self._categories_from_settings())

    # ========== 登记库持久化 ==========
    def _categories_from_settings(self) -> list[str]:
        raw = self.settings.kb_categories
        cats = [c.strip() for c in raw.split(",") if c.strip()]
        return cats or list(DEFAULT_CATEGORIES)

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            try:
                with self.registry_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("documents", {})
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("知识库登记文件损坏，将重建: %s", exc)
        return {"documents": {}, "categories": list(DEFAULT_CATEGORIES)}

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(self._registry, f, ensure_ascii=False, indent=2)

    @property
    def documents(self) -> dict[str, dict[str, Any]]:
        return self._registry["documents"]

    # ========== 3. 分类目录 ==========
    def list_categories(self) -> list[str]:
        """返回分类目录及其文档计数。"""
        return list(self._registry.get("categories", []))

    def category_stats(self) -> list[dict[str, Any]]:
        """每个分类的文档数（按 active 文档统计）。"""
        counts: dict[str, int] = {c: 0 for c in self.list_categories()}
        for doc in self.documents.values():
            if doc["status"] == "active":
                counts[doc["category"]] = counts.get(doc["category"], 0) + 1
        return [{"name": c, "active_docs": counts.get(c, 0)} for c in self.list_categories()]

    def add_category(self, name: str) -> bool:
        name = name.strip()
        if not name:
            raise ValueError("分类名不能为空。")
        with self._lock:
            cats = self._registry.setdefault("categories", [])
            if name in cats:
                return False
            cats.append(name)
            self._save_registry()
            return True

    # ========== 1 & 2. 上传 / 分块 / 向量化 / 增量入库 ==========
    def upload_file(
        self,
        file_path: Path | str,
        *,
        category: str = "临床指南",
        title: Optional[str] = None,
        version: str = "v1.0",
        visibility: str = "public",
        allowed_accounts: Optional[list[str]] = None,
        replace_latest: bool = True,
        separators: Optional[list[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> dict[str, Any]:
        """上传单个文档并增量入库（支持 PDF/DOCX/MD/TXT）。

        流程：校验 → 归档旧版 → 分块（医疗分隔符）→ 打 doc_id 标签 →
        向量化增量加入 FAISS → 更新索引 → 登记元数据。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if visibility not in VALID_VISIBILITIES:
            raise ValueError(f"未知可见性: {visibility}，可选: {sorted(VALID_VISIBILITIES)}")

        with self._lock:
            # 分类规范化
            if category not in self.list_categories():
                self.add_category(category)

            # 4) 同 title 的旧版本：归档 + 标记失效（replace_latest 时）
            archived: list[str] = []
            if replace_latest:
                for old in self.documents.values():
                    if old["title"] == (title or path.stem) and old["status"] == "active":
                        self._set_status_locked(old["doc_id"], "archived")
                        archived.append(old["doc_id"])

            # 1) 分块：医疗文档优先章节/条款分隔符
            base_docs = self.processor.load_file(path)
            chunks = self.processor.split_documents(
                base_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators if separators is not None else MEDICAL_SEPARATORS,
            )
            if not chunks:
                raise ValueError(f"未从文件中解析出任何文本块: {path.name}")

            doc_id = f"doc-{uuid.uuid4().hex[:12]}"
            for c in chunks:
                c.metadata["doc_id"] = doc_id
                c.metadata["doc_status"] = "active"
                c.metadata["visibility"] = visibility
                c.metadata["kb_department"] = (
                    (allowed_accounts or ["default"])[0] if visibility == "department" else None
                )
                c.metadata["kb_category"] = category

            # 2) 向量化 + 增量入库
            if self.vsm.exists() and self.vsm._store is None:
                self.vsm.load()
            if self.vsm._store is None:
                self.vsm.build(chunks)
            else:
                self.vsm.add_documents(chunks)
            self.vsm.save()

            # 登记元数据
            record = {
                "doc_id": doc_id,
                "title": title or path.stem,
                "filename": path.name,
                "category": category,
                "version": version,
                "visibility": visibility,
                "allowed_accounts": allowed_accounts or [],
                "department": None if visibility != "department" else (allowed_accounts or ["default"])[0],
                "status": "active",
                "chunk_count": len(chunks),
                "uploaded_at": _now(),
                "source_path": str(path.resolve()),
                "archived_versions": archived,
            }
            self.documents[doc_id] = record
            self._save_registry()
            logger.info("文档入库: %s (%s, %d chunks)", record["title"], doc_id, len(chunks))
            return dict(record)

    def upload_batch(self, files: Iterable[tuple[Path | str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """批量导入：(文件路径, 上传参数 dict) 列表，逐篇调用 upload_file。"""
        results = []
        for path, kwargs in files:
            try:
                results.append(self.upload_file(path, **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.error("批量导入失败 %s: %s", path, exc)
                results.append({"filename": str(path), "error": str(exc)})
        return results

    def save_upload(self, filename: str, content: bytes) -> Path:
        """保存上传字节到暂存目录（供 API 层调用），返回落盘路径。"""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        safe = Path(filename).name  # 防目录穿越
        target = self.upload_dir / f"{uuid.uuid4().hex[:8]}_{safe}"
        target.write_bytes(content)
        return target

    # ========== 4. 版本管理 ==========
    def list_versions(self, title: Optional[str] = None, doc_id: Optional[str] = None) -> list[dict[str, Any]]:
        """列出同一文档（按 title）的全部版本，按上传时间倒序。"""
        docs = list(self.documents.values())
        if doc_id:
            docs = [d for d in docs if d["doc_id"] == doc_id]
        elif title:
            docs = [d for d in docs if d["title"] == title]
        return sorted(docs, key=lambda d: d["uploaded_at"], reverse=True)

    def _set_status_locked(self, doc_id: str, status: str) -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"未知状态: {status}")
        if doc_id not in self.documents:
            raise KeyError(f"文档不存在: {doc_id}")
        self.documents[doc_id]["status"] = status
        self.documents[doc_id]["updated_at"] = _now()
        self._save_registry()
        # 同步刷新内存索引中 chunk 的状态标签（如已加载），确保当前会话立即生效
        if self.vsm._store is not None:
            for doc in self.vsm.store.docstore._dict.values():  # type: ignore[attr-defined]
                if doc.metadata.get("doc_id") == doc_id:
                    doc.metadata["doc_status"] = status
        return dict(self.documents[doc_id])

    def set_status(self, doc_id: str, status: str) -> dict[str, Any]:
        """归档 / 标记失效 / 重新激活。"""
        with self._lock:
            return self._set_status_locked(doc_id, status)

    def archive(self, doc_id: str) -> dict[str, Any]:
        return self.set_status(doc_id, "archived")

    def invalidate(self, doc_id: str) -> dict[str, Any]:
        return self.set_status(doc_id, "invalid")

    def reactivate(self, doc_id: str) -> dict[str, Any]:
        return self.set_status(doc_id, "active")

    # ========== 5. 权限隔离 ==========
    def _can_access(self, doc: Document, account: Optional[str], department: Optional[str]) -> bool:
        vis = doc.metadata.get("visibility", "public")
        if vis == "public":
            return True
        if vis == "department":
            dept_tag = doc.metadata.get("kb_department")
            return department is not None and department in (dept_tag or "")
        # confidential：必须账号在白名单内
        allowed = self.documents.get(doc.metadata.get("doc_id", ""), {}).get("allowed_accounts", [])
        return account is not None and account in allowed

    # ========== 4 + 5. 过滤检索 ==========
    def filtered_similarity_search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        account: Optional[str] = None,
        department: Optional[str] = None,
        include_invalid: bool = False,
        category: Optional[str] = None,
    ) -> list[Document]:
        """相似度检索 + 失效过滤 + 权限过滤（默认过滤失效文档）。

        多召回 candidate_buffer 条候选，再按状态/权限/分类过滤并裁剪到 k 条。
        """
        k = k or self.settings.top_k
        fetch = k + self.settings.kb_candidate_buffer
        store = self.vsm.store  # 自动加载
        candidates = store.similarity_search(query, k=fetch)

        result: list[Document] = []
        for doc in candidates:
            meta = doc.metadata
            # 状态过滤：默认排除 archived / invalid
            if not include_invalid and meta.get("doc_status", "active") != "active":
                continue
            # 分类过滤
            if category and meta.get("kb_category") != category:
                continue
            # 权限过滤
            if not self._can_access(doc, account, department):
                continue
            result.append(doc)
            if len(result) >= k:
                break
        logger.info(
            "知识库检索: query=%r 候选=%d 通过过滤=%d (account=%s, dept=%s)",
            query, len(candidates), len(result), account, department,
        )
        return result

    # ========== 6. 检索测试面板 ==========
    def debug_retrieval(
        self,
        question: str,
        *,
        account: Optional[str] = None,
        department: Optional[str] = None,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        recall_k: Optional[int] = None,
        use_rerank: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        include_invalid: bool = False,
    ) -> dict[str, Any]:
        """调试面板：展示召回片段 → 过滤 → 重排全流程，并支持参数调优。

        返回结构便于前端/管理员逐环节排查：
          - recall: 原始召回（含得分、状态、可见性）
          - filtered: 过滤后
          - reranked: 重排后（若启用）
          - params: 本次实际使用的参数
          - chunking_preview: 用指定 chunk_size 重新切分的预览统计
        """
        top_k = top_k if top_k is not None else self.settings.kb_debug_top_k
        recall_k = recall_k if recall_k is not None else self.settings.kb_recall_k
        use_rerank = (
            use_rerank if use_rerank is not None else self.settings.kb_debug_use_rerank
        )
        eff_chunk = chunk_size or (self.settings.kb_debug_chunk_size or None)

        store = self.vsm.store
        # 带分数的原始召回
        raw = store.similarity_search_with_score(question, k=recall_k)
        recall_items = [
            {
                "rank": i + 1,
                "chunk_id": d.metadata.get("chunk_id"),
                "doc_id": d.metadata.get("doc_id"),
                "title": d.metadata.get("doc_title"),
                "category": d.metadata.get("kb_category"),
                "status": d.metadata.get("doc_status", "active"),
                "visibility": d.metadata.get("visibility", "public"),
                "score": round(float(s), 4),
                "content": d.page_content[:300],
            }
            for i, (d, s) in enumerate(raw)
        ]

        # 应用过滤
        filtered: list[Document] = []
        for d, _ in raw:
            if not include_invalid and d.metadata.get("doc_status", "active") != "active":
                continue
            if category and d.metadata.get("kb_category") != category:
                continue
            if not self._can_access(d, account, department):
                continue
            filtered.append(d)

        # 重排（复用临床启发式：指南权重 + 关键词命中，离线可用）
        rerank_used = False
        final = filtered[:top_k]
        if use_rerank and filtered:
            try:
                from .clinical_retriever import ClinicalRetriever

                cr = ClinicalRetriever(self.settings, self._debug_llm(), self.vsm)
                final = cr._rerank(question, filtered, top_k)
                rerank_used = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("调试面板重排失败，使用召回顺序: %s", exc)

        # chunk 大小预览：用指定 size 重新切分已有原文，给出块数变化
        chunking_preview = None
        if eff_chunk:
            full_texts = [d.metadata.get("full_content", d.page_content) for d, _ in raw[:5]]
            docs = [Document(page_content=t) for t in full_texts]
            preview_chunks = self.processor.split_documents(docs, chunk_size=eff_chunk)
            avg_len = (
                round(sum(len(c.page_content) for c in preview_chunks) / len(preview_chunks), 1)
                if preview_chunks else 0
            )
            chunking_preview = {
                "chunk_size": eff_chunk,
                "preview_source_docs": len(docs),
                "resulting_chunks": len(preview_chunks),
                "avg_chunk_chars": avg_len,
            }

        return {
            "question": question,
            "params": {
                "top_k": top_k,
                "recall_k": recall_k,
                "use_rerank": use_rerank,
                "chunk_size": eff_chunk,
                "include_invalid": include_invalid,
                "category": category,
                "account": account,
                "department": department,
            },
            "recall": recall_items,
            "recall_count": len(recall_items),
            "filtered_count": len(filtered),
            "rerank_used": rerank_used,
            "results": [
                {
                    "rank": i + 1,
                    "chunk_id": d.metadata.get("chunk_id"),
                    "title": d.metadata.get("doc_title"),
                    "section": d.metadata.get("section", ""),
                    "content": d.page_content,
                    "doc_status": d.metadata.get("doc_status", "active"),
                }
                for i, d in enumerate(final)
            ],
            "chunking_preview": chunking_preview,
        }

    def _debug_llm(self):
        """调试面板重排可能用到 LLM；复用 Fake 避免真实调用。"""
        from .llm import build_llm
        from .config import Settings

        if self.settings.llm_provider.lower() in {"fake", "test"}:
            from .llm import FakeChatModel

            return FakeChatModel()
        return build_llm(self.settings)

    # ========== 概览 ==========
    def overview(self) -> dict[str, Any]:
        docs = list(self.documents.values())
        return {
            "kb_enabled": self.settings.kb_enabled,
            "total_documents": len(docs),
            "active_documents": sum(1 for d in docs if d["status"] == "active"),
            "archived_documents": sum(1 for d in docs if d["status"] == "archived"),
            "invalid_documents": sum(1 for d in docs if d["status"] == "invalid"),
            "total_chunks": sum(d["chunk_count"] for d in docs),
            "categories": self.category_stats(),
            "vector_index_ready": self.vsm.exists(),
        }
