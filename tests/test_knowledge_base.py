"""功能四：知识库管理后台测试（离线 Fake 模式）。

覆盖：
  1. 文档上传与增量入库 / 医疗分隔符分块
  2. 文档向量化与批量导入
  3. 知识库分类目录
  4. 文档版本管理（归档/失效 + 检索过滤）
  5. 知识库权限隔离（public/department/confidential）
  6. 检索测试面板（召回/过滤/重排/参数调试）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.config import Settings  # noqa: E402
from src.document_processor import DocumentProcessor  # noqa: E402
from src.embeddings import FakeEmbeddings  # noqa: E402
from src.knowledge_base import (  # noqa: E402
    DEFAULT_CATEGORIES,
    MEDICAL_SEPARATORS,
    KnowledgeBaseManager,
)
from src.rag_system import MedRAGSystem  # noqa: E402
from src.retriever import build_retriever  # noqa: E402
from src.vectorstore import VectorStoreManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_kb",
        data_dir=str(ROOT / "data" / "raw"),
        kb_registry_path=str(tmp_path / "kb" / "registry.json"),
        kb_upload_dir=str(tmp_path / "kb" / "uploads"),
        kb_candidate_buffer=15,
        kb_recall_k=10,
        kb_debug_top_k=3,
        enable_safety_guard=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_kb(tmp_path, **overrides):
    settings = make_settings(tmp_path, **overrides)
    vsm = VectorStoreManager(settings, FakeEmbeddings())
    return settings, vsm, KnowledgeBaseManager(settings, vsm, DocumentProcessor(settings))


def write_md(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------- 1. 上传 / 分块 ----------
def test_upload_and_incremental_ingest(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    f1 = write_md(
        tmp_path / "guide.md",
        "# 高血压指南\n\n## 诊断标准\n收缩压>=140mmHg。\n\n## 用药\n一线用药包括ACEI。",
    )
    rec1 = kb.upload_file(f1, category="临床指南", title="高血压指南", version="v1.0")
    assert rec1["doc_id"].startswith("doc-")
    assert rec1["status"] == "active"
    assert rec1["chunk_count"] >= 1
    assert vsm.exists()  # 已向量化并持久化

    # 增量入库：第二个文档不覆盖第一个
    f2 = write_md(tmp_path / "drug.txt", "二甲双胍用于2型糖尿病，eGFR<30禁用。")
    rec2 = kb.upload_file(f2, category="药学", title="糖尿病用药")
    assert rec2["doc_id"] != rec1["doc_id"]
    assert len(kb.documents) == 2
    assert len(list(vsm.store.docstore._dict.values())) == (
        rec1["chunk_count"] + rec2["chunk_count"]
    )


def test_medical_separators_used(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    f = write_md(
        tmp_path / "clause.txt",
        "第一条 患者收缩压大于等于140mmHg可诊断高血压。第二条 需排除继发性原因。第三条 起始剂量宜小。",
    )
    rec = kb.upload_file(
        f, title="条款文档", separators=["第", "。", "\n"], chunk_size=20, chunk_overlap=0
    )
    assert rec["chunk_count"] >= 2  # 按“第N条”拆开
    assert MEDICAL_SEPARATORS[0] == "\n第"


# ---------- 2. 批量导入 ----------
def test_batch_import(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    files = []
    for i in range(3):
        p = write_md(tmp_path / f"doc{i}.md", f"# 指南{i}\n\n内容要点{i}，诊断与治疗规范。")
        files.append((p, {"category": "内科", "title": f"指南{i}"}))
    results = kb.upload_batch(files)
    assert len(results) == 3
    assert all(r.get("status") == "active" for r in results)
    assert len(kb.documents) == 3
    assert kb.overview()["total_chunks"] == sum(r["chunk_count"] for r in results)


# ---------- 3. 分类目录 ----------
def test_categories(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    cats = kb.list_categories()
    for c in DEFAULT_CATEGORIES:
        assert c in cats
    assert kb.add_category("临床指南") is False  # 已存在
    assert kb.add_category("新增分类") is True
    assert "新增分类" in kb.list_categories()

    f = write_md(tmp_path / "a.md", "# 测试\n临床内容。")
    kb.upload_file(f, category="药学", title="药学资料")
    stats = {s["name"]: s["active_docs"] for s in kb.category_stats()}
    assert stats["药学"] == 1
    assert stats["临床指南"] == 0


# ---------- 4. 版本管理 ----------
def test_version_archive_and_invalid_filter(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    f = write_md(tmp_path / "bp.md", "# 高血压指南\n\n## 诊断标准\n旧标准：收缩压>=160mmHg。")
    v1 = kb.upload_file(f, category="临床指南", title="高血压指南", version="v1.0")

    # 新版入库：旧版自动归档失效
    f.write_text("# 高血压指南\n\n## 诊断标准\n新标准：收缩压>=140mmHg。", encoding="utf-8")
    v2 = kb.upload_file(f, category="临床指南", title="高血压指南", version="v2.0")

    assert kb.documents[v1["doc_id"]]["status"] == "archived"
    assert kb.documents[v2["doc_id"]]["status"] == "active"
    versions = kb.list_versions(title="高血压指南")
    assert len(versions) == 2
    assert versions[0]["version"] == "v2.0"

    # 检索默认过滤旧版：新标准的 chunk 被召回，旧标准被过滤
    hits = kb.filtered_similarity_search("收缩压诊断标准", k=5, category="临床指南")
    assert hits
    assert all(d.metadata.get("doc_status") == "active" for d in hits)
    assert all("140" not in d.page_content or d.metadata["doc_id"] == v2["doc_id"] for d in hits)

    # 标记失效后，该文档所有 chunk 都不再被召回
    kb.invalidate(v2["doc_id"])
    hits2 = kb.filtered_similarity_search("收缩压诊断标准", k=5, category="临床指南")
    assert all(d.metadata.get("doc_id") != v2["doc_id"] for d in hits2)

    # 重新激活恢复
    kb.reactivate(v2["doc_id"])
    assert kb.documents[v2["doc_id"]]["status"] == "active"


def test_default_retriever_filters_invalid(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)
    f = write_md(tmp_path / "x.md", "# 标题\n过期内容。")
    rec = kb.upload_file(f, title="过期文档")
    kb.invalidate(rec["doc_id"])
    retriever = build_retriever(settings, vsm)
    docs = retriever.invoke("过期内容")
    assert all(d.metadata.get("doc_id") != rec["doc_id"] for d in docs)


# ---------- 5. 权限隔离 ----------
def test_permission_isolation(tmp_path):
    settings, vsm, kb = make_kb(tmp_path)

    pub = write_md(tmp_path / "public.md", "# 全院通用指南\n所有人可查。")
    dept = write_md(tmp_path / "dept.md", "# 心内科资料\n科室内部学习。")
    secret = write_md(tmp_path / "secret.md", "# 涉密资料\n仅指定账号可检索。")

    kb.upload_file(pub, title="全院指南", visibility="public")
    kb.upload_file(dept, title="心内科资料", visibility="department", allowed_accounts=["心内科"])
    kb.upload_file(secret, title="涉密资料", visibility="confidential", allowed_accounts=["admin"])

    q = "指南 资料 学习"
    # 匿名：只能看 public
    anon = kb.filtered_similarity_search(q, k=10)
    assert anon and all(d.metadata.get("visibility") == "public" for d in anon)

    # 心内科账号：可看 public + department
    dept_hits = kb.filtered_similarity_search(q, k=10, department="心内科")
    assert any(d.metadata.get("visibility") == "department" for d in dept_hits)
    assert all(d.metadata.get("visibility") != "confidential" for d in dept_hits)

    # admin：可看 confidential
    admin_hits = kb.filtered_similarity_search(q, k=10, account="admin")
    assert any(d.metadata.get("visibility") == "confidential" for d in admin_hits)


def test_kb_query_with_acl(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    f = write_md(tmp_path / "s.md", "# 涉密指南\n只有admin能看到的药量限制内容。")
    system.kb.upload_file(f, title="涉密指南", visibility="confidential", allowed_accounts=["admin"])
    # 匿名查不到
    r1 = system.kb_query("药量限制", account=None)
    assert r1["source_documents"] == []
    # admin 可查
    r2 = system.kb_query("药量限制", account="admin")
    assert r2["source_documents"]


# ---------- 6. 检索测试面板 ----------
def test_debug_panel(tmp_path):
    settings, vsm, kb = make_kb(tmp_path, kb_recall_k=8)
    f = write_md(
        tmp_path / "panel.md",
        "# 临床指南\n\n## 诊断\n高血压诊断标准140。\n\n## 用药\nACEI为一线。",
    )
    kb.upload_file(f, category="临床指南", title="测试指南")
    kb.invalidate(kb.list_versions(title="测试指南")[0]["doc_id"])  # 先失效再重激活以观察过滤
    kb.reactivate(kb.list_versions(title="测试指南")[0]["doc_id"])

    report = kb.debug_retrieval(
        "高血压诊断标准",
        top_k=2,
        recall_k=8,
        use_rerank=True,
        chunk_size=100,
    )
    assert report["recall_count"] >= 1
    assert report["recall"]
    assert "score" in report["recall"][0]
    assert "status" in report["recall"][0]
    assert len(report["results"]) <= 2
    assert report["params"]["top_k"] == 2
    assert report["chunking_preview"]["chunk_size"] == 100
    assert report["chunking_preview"]["resulting_chunks"] >= 1

    # include_invalid=True 时召回环节可看到失效文档
    doc_id = kb.list_versions(title="测试指南")[0]["doc_id"]
    kb.invalidate(doc_id)
    report2 = kb.debug_retrieval("高血压", include_invalid=True, recall_k=10)
    assert any(x["status"] == "invalid" for x in report2["recall"])
    report3 = kb.debug_retrieval("高血压", include_invalid=False, recall_k=10)
    # 过滤后列表中不应含失效状态的结果
    assert all(r["doc_status"] != "invalid" for r in report3["results"])


# ---------- 顶层集成 & API ----------
def test_system_integration_and_overview(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    f = write_md(tmp_path / "i.md", "# 集成测试\n高血压指南内容。")
    system.kb.upload_file(f, category="临床指南", title="集成文档")
    ov = system.kb.overview()
    assert ov["total_documents"] == 1
    assert ov["active_documents"] == 1
    assert ov["vector_index_ready"] is True
    assert any(c["name"] == "临床指南" and c["active_docs"] == 1 for c in ov["categories"])


def test_api_routes_registered():
    from src.api import app

    routes = {r.path for r in app.routes}
    for p in [
        "/kb/overview", "/kb/categories", "/kb/upload", "/kb/upload/batch",
        "/kb/documents", "/kb/query", "/kb/debug",
    ]:
        assert p in routes
