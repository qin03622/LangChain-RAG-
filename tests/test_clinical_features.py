"""临床增强功能测试（离线 Fake 模式，无需 API Key）。

覆盖：引用溯源、双模式检索、多轮会话记忆（TTL/敏感清理）、反馈调优。
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.citation import build_citations, get_original_fragment, render_citations_markdown  # noqa: E402
from src.clinical_retriever import ClinicalRetriever  # noqa: E402
from src.config import Settings  # noqa: E402
from src.document_processor import DocumentProcessor  # noqa: E402
from src.embeddings import FakeEmbeddings  # noqa: E402
from src.feedback import FeedbackStore  # noqa: E402
from src.llm import FakeChatModel  # noqa: E402
from src.memory import SessionMemory  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_clinical",
        session_store_dir=str(tmp_path / "sessions"),
        feedback_store_path=str(tmp_path / "feedback.jsonl"),
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        precise_top_k=2,
        comprehensive_top_k=3,
        clinical_mode="precise",
        use_hyde=False,
        use_multi_query=False,
        use_rerank=False,
        memory_enabled=True,
        memory_ttl_seconds=3600,
        memory_auto_purge_sensitive=True,
        feedback_enabled=True,
        feedback_tuning_enabled=True,
        enable_safety_guard=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def sample_docs():
    return [
        Document(
            page_content="高血压诊断标准：收缩压>=140 或舒张压>=90mmHg。指南原文需核对。",
            metadata={"source": "guide.pdf", "doc_title": "高血压指南", "section": "诊断标准", "page": 12},
        ),
        Document(
            page_content="二甲双胍是 2 型糖尿病一线用药，eGFR<30 禁用。",
            metadata={"source": "diabetes.pdf", "doc_title": "糖尿病指南", "section": "用药", "page": 5},
        ),
    ]


# ---------- 引用溯源 ----------

def test_citations_have_title_section_page():
    settings = make_settings(Path("/tmp"))
    chunks = DocumentProcessor(settings).split_documents(sample_docs())
    citations = build_citations(chunks)
    assert citations
    c = citations[0]
    assert c.doc_title and c.chunk_id
    assert c.index == 1
    md = render_citations_markdown(citations)
    assert "点击查看原文" in md
    assert "```text" in md


def test_original_fragment_lookup():
    settings = make_settings(Path("/tmp"))
    chunks = DocumentProcessor(settings).split_documents(sample_docs())
    cid = chunks[0].metadata["chunk_id"]
    frag = get_original_fragment(chunks, cid)
    assert frag and frag["chunk_id"] == cid
    assert frag["full_content"]
    assert get_original_fragment(chunks, "not-exist") is None


# ---------- 双模式检索 ----------

def test_precise_vs_comprehensive_top_k(tmp_path):
    settings = make_settings(tmp_path)
    vsm = __import__("src.vectorstore", fromlist=["VectorStoreManager"]).VectorStoreManager(
        settings, FakeEmbeddings()
    )
    chunks = DocumentProcessor(settings).split_documents(sample_docs() * 4)
    vsm.build(chunks)
    retriever = ClinicalRetriever(settings, FakeChatModel(), vsm)

    precise = retriever.retrieve("高血压怎么诊断？", mode="precise")
    comprehensive = retriever.retrieve("高血压怎么诊断？", mode="comprehensive")
    assert len(precise) <= settings.precise_top_k
    assert len(comprehensive) <= settings.comprehensive_top_k


# ---------- 会话记忆 ----------

def test_memory_multi_turn_and_ttl(tmp_path):
    settings = make_settings(tmp_path, memory_ttl_seconds=1)
    mem = SessionMemory(settings, FakeChatModel())
    sid = mem.new_session()
    mem.add_turn(sid, "患者老年、糖尿病、肌酐升高", "已了解背景")
    mem.add_turn(sid, "那怎么选药？", "建议...")
    history = mem.get_history(sid)
    assert len(history) == 4
    assert "肌酐升高" in mem.get_context_text(sid)
    time.sleep(1.1)
    assert mem.get(sid) is None  # 过期自动清理


def test_memory_sensitive_auto_purge(tmp_path):
    settings = make_settings(tmp_path)
    mem = SessionMemory(settings, FakeChatModel())
    sid = mem.new_session()
    mem.add_turn(sid, "我的身份证号是 123456789012345678，请记住", "好的")
    session = mem.get(sid)
    assert session and session.sensitive is True
    assert not (mem.store_path / f"{sid}.json").exists()  # 敏感会话不落盘
    mem.clear(sid)
    assert mem.get(sid) is None


# ---------- 反馈 ----------

def test_feedback_stats_and_tuning(tmp_path):
    settings = make_settings(tmp_path)
    store = FeedbackStore(settings)
    store.add("q1", "a1", "correct", chunk_ids=["c1"])
    store.add("q2", "a2", "hallucination", chunk_ids=["c1", "c2"])
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["by_label"]["hallucination"] == 1
    q = store.chunk_quality()
    assert q["c2"] < q.get("c3", 1.0)
    assert "幻觉" in store.tuning_hint()

    d1 = Document(page_content="q", metadata={"chunk_id": "c1"})
    d2 = Document(page_content="q", metadata={"chunk_id": "c2"})
    ranked = store.rerank_by_quality([d2, d1])
    assert ranked[0].metadata["chunk_id"] == "c1"


# ---------- 顶层集成 ----------

def test_full_clinical_query(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    system.ingest_documents(sample_docs())

    sid = system.new_session()
    r1 = system.clinical_query("高血压诊断标准？", mode="precise", session_id=sid)
    assert r1["mode"] == "precise"
    assert r1["citations"]
    assert "[来源" in r1["answer"] or r1["citations"]
    assert "【引用来源" in system.clinical.format_result(r1)

    # 多轮：带上病史背景后继续提问
    system.memory.add_turn(sid, "患者老年、糖尿病、肌酐升高", "已知")
    r2 = system.clinical_query("那如何选药？", session_id=sid)
    assert r2["session_id"] == sid

    # 反馈
    fb = system.submit_feedback("q", "a", "incomplete", result=r1)
    assert fb and fb.get("id") and fb.get("label") == "incomplete"
    assert system.feedback_stats()["total"] == 1

    # 原文跳转
    cid = r1["citations"][0]["chunk_id"]
    assert system.get_original(r1, cid) is not None
