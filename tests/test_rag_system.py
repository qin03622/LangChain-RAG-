"""RAG 系统集成测试（离线模式，无需真实 API Key）。

使用 FakeEmbeddings + FakeChatModel 验证完整链路：
加载文档 → 切分 → 构建索引 → 检索 → 生成 → 后处理。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.config import Settings  # noqa: E402
from src.document_processor import DocumentProcessor  # noqa: E402
from src.embeddings import FakeEmbeddings  # noqa: E402
from src.llm import FakeChatModel  # noqa: E402
from src.prompts import format_context  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402
from src.utils import safety_check, clean_text  # noqa: E402
from src.vectorstore import VectorStoreManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fake_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test",
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        retrieval_strategy="similarity",
        enable_safety_guard=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def sample_docs():
    return [
        Document(
            page_content=(
                "高血压诊断标准：非同日三次测量，收缩压>=140mmHg 或舒张压>=90mmHg 可诊断高血压。"
                "一线降压药包括利尿剂、CCB、ACEI、ARB 和 β 受体阻滞剂。"
            ),
            metadata={"source": "guide.md"},
        ),
        Document(
            page_content=(
                "2 型糖尿病一线用药为二甲双胍，糖化血红蛋白控制目标一般为 <7.0%。"
                "需定期筛查眼底与尿蛋白。"
            ),
            metadata={"source": "diabetes.md"},
        ),
    ]


# ---------- 单元级 ----------

def test_clean_text():
    assert clean_text("a  b\n\n\n\nc") == "a b\n\nc"


def test_safety_check_flags_high_risk():
    passed, warning = safety_check("我想自行停用降压药可以吗？", "可以停。")
    assert passed is False
    assert "高风险" in warning


def test_safety_check_normal():
    passed, warning = safety_check("高血压怎么治？", "请就医。")
    assert passed is True
    assert warning == ""


def test_document_processor_split():
    settings = fake_settings(Path("/tmp"))
    proc = DocumentProcessor(settings)
    chunks = proc.split_documents(sample_docs())
    assert len(chunks) >= 2
    assert all(c.metadata.get("chunk_id") for c in chunks)
    assert all(c.page_content.strip() for c in chunks)


def test_format_context():
    ctx = format_context(sample_docs())
    assert "[1]" in ctx
    assert "guide.md" in ctx


def test_fake_embeddings_dimension():
    emb = FakeEmbeddings(dimension=32)
    v = emb.embed_query("高血压")
    assert len(v) == 32
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


# ---------- 集成级 ----------

def test_vectorstore_build_and_search(tmp_path):
    settings = fake_settings(tmp_path)
    vsm = VectorStoreManager(settings, FakeEmbeddings())
    chunks = DocumentProcessor(settings).split_documents(sample_docs())
    vsm.build(chunks)
    vsm.save()
    assert vsm.exists()

    vsm2 = VectorStoreManager(settings, FakeEmbeddings())
    vsm2.load()
    results = vsm2.store.similarity_search("糖尿病用药", k=2)
    assert len(results) == 2


def test_full_rag_pipeline(tmp_path):
    settings = fake_settings(tmp_path)
    system = MedRAGSystem(settings)
    n = system.ingest_documents(sample_docs())
    assert n >= 2

    result = system.query("高血压的诊断标准是什么？")
    assert "answer" in result
    assert "测试模型响应" in result["answer"]
    assert result["source_documents"]
    assert result["safety_passed"] in (True, False)
    assert "仅供参考" in result["disclaimer"]

    rendered = system.format_result(result)
    assert "【回答】" in rendered
    assert "【参考来源】" in rendered


def test_query_empty_raises(tmp_path):
    system = MedRAGSystem(fake_settings(tmp_path))
    with pytest.raises(ValueError):
        system.query("")
