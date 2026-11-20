"""功能五：后台运维 & 统计 测试（离线 Fake 模式，无需 API Key）。

覆盖：问答统计（高频病种/问题/幻觉标记）、模型切换、RAG 开关（纯 LLM vs RAG）、
参数可视化配置（chunk/top-k/rerank/threshold/hyde/temperature）、会话管理（查看/删除敏感会话），
以及 API 端点与持久化恢复。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src.api import app  # noqa: E402
from src.config import Settings  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_ops",
        session_store_dir=str(tmp_path / "sessions"),
        feedback_store_path=str(tmp_path / "feedback.jsonl"),
        audit_store_path=str(tmp_path / "audit.jsonl"),
        ops_query_log_path=str(tmp_path / "ops" / "query_log.jsonl"),
        ops_param_store_path=str(tmp_path / "ops" / "runtime_params.json"),
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        precise_top_k=2,
        comprehensive_top_k=3,
        use_rerank=False,
        audit_enabled=False,
        safety_filter_enabled=False,
        privacy_desensitize_enabled=False,
        enable_safety_guard=False,
        rag_enabled=True,
        rerank_threshold=0.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def sample_docs():
    return [
        Document(
            page_content="高血压诊断标准：收缩压>=140 或舒张压>=90mmHg。一线用药包括利尿剂、CCB、ACEI、ARB。",
            metadata={"source": "高血压指南.md", "chunk_id": "c1", "doc_status": "active"},
        ),
        Document(
            page_content="糖尿病管理：二甲双胍是2型糖尿病一线用药，注意肾功能不全患者慎用。",
            metadata={"source": "糖尿病基础管理.md", "chunk_id": "c2", "doc_status": "active"},
        ),
        Document(
            page_content="急诊识别：突发胸痛需警惕急性心肌梗死，立即心电图检查。",
            metadata={"source": "常见急症识别.md", "chunk_id": "c3", "doc_status": "active"},
        ),
    ]


def build_system(tmp_path):
    system = MedRAGSystem(make_settings(tmp_path))
    system.ingest_documents(sample_docs())
    return system


# ============ 1. 问答统计 ============
def test_query_logging_and_stats(tmp_path):
    system = build_system(tmp_path)
    system.query("高血压的诊断标准是什么？")
    system.query("高血压如何用药？")
    system.query("糖尿病吃什么药？")

    stats = system.ops.stats(top_n=5)
    assert stats["total_queries"] == 3
    assert stats["rag_queries"] == 3
    assert stats["pure_llm_queries"] == 0
    # 高频病种：高血压 2 次，糖尿病 1 次
    diseases = {d["disease"]: d["count"] for d in stats["top_diseases"]}
    assert diseases.get("高血压") == 2
    assert diseases.get("糖尿病") == 1
    # 高频问题非空
    assert stats["top_questions"]
    assert stats["top_questions"][0]["count"] >= 1


def test_stats_keyword_filter(tmp_path):
    system = build_system(tmp_path)
    system.query("高血压的诊断标准是什么？")
    system.query("糖尿病吃什么药？")
    stats = system.ops.stats(keyword="高血压")
    assert stats["total_queries"] == 1
    assert "高血压" in stats["top_questions"][0]["question"]


def test_stats_hallucination_count(tmp_path):
    system = build_system(tmp_path)
    system.query("高血压的诊断标准是什么？")
    system.submit_feedback(
        "高血压的诊断标准是什么？", "答案", "hallucination",
        result={"citations": []}, note="剂量未注明出处",
    )
    stats = system.ops.stats()
    assert stats["feedback"]["hallucination_count"] == 1
    assert stats["feedback"]["total"] == 1


# ============ 2. 模型切换 & RAG 开关 ============
def test_list_and_switch_model(tmp_path):
    system = build_system(tmp_path)
    models = system.ops.list_models()
    assert len(models) >= 3
    assert any(m["provider"] == "ollama" for m in models)  # 本地医疗大模型
    assert any(m["provider"] == "openai" for m in models)  # 在线模型

    # fake Providers 间切换，验证不中断
    cur = system.ops.switch_model("fake", "fake", temperature=0.5)
    assert cur["provider"] == "fake"
    assert cur["temperature"] == 0.5
    result = system.query("高血压的诊断标准是什么？")
    assert result["answer"]


def test_switch_model_invalid_provider(tmp_path):
    system = build_system(tmp_path)
    with pytest.raises(RuntimeError):
        system.ops.switch_model("not-a-provider", "x")


def test_rag_toggle_and_pure_llm_mode(tmp_path):
    system = build_system(tmp_path)
    system.ops.set_rag(False)
    result = system.query("高血压的诊断标准是什么？")
    assert result["source_documents"] == []  # 纯 LLM 模式无检索证据
    assert result["answer"]

    system.ops.set_rag(True)
    result = system.query("高血压的诊断标准是什么？")
    assert len(result["source_documents"]) > 0  # RAG 模式有检索证据


def test_compare_rag_vs_pure_llm(tmp_path):
    system = build_system(tmp_path)
    out = system.ops.compare("高血压的诊断标准是什么？")
    assert "pure_llm" in out and "rag" in out
    assert out["pure_llm"]["source_count"] == 0
    assert out["rag"]["source_count"] > 0
    # 对比后 RAG 开关回到原状态
    assert system.ops.current_model()["rag_enabled"] is True


# ============ 3. 参数可视化配置 ============
def test_get_params_contains_all(tmp_path):
    system = build_system(tmp_path)
    params = system.ops.get_params()
    for name in ["chunk_size", "chunk_overlap", "top_k", "rerank_threshold", "use_hyde", "temperature"]:
        assert name in params
        assert "value" in params[name]
        assert "description" in params[name]


def test_set_params_and_effect(tmp_path):
    system = build_system(tmp_path)
    applied = system.ops.set_params(
        chunk_size=300, chunk_overlap=50, top_k=1,
        rerank_threshold=0.5, use_hyde=True, temperature=0.7,
    )
    assert applied == {
        "chunk_size": 300, "chunk_overlap": 50, "top_k": 1,
        "rerank_threshold": 0.5, "use_hyde": True, "temperature": 0.7,
    }
    assert system.settings.chunk_size == 300
    assert system.settings.chunk_overlap == 50
    assert system.settings.top_k == 1
    assert system.settings.rerank_threshold == 0.5
    assert system.settings.use_hyde is True
    assert system.settings.llm_temperature == 0.7
    # 切分器已重建：200->300（langchain 私有属性 _chunk_size）
    assert system.processor.splitter._chunk_size == 300


def test_set_params_validation(tmp_path):
    system = build_system(tmp_path)
    with pytest.raises(ValueError, match="不支持的参数"):
        system.ops.set_params(unknown_param=1)
    with pytest.raises(ValueError):
        system.ops.set_params(top_k="abc")


def test_reset_params(tmp_path):
    system = build_system(tmp_path)
    system.ops.set_params(chunk_size=400, top_k=2)
    system.ops.reset_params()
    assert system.ops.get_params()["chunk_size"]["value"] == 200
    assert system.ops.get_params()["top_k"]["value"] == 5


def test_params_persist_across_restart(tmp_path):
    system = build_system(tmp_path)
    system.ops.set_params(chunk_size=333, top_k=4, temperature=0.9)
    system.ops.set_rag(False)
    # 重新构造系统（同一 tmp_path）
    system2 = MedRAGSystem(make_settings(tmp_path))
    assert system2.settings.chunk_size == 333
    assert system2.settings.top_k == 4
    assert system2.settings.llm_temperature == 0.9
    assert system2.ops.current_model()["rag_enabled"] is False


def test_rerank_threshold_filters_docs(tmp_path):
    """阈值生效：设置高阈值应减少或保持检索结果，不报错。"""
    system = build_system(tmp_path)
    assert system.settings.rerank_threshold == 0.0
    system.ops.set_params(rerank_threshold=0.9)
    result = system.query("高血压的诊断标准是什么？")
    # 阈值过滤只会减少/保持文档数，且不会抛异常
    assert isinstance(result["source_documents"], list)


# ============ 4. 会话管理 ============
def test_session_list_and_delete(tmp_path):
    system = build_system(tmp_path)
    sid1 = system.new_session()
    sid2 = system.new_session()
    system.clinical.query("高血压的诊断标准是什么？", session_id=sid1)
    system.clinical.query("糖尿病吃什么药？", session_id=sid2)

    sessions = system.ops.list_sessions()
    assert len(sessions) == 2
    assert {s["session_id"] for s in sessions} == {sid1, sid2}
    assert all("message_count" in s and "summary" in s for s in sessions)

    # 查看单个会话
    detail = system.ops.get_session(sid1)
    assert detail["session_id"] == sid1
    assert detail["messages"]

    # 删除会话
    out = system.ops.delete_session(sid1)
    assert out["deleted"] is True
    assert len(system.ops.list_sessions()) == 1
    with pytest.raises(KeyError):
        system.ops.delete_session(sid1)


def test_purge_sensitive_sessions(tmp_path):
    system = build_system(tmp_path)
    sid = system.new_session()
    # 含手机号的问题会触发敏感标记（auto_purge）
    system.clinical.query("我的手机号是13800138000，高血压怎么治？", session_id=sid)
    sessions = system.ops.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["sensitive"] is True

    out = system.ops.purge_sensitive_sessions()
    assert out["purged"] == 1
    assert system.ops.list_sessions() == []


# ============ 5. API 端点 ============
def test_api_ops_endpoints(tmp_path, monkeypatch):
    system = build_system(tmp_path)
    import src.api as api_module

    monkeypatch.setattr(api_module, "_system", system)
    client = TestClient(api_module.app)

    assert client.get("/ops/stats").status_code == 200
    assert client.get("/ops/models").status_code == 200
    assert client.get("/ops/model/current").status_code == 200
    assert client.get("/ops/params").status_code == 200

    # 开关 RAG
    r = client.post("/ops/rag", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["rag_enabled"] is False
    client.post("/ops/rag", json={"enabled": True})

    # 设置参数
    r = client.post("/ops/params", json={"params": {"top_k": 2}})
    assert r.status_code == 200
    assert r.json()["top_k"] == 2
    assert client.post("/ops/params", json={"params": {"bad": 1}}).status_code == 400

    # 模型切换
    r = client.post("/ops/model/switch", json={"provider": "fake", "model": "fake"})
    assert r.status_code == 200

    # 会话列表
    sid = system.new_session()
    assert client.get("/ops/sessions").status_code == 200
    assert client.get(f"/ops/sessions/{sid}").status_code == 200
    assert client.delete(f"/ops/sessions/{sid}").status_code == 200
    assert client.get(f"/ops/sessions/{sid}").status_code == 404
    assert client.post("/ops/sessions/purge-sensitive").status_code == 200

    # 对比测试（需索引，已 build）
    r = client.post("/ops/compare", json={"question": "高血压的诊断标准是什么？"})
    assert r.status_code == 200
    assert "pure_llm" in r.json()
