"""统一管理后端（/admin）测试：鉴权 + 五个功能的管理端点 + 操作审计。

离线 Fake 模式，无需真实 API Key。测试策略：
  - monkeypatch src.admin_api.get_settings 返回带固定 ADMIN_API_KEY 的测试配置；
  - monkeypatch src.api._system 注入隔离的 MedRAGSystem；
  - 所有 /admin 请求携带 Authorization: Bearer <测试密钥>。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import src.admin_api as admin_api_module  # noqa: E402
import src.api as api_module  # noqa: E402
from src.config import Settings  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ADMIN_KEY = "test-admin-key-1234"


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_admin",
        session_store_dir=str(tmp_path / "sessions"),
        feedback_store_path=str(tmp_path / "feedback.jsonl"),
        audit_store_path=str(tmp_path / "audit.jsonl"),
        ops_query_log_path=str(tmp_path / "ops" / "query_log.jsonl"),
        ops_param_store_path=str(tmp_path / "ops" / "runtime_params.json"),
        kb_registry_path=str(tmp_path / "kb" / "registry.json"),
        kb_upload_dir=str(tmp_path / "kb" / "uploads"),
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        precise_top_k=2,
        comprehensive_top_k=3,
        use_rerank=False,
        audit_enabled=True,
        safety_filter_enabled=True,
        privacy_desensitize_enabled=True,
        enable_safety_guard=False,
        rag_enabled=True,
        rerank_threshold=0.0,
        llm_api_key="sk-secret-real-key-9876",  # 验证掩码
        admin_api_key=ADMIN_KEY,
        admin_action_log_path=str(tmp_path / "admin" / "action_log.jsonl"),
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
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """构造隔离环境：测试 system + 测试 settings + 注入全局依赖。"""
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    system.ingest_documents(sample_docs())

    monkeypatch.setattr(api_module, "_system", system)
    monkeypatch.setattr(admin_api_module, "get_settings", lambda: settings)

    client = TestClient(api_module.app)
    headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
    return client, headers, system, settings


# ============ 0. 管理后台网页 ============
def test_admin_ui_served():
    client = TestClient(api_module.app)
    r = client.get("/admin/ui")
    assert r.status_code == 200
    assert "MedRAG · 统一管理后台" in r.text
    assert "/admin" in r.text  # 前端通过 /admin/* API 取数
    assert client.get("/admin/ui/", follow_redirects=False).status_code == 307


# ============ 1. 鉴权 ============
def test_admin_requires_key(env):
    client, _, _, _ = env
    assert client.get("/admin/overview").status_code == 401
    assert client.get(
        "/admin/overview", headers={"Authorization": "Bearer wrong-key"}
    ).status_code == 403


def test_admin_rejects_when_no_key_configured(env, monkeypatch):
    client, _, _, settings = env
    settings.admin_api_key = ""
    assert client.get(
        "/admin/overview", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
    ).status_code == 503


# ============ 2. 总览与功能目录 ============
def test_admin_overview(env):
    client, headers, _, _ = env
    r = client.get("/admin/overview", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    data = body["data"]
    assert data["app"] == "MedRAG 统一管理后端"
    assert set(data["features"]) == {"rag", "specialty", "safety", "kb", "ops"}
    assert [f["id"] for f in data["features_catalog"]] == [
        "rag", "specialty", "safety", "kb", "ops"
    ]
    assert data["system"]["index_ready"] is True


def test_admin_features_catalog(env):
    client, headers, _, _ = env
    r = client.get("/admin/features", headers=headers)
    assert r.status_code == 200
    assert [f["id"] for f in r.json()["data"]["features"]] == [
        "rag", "specialty", "safety", "kb", "ops"
    ]


def test_admin_unknown_feature_returns_400(env):
    client, headers, _, _ = env
    assert client.get("/admin/features/nope", headers=headers).status_code == 400


# ============ 3. 功能状态 / 统计 / 配置 ============
@pytest.mark.parametrize("feature", ["rag", "specialty", "safety", "kb", "ops"])
def test_admin_feature_status_and_stats(env, feature):
    client, headers, _, _ = env
    s = client.get(f"/admin/features/{feature}", headers=headers)
    assert s.status_code == 200, s.text
    assert s.json()["ok"] is True

    st = client.get(f"/admin/features/{feature}/stats", headers=headers)
    assert st.status_code == 200, st.text
    assert st.json()["ok"] is True


def test_admin_feature_config_masks_secrets(env):
    client, headers, _, _ = env
    r = client.get("/admin/features/rag/config", headers=headers)
    assert r.status_code == 200
    cfg = r.json()["data"]["config"]
    key = cfg["llm_api_key"]
    assert key["sensitive"] is True
    assert key["value"].endswith("9876") and key["value"].startswith("*")  # 只显示尾号
    assert len(key["value"]) == len("sk-secret-real-key-9876")  # 掩码保持等长
    assert "sk-secret" not in str(cfg)
    # 非敏感值原样可见
    assert cfg["llm_provider"]["value"] == "fake"
    assert cfg["llm_provider"]["sensitive"] is False


# ============ 4. 配置更新（白名单 + 掩码 + 联动 + 审计） ============
def test_admin_update_config_whitelist_and_audit(env):
    client, headers, _, settings = env
    # 非法字段被拒
    bad = client.post(
        "/admin/features/rag/config",
        headers=headers,
        json={"updates": {"not_a_config": 1}},
    )
    assert bad.status_code == 400
    assert "不支持" in bad.json()["detail"]

    # 合法更新：类型自动转换
    r = client.post(
        "/admin/features/rag/config",
        headers=headers,
        json={"updates": {"top_k": 9, "use_hyde": "true"}},
    )
    assert r.status_code == 200, r.text
    applied = r.json()["data"]["applied"]
    assert applied["top_k"] == 9
    assert applied["use_hyde"] is True
    assert settings.top_k == 9
    assert settings.use_hyde is True

    # 敏感字段可更新但只回显掩码
    r = client.post(
        "/admin/features/rag/config",
        headers=headers,
        json={"updates": {"llm_api_key": "sk-new-secret-4321"}},
    )
    masked = r.json()["data"]["applied"]["llm_api_key"]
    assert masked.endswith("4321") and masked.startswith("*")
    assert settings.llm_api_key == "sk-new-secret-4321"

    # 操作日志记录 actor
    logs = client.get("/admin/actions", headers=headers).json()["data"]["actions"]
    assert any(a["action"] == "update_config" and a["feature"] == "rag" for a in logs)
    assert any(a["actor"].startswith("key:") and a["actor"].endswith("1234") for a in logs)


def test_admin_update_config_type_error(env):
    client, headers, _, _ = env
    bad = client.post(
        "/admin/features/rag/config",
        headers=headers,
        json={"updates": {"top_k": "not-an-int"}},
    )
    assert bad.status_code == 400


# ============ 5. 功能一：重建索引 ============
def test_admin_rebuild_index(env):
    client, headers, _, _ = env
    r = client.post("/admin/actions/rebuild-index", headers=headers, json={})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["chunks"] > 0
    assert r.json()["data"]["index_ready"] is True


# ============ 6. 功能二：专项辅助 ============
def test_admin_specialty_operations_and_knowledge(env):
    client, headers, _, _ = env
    ops = client.get("/admin/specialty/operations", headers=headers).json()["data"]["operations"]
    assert len(ops) == 9
    assert {o["operation"] for o in ops} == {
        "differential", "lab_interpret", "drug_info", "drug_interactions",
        "guideline_search", "document_summarize", "document_template",
        "pathway", "allergy_risk",
    }

    kb = client.get("/admin/specialty/knowledge", headers=headers).json()["data"]
    assert "tables" in kb
    assert "note" in kb


# ============ 7. 功能三：安全审计 ============
def test_admin_safety_audit(env):
    client, headers, system, _ = env
    system.safe_query("高血压的诊断标准是什么？")  # 安全链写审计
    r = client.post("/admin/safety/audit/query", headers=headers, json={"limit": 10})
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["data"]["entries"], list)

    stats = client.get("/admin/safety/audit/stats", headers=headers)
    assert stats.status_code == 200
    assert stats.json()["data"]["total"] >= 1


# ============ 8. 功能四：知识库 ============
def test_admin_kb_documents_status_debug(env):
    client, headers, system, _ = env
    # 通过 KB manager 添加一份文档
    doc = system.kb.upload_file(
        path=str(ROOT / "data" / "raw" / "高血压指南.md"),
        title="高血压管理指南",
        category="临床指南",
    ) if (ROOT / "data" / "raw" / "高血压指南.md").exists() else None
    # 若示例文件不存在，直接操作 documents 字典（测试环境允许）
    if doc is None:
        system.kb.documents["doc-1"] = {
            "id": "doc-1", "title": "测试文档",
            "status": "active", "category": "临床指南",
        }

    docs = client.get("/admin/kb/documents", headers=headers).json()["data"]["documents"]
    assert len(docs) >= 1

    # 状态变更
    doc_id = docs[0]["id"]
    r = client.post(
        f"/admin/kb/documents/{doc_id}/status", headers=headers, json={"status": "archived"}
    )
    assert r.status_code == 200, r.text

    # 过滤
    filtered = client.get(
        "/admin/kb/documents", headers=headers, params={"status": "archived"}
    ).json()["data"]["documents"]
    assert any(d["id"] == doc_id for d in filtered)

    # 检索调试
    r = client.post(
        "/admin/kb/debug", headers=headers, json={"question": "高血压用药", "top_k": 2}
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


# ============ 9. 功能五：运维 ============
def test_admin_ops_endpoints(env):
    client, headers, system, _ = env

    # 模型切换
    r = client.post(
        "/admin/ops/model/switch", headers=headers,
        json={"provider": "fake", "model": "fake", "temperature": 0.5},
    )
    assert r.status_code == 200, r.text
    assert r.json()["data"]["provider"] == "fake"

    # RAG 开关（同时联动 settings.rag_enabled）
    r = client.post("/admin/ops/rag", headers=headers, json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["data"]["rag_enabled"] is False

    # 参数：合法 + 非法
    r = client.post("/admin/ops/params", headers=headers, json={"params": {"top_k": 3}})
    assert r.status_code == 200, r.text
    bad = client.post("/admin/ops/params", headers=headers, json={"params": {"bad_param": 1}})
    assert bad.status_code == 400

    r = client.post("/admin/ops/params/reset", headers=headers, json={})
    assert r.status_code == 200

    # 会话管理
    sid = system.new_session()
    assert client.get("/admin/ops/sessions", headers=headers).status_code == 200
    assert client.delete(f"/admin/ops/sessions/{sid}", headers=headers).status_code == 200
    assert client.post(
        "/admin/ops/sessions/purge-sensitive", headers=headers, json={}
    ).status_code == 200

    # 操作日志涵盖各功能
    logs = client.get("/admin/actions", headers=headers,
                      params={"limit": 100}).json()["data"]["actions"]
    actions = {a["action"] for a in logs}
    for expected in ("switch_model", "set_rag", "set_params", "reset_params",
                     "delete_session", "purge_sensitive"):
        assert expected in actions
