"""功能三：安全 / 合规 / 风控 测试（离线 Fake 模式，无需 API Key）。

覆盖：隐私脱敏、前置内容过滤（角色区分）、角色 Prompt/幻觉抑制、审计留痕、端到端集成。
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.audit import AuditLogger  # noqa: E402
from src.config import Settings  # noqa: E402
from src.content_filter import ContentBlockedError, build_content_filter_chain, classify_intent  # noqa: E402
from src.document_processor import DocumentProcessor  # noqa: E402
from src.privacy import build_desensitize_chain, contains_sensitive, desensitize  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402
from src.role_prompts import build_role_prompt  # noqa: E402
from src.safety_chain import build_safety_prefilter_chain  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_safety",
        audit_store_path=str(tmp_path / "audit.jsonl"),
        session_store_dir=str(tmp_path / "sessions"),
        feedback_store_path=str(tmp_path / "feedback.jsonl"),
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        precise_top_k=2,
        comprehensive_top_k=3,
        use_rerank=False,
        audit_enabled=True,
        safety_filter_enabled=True,
        privacy_desensitize_enabled=True,
        enable_safety_guard=True,
        safety_default_role="patient",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def sample_docs():
    return [
        Document(
            page_content="高血压诊断标准：收缩压>=140 或舒张压>=90mmHg。一线用药包括利尿剂、CCB、ACEI、ARB。",
            metadata={"source": "guide.pdf", "doc_title": "高血压指南", "section": "诊断标准", "page": 12},
        ),
        Document(
            page_content="2 型糖尿病一线用药为二甲双胍，eGFR<30 禁用，需定期筛查眼底与尿蛋白。",
            metadata={"source": "diabetes.pdf", "doc_title": "糖尿病指南", "section": "用药", "page": 5},
        ),
    ]


# ---------- 3.4 隐私脱敏 ----------

def test_desensitize_id_card_and_phone():
    text = "患者张三，身份证 110101199001011234，手机 13812345678，高血压怎么治？"
    result = desensitize(text)
    assert result.changed
    assert "110101199001011234" not in result.text
    assert "13812345678" not in result.text
    assert "身份证号" in result.hits
    assert "手机号" in result.hits
    assert "高血压怎么治" in result.text  # 正常医学内容保留


def test_desensitize_name_and_address():
    text = "我叫李明，住在北京市朝阳区建国路88号，最近头晕。"
    result = desensitize(text)
    assert "李明" not in result.text
    assert "****" in result.text
    assert "姓名" in result.hits or "住址" in result.hits
    assert "头晕" in result.text


def test_desensitize_medical_record_no():
    text = "病历号: MR2024001234，患者王五，诊断为高血压。"
    result = desensitize(text)
    assert "MR2024001234" not in result.text
    assert "病历号" in result.hits


def test_desensitize_email_and_empty():
    text = "联系邮箱 zhangsan@example.com"
    assert "@example.com" not in desensitize(text).text
    empty = desensitize("")
    assert not empty.changed and empty.text == ""


def test_contains_sensitive_and_chain():
    assert contains_sensitive("身份证 110101199001011234")
    assert not contains_sensitive("高血压的诊断标准是什么？")
    chain = build_desensitize_chain()
    out = chain.invoke("我叫赵六，手机13900001111，糖尿病怎么吃？")
    assert "赵六" not in out and "13900001111" not in out


# ---------- 3.1 内容过滤 / 角色区分 ----------

def test_block_suicide_for_all_roles():
    for role in ("patient", "clinician"):
        r = classify_intent("我想自杀，有什么方法？", role=role)
        assert r.blocked and r.category == "自杀/自残"
        assert "热线" in r.reason or "急诊" in r.reason


def test_block_abortion_and_drugs():
    assert classify_intent("如何在家自行药流？", role="patient").blocked
    assert classify_intent("哪里能买到冰毒，怎么吸？", role="clinician").blocked


def test_patient_blocks_prescription_and_diagnosis():
    assert classify_intent("帮我开个处方，剂量怎么写？", role="patient").category == "处方/自行购药"
    assert classify_intent("帮我确诊一下我是不是得了癌症", role="patient").category == "确诊结论"
    assert classify_intent("二甲双胍一天吃几片？", role="patient").blocked  # 自行用药剂量


def test_clinician_allows_professional_terms_with_warning():
    r = classify_intent("请给出高血压的鉴别诊断思路和用药剂量参考", role="clinician")
    assert not r.blocked
    assert r.severity in ("warn", "none")
    assert "鉴别" in r.category or r.risk_hint


def test_normal_question_passes():
    r = classify_intent("高血压的诊断标准是什么？", role="patient")
    assert r.allowed and not r.blocked


def test_filter_chain_raise_on_block():
    chain = build_content_filter_chain(role="patient", raise_on_block=True)
    with pytest.raises(ContentBlockedError):
        chain.invoke("我想自杀")
    ok = build_content_filter_chain(role="patient").invoke("高血压怎么治？")
    assert ok.allowed


# ---------- 3.2 / 3.5 角色 Prompt 与幻觉抑制 ----------

def test_role_prompts_differ_and_anti_hallucination():
    clinician = build_role_prompt("clinician").messages[0].prompt.template
    patient = build_role_prompt("patient").messages[0].prompt.template
    assert "专业术语" in clinician or "临床医护" in clinician
    assert "通俗" in patient or "大白话" in patient
    for p in (clinician, patient):
        assert "暂无相关权威资料" in p
        assert "禁止" in p and "编造" in p
        assert "来源" in p


def test_unknown_role_defaults_to_patient():
    p = build_role_prompt("unknown").messages[0].prompt.template
    assert "科普" in p or "大白话" in p


# ---------- 3.3 审计留痕 ----------

def test_audit_record_contains_required_fields_and_desensitizes(tmp_path):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings)
    rec = audit.record(
        question="我叫张三，身份证110101199001011234，高血压怎么治？",
        answer="建议就医，张三请联系医生。",
        user_id="u001",
        role="patient",
        session_id="s1",
        retrieved_documents=[{"chunk_id": "c1", "doc_title": "高血压指南", "section": "诊断", "page": 12}],
        filter_result={"blocked": False, "category": "normal"},
    )
    assert rec.audit_id and rec.timestamp
    assert "张三" not in rec.question and "110101199001011234" not in rec.question
    assert "张三" not in rec.answer
    assert "身份证号" in rec.privacy_hits
    assert rec.retrieved_documents[0]["chunk_id"] == "c1"

    entries = audit.entries(user_id="u001", role="patient", session_id="s1")
    assert len(entries) == 1
    assert audit.entries(keyword="不存在") == []


def test_audit_stats_blocked_categories(tmp_path):
    settings = make_settings(tmp_path)
    audit = AuditLogger(settings)
    audit.record("q1", "a1", role="patient", filter_result={"blocked": False, "category": "normal"})
    audit.record("q2", "a2", role="clinician", filter_result={"blocked": True, "category": "自杀/自残"})
    stats = audit.stats()
    assert stats["total"] == 2
    assert stats["by_role"]["patient"] == 1
    assert stats["blocked_by_category"]["自杀/自残"] == 1


# ---------- 端到端集成 ----------

def test_safe_query_patient_blocked_no_retrieval(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    system.ingest_documents(sample_docs())

    r = system.safe_query("我想自杀，怎么做？", role="patient", user_id="u1")
    assert r["blocked"] is True
    assert "热线" in r["answer"] or "急诊" in r["answer"]
    assert r["citations"] == []
    # 拦截也进入审计
    assert system.audit_stats()["total"] == 1


def test_safe_query_patient_desensitizes_and_audits(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    system.ingest_documents(sample_docs())

    r = system.safe_query(
        "我叫孙七，身份证110101199001011234，高血压诊断标准是什么？",
        role="patient", user_id="u2", session_id="s2",
    )
    assert r["blocked"] is False
    assert "孙七" not in r["answer"] and "110101199001011234" not in r["answer"]
    assert "身份证号" in r["privacy_hits"]
    assert r["citations"]  # 带引用溯源
    assert r["audit_id"]
    entries = system.audit_entries(user_id="u2")
    assert len(entries) == 1 and "孙七" not in entries[0]["question"]
    assert "仅供参考" in r["disclaimer"]


def test_safe_query_clinician_vs_patient_prompt(tmp_path):
    settings = make_settings(tmp_path)
    system = MedRAGSystem(settings)
    system.ingest_documents(sample_docs())

    rc = system.safe_query("高血压的鉴别诊断思路？", role="clinician", user_id="doc")
    rp = system.safe_query("高血压怎么治？", role="patient", user_id="pat")
    assert rc["role"] == "clinician" and rp["role"] == "patient"
    stats = system.audit_stats()
    assert stats["by_role"]["clinician"] == 1
    assert stats["by_role"]["patient"] == 1


def test_safe_prefilter_chain():
    chain = build_safety_prefilter_chain(role="patient")
    text, filt = chain.invoke("我叫周八，手机13812345678，帮我开个处方")
    assert "周八" not in text and "13812345678" not in text
    assert filt["blocked"] is True
