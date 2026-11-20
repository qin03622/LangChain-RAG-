"""功能二：面向临床医护的专项辅助功能测试（离线 Fake 模式，无需 API Key）。

覆盖：7 大功能 / 9 个操作、强制免责声明、诊断式表述警示、
结构化知识来源标注、版本区分、联用风险筛查、文书法律边界。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.rag_system import MedRAGSystem  # noqa: E402
from src.specialty_assistant import ClinicalSpecialtyAssistant  # noqa: E402
from src.specialty_knowledge import (  # noqa: E402
    lookup_drug,
    lookup_guidelines,
    lookup_labs,
    population_reminders,
    screen_allergy,
    screen_interactions,
)

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **overrides):
    defaults = dict(
        llm_provider="fake",
        embedding_provider="fake",
        vector_store_dir=str(tmp_path / "vs"),
        collection_name="test_specialty",
        session_store_dir=str(tmp_path / "sessions"),
        feedback_store_path=str(tmp_path / "feedback.jsonl"),
        data_dir=str(ROOT / "data" / "raw"),
        chunk_size=200,
        chunk_overlap=40,
        precise_top_k=2,
        comprehensive_top_k=3,
        use_rerank=False,
        enable_safety_guard=True,
        specialty_enabled=True,
        specialty_knowledge_strict=True,
        specialty_top_k=3,
        specialty_append_disclaimer=True,
        specialty_warn_diagnosis_assertion=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def system(tmp_path):
    s = MedRAGSystem(make_settings(tmp_path))
    s.ingest_documents([])  # 不构建索引，验证无索引时仍能降级运行
    return s


def assert_common(result, op):
    """所有专项结果都必须具备的字段与安全护栏。"""
    assert result["feature"] == op
    assert result["answer"]
    assert result["disclaimer"]
    assert "【重要声明】" in result["disclaimer"]
    assert isinstance(result["citations"], list)
    assert "citations_markdown" in result
    assert "safety_warning" in result
    assert "diagnosis_warning" in result


# ---------- 结构化知识层 ----------

def test_structured_knowledge_require_source():
    # strict 模式下所有返回的条目都必须带 source
    for item in lookup_labs(["白细胞", "CRP"], strict=True):
        assert item.get("source")
    for item in lookup_guidelines("高血压", strict=True):
        assert item.get("source")
    assert lookup_drug("二甲双胍", strict=True)["source"]
    for item in screen_interactions(["依那普利", "氢氯噻嗪"], strict=True):
        assert item.get("source")
    for item in screen_allergy(["哮喘"], ["美托洛尔"], strict=True):
        assert item.get("source")
    for item in population_reminders(["老年", "肾功能"], strict=True):
        assert item.get("source")


def test_guideline_version_distinction():
    latest = lookup_guidelines("高血压", strict=True, only_latest=True)
    assert latest and all(g["is_latest"] for g in latest)
    all_versions = lookup_guidelines("高血压", strict=True, only_latest=None)
    assert any(not g["is_latest"] for g in all_versions)
    old = [g for g in all_versions if not g["is_latest"]][0]
    assert "旧版" in old["status"] or "替代" in old["status"]
    assert all("recommendation_class" in g and "evidence_level" in g for g in all_versions)


def test_interaction_screening_detects_pair():
    risks = screen_interactions(["依那普利", "氢氯噻嗪"], strict=True)
    assert len(risks) == 1
    assert risks[0]["drug_a"] != risks[0]["drug_b"]
    assert risks[0]["level"] and risks[0]["advice"]
    assert screen_interactions(["二甲双胍", "氨氯地平"], strict=True) == []


# ---------- 1. 症状鉴别 ----------

def test_differential_has_disclaimer(system):
    r = system.specialty_run(
        "differential",
        presentation="中年女性，发热伴干咳 3 天",
        history="类风湿，长期用激素",
    )
    assert_common(r, "differential")
    assert "仅供鉴别思路，不能作为确诊依据" in r["disclaimer"]
    assert r["structured"]["结构化鉴别线索"]["priority"]
    assert "排除" in r["structured"]["结构化鉴别线索"]["priority"]


def test_differential_rejects_empty(system):
    with pytest.raises(ValueError):
        system.specialty_run("differential", presentation="")


# ---------- 2. 报告解读 ----------

def test_lab_interpret(system):
    r = system.specialty_run("lab_interpret", items=["白细胞计数", "CRP"])
    assert_common(r, "lab_interpret")
    assert "不能直接据此诊断疾病" in r["disclaimer"]
    refs = r["structured"]["本地参考范围条目"]
    assert len(refs) == 2
    for ref in refs:
        assert ref["reference_range"] and ref["source"]
        assert "需结合" in ref["context"]


# ---------- 3. 药学 ----------

def test_drug_info(system):
    r = system.specialty_run("drug_info", drug="二甲双胍")
    assert_common(r, "drug")
    info = r["structured"]["结构化药品信息"]
    for key in ("indication", "usage", "contraindication", "adverse", "interaction",
                "renal", "hepatic", "pregnancy", "source"):
        assert key in info


def test_drug_interactions(system):
    r = system.specialty_run("drug_interactions", drugs=["依那普利", "氢氯噻嗪"])
    assert_common(r, "drug")
    assert len(r["structured"]["检出相互作用/禁忌"]) == 1
    with pytest.raises(ValueError):
        system.specialty_run("drug_interactions", drugs=["二甲双胍"])


# ---------- 4. 指南检索 ----------

def test_guideline_search(system):
    r = system.specialty_run("guideline_search", disease="高血压", version="latest")
    assert_common(r, "guideline")
    assert "避免采用过时方案" in r["disclaimer"]
    assert all(g["is_latest"] for g in r["structured"]["指南目录条目"])


# ---------- 5. 文书 ----------

def test_document_summarize(system):
    r = system.specialty_run(
        "document_summarize", text="患者女，45岁，发热伴干咳3天，既往类风湿，长期用激素。"
    )
    assert_common(r, "document")
    assert "禁止直接复制生成完整病历" in r["disclaimer"]


def test_document_template(system):
    r = system.specialty_run("document_template", disease="高血压")
    assert_common(r, "document")
    assert "禁止直接复制生成完整病历提交" in r["disclaimer"]


# ---------- 6. 路径 ----------

def test_pathway(system):
    r = system.specialty_run("pathway", subject="高血压")
    assert_common(r, "pathway")
    p = r["structured"]["结构化路径信息"]
    for key in ("pathway", "pre_op", "post_op_complications", "nursing", "informed_consent"):
        assert p[key]


# ---------- 7. 过敏/特殊人群 ----------

def test_allergy_risk(system):
    r = system.specialty_run(
        "allergy_risk",
        drugs=["美托洛尔"],
        allergies=["哮喘"],
        population=["老年", "肾功能"],
    )
    assert_common(r, "allergy")
    assert r["structured"]["过敏/交叉过敏风险条目"]
    assert r["structured"]["特殊人群提醒条目"]


# ---------- 通用入口与格式化 ----------

def test_unknown_operation(system):
    with pytest.raises(ValueError, match="未知专项操作"):
        system.specialty_run("not_exist")


def test_format_result(system):
    r = system.specialty_run("differential", presentation="发热伴干咳")
    text = system.format_specialty_result(r)
    assert "【专项辅助" in text
    assert r["disclaimer"] in text


def test_disabled_feature(tmp_path):
    s = MedRAGSystem(make_settings(tmp_path, specialty_enabled=False))
    with pytest.raises(RuntimeError, match="已在配置中禁用"):
        s.specialty_run("differential", presentation="发热")


def test_diagnosis_assertion_warning():
    """模型若输出确诊式表述，必须给出警示（用自定义假模型验证）。"""
    from langchain_core.language_models.chat_models import BaseChatModel

    class AssertiveModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "assertive"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="根据资料，确诊为肺炎。"))])

    settings = make_settings(Path("/tmp"))
    assistant = ClinicalSpecialtyAssistant(settings, AssertiveModel(), __import__(
        "src.vectorstore", fromlist=["VectorStoreManager"]
    ).VectorStoreManager(settings, __import__(
        "src.embeddings", fromlist=["FakeEmbeddings"]
    ).FakeEmbeddings()))
    r = assistant.differential("发热伴干咳")
    assert "确诊式表述" in r["diagnosis_warning"]
