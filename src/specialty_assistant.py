"""面向临床医护的专项辅助服务（功能二）。

实现 7 大专项功能（9 个操作）：
1. 症状鉴别辅助      differential()
2. 检验/检查报告解读  lab_interpret()
3. 药物药学辅助      drug_info() / drug_interactions()
4. 指南共识检索      guideline_search()
5. 病例文书辅助      document_summarize() / document_template()
6. 临床路径/操作规范  pathway()
7. 不良反应/过敏风险  allergy_risk()

所有功能：RAG 检索证据 → Prompt 调用 LLM → 结构化知识补充 → 安全护栏 → 强制免责声明。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from . import specialty_knowledge as sk
from .citation import build_citations, citations_to_context, render_citations_markdown
from .config import Settings
from .prompts import COMPREHENSIVE_EXTRA
from .specialty_prompts import DISCLAIMERS, PROMPTS
from .utils import get_logger
from .vectorstore import VectorStoreManager

logger = get_logger(__name__)

# 确诊式表述检测（用于警示，不阻断输出）
_DIAGNOSIS_ASSERTION = re.compile(
    r"(确诊为|诊断为|就是[某某一].*症|得了[某一].*病|确定是[某一].*病|患有[某一].*病)"
)


class ClinicalSpecialtyAssistant:
    """临床医护专项辅助功能统一服务。"""

    def __init__(self, settings: Settings, llm: BaseChatModel, vsm: VectorStoreManager):
        self.settings = settings
        self.llm = llm
        self.vsm = vsm
        self._chains = {name: prompt | llm | StrOutputParser() for name, prompt in PROMPTS.items()}
        self.strict = settings.specialty_knowledge_strict
        self.top_k = settings.specialty_top_k

    # ---------- 内部通用流程 ----------
    def _retrieve(self, query: str) -> list[Document]:
        """从向量库检索证据；索引不存在时返回空列表（降级为仅结构化知识）。"""
        try:
            store = self.vsm.store
        except Exception as exc:  # noqa: BLE001
            logger.warning("专项功能检索跳过（索引不可用）: %s", exc)
            return []
        try:
            return store.similarity_search(query, k=self.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("专项功能检索失败: %s", exc)
            return []

    def _run(
        self,
        op: str,
        query: str,
        variables: dict[str, Any],
        structured: dict[str, Any] | list[dict[str, Any]] | None = None,
        extra_instruction: str = COMPREHENSIVE_EXTRA,
    ) -> dict[str, Any]:
        docs = self._retrieve(query)
        citations = build_citations(docs)
        context = citations_to_context(citations) or "（参考资料库暂无相关证据，以下主要依据本地结构化知识，须谨慎核对。）"
        chain = self._chains[op]
        answer = chain.invoke(
            {
                "context": context,
                "extra_instruction": extra_instruction,
                **variables,
            }
        )
        answer = str(answer)

        disclaimer = DISCLAIMERS.get(op, DISCLAIMERS["differential"])
        safety_warning = ""
        diagnosis_warning = ""
        if self.settings.enable_safety_guard:
            # 复用通用高危意图护栏
            from .utils import safety_check

            passed, warning = safety_check(query, answer)
            safety_warning = warning
        if self.settings.specialty_warn_diagnosis_assertion and _DIAGNOSIS_ASSERTION.search(answer):
            diagnosis_warning = (
                "⚠️ 检测到疑似确诊式表述：本工具仅提供鉴别/参考思路，"
                "任何诊断结论必须由执业医师作出。"
            )

        result: dict[str, Any] = {
            "feature": op,
            "answer": answer,
            "disclaimer": disclaimer,
            "citations": [c.as_dict() for c in citations],
            "citations_markdown": render_citations_markdown(citations),
            "structured": structured or {},
            "safety_warning": safety_warning,
            "diagnosis_warning": diagnosis_warning,
        }
        return result

    @staticmethod
    def _fmt(value: Any) -> str:
        """把结构化知识渲染成 Prompt 可读文本。"""
        if not value:
            return "（本地知识库无匹配条目）"
        return json.dumps(value, ensure_ascii=False, indent=2)

    # ---------- 1. 症状鉴别辅助 ----------
    def differential(self, presentation: str, history: str = "") -> dict[str, Any]:
        """输入主诉+体征+基础病史，输出鉴别思路（禁止确诊式结论）。"""
        if not presentation or not presentation.strip():
            raise ValueError("主诉/体征不能为空。")
        query = f"鉴别诊断 {presentation} {history}"
        hint = sk.lookup_differential_hints(presentation + " " + history, self.strict)
        structured = {"结构化鉴别线索": hint} if hint else {}
        return self._run(
            "differential",
            query,
            {"presentation": presentation, "history": history or "未提供"},
            structured,
        )

    # ---------- 2. 检验/检查报告解读 ----------
    def lab_interpret(self, items: list[str] | str) -> dict[str, Any]:
        """输入指标/术语列表，逐项解读意义、参考范围、异常提示与需结合的临床信息。"""
        if isinstance(items, str):
            items = [line.strip() for line in items.replace("，", "\n").splitlines() if line.strip()]
        items = [str(i).strip() for i in items if str(i).strip()]
        if not items:
            raise ValueError("待解读指标不能为空。")
        query = "检验检查报告术语解读 " + " ".join(items)
        refs = sk.lookup_labs(items, self.strict)
        structured = {"本地参考范围条目": refs} if refs else {}
        return self._run(
            "lab_interpret",
            query,
            {"items": "\n".join(f"- {i}" for i in items)},
            structured,
        )

    # ---------- 3. 药物药学辅助 ----------
    def drug_info(self, drug: str, question: str = "") -> dict[str, Any]:
        """单药信息：适应症/用法用量/禁忌/不良反应/相互作用/肝肾调整/妊娠分级。"""
        if not drug or not drug.strip():
            raise ValueError("药品名称不能为空。")
        info = sk.lookup_drug(drug, self.strict)
        q = f"{drug} {question} 适应症 用法用量 禁忌 不良反应 相互作用 肝肾功能调整 妊娠分级"
        structured = {"结构化药品信息": info} if info else {}
        return self._run(
            "drug",
            q,
            {"question": question or f"请说明 {drug} 的药学信息（适应症、用法用量、禁忌症、不良反应、相互作用、肝肾功能调整、妊娠分级）。",
             "structured": self._fmt(structured)},
            structured,
        )

    def drug_interactions(self, drugs: list[str]) -> dict[str, Any]:
        """联用风险筛查：逐对分析配伍禁忌/相互作用。"""
        drugs = [str(d).strip() for d in drugs if str(d).strip()]
        if len(drugs) < 2:
            raise ValueError("联用风险筛查至少需要 2 种药品。")
        risks = sk.screen_interactions(drugs, self.strict)
        missing = [d for d in drugs if sk.lookup_drug(d, self.strict) is None]
        structured = {
            "输入药品": drugs,
            "检出相互作用/禁忌": risks,
            "本地知识库未收录的药品": missing,
        }
        return self._run(
            "drug",
            "药物相互作用 配伍禁忌 联用风险 " + " ".join(drugs),
            {"question": f"请对以下药品做联用风险筛查：{'、'.join(drugs)}。逐对分析配伍禁忌与相互作用，并给出处理建议。",
             "structured": self._fmt(structured)},
            structured,
        )

    # ---------- 4. 指南、共识快速检索 ----------
    def guideline_search(self, disease: str, version: str = "latest") -> dict[str, Any]:
        """按疾病检索指南/共识，含推荐等级、证据级别与版本区分。"""
        if not disease or not disease.strip():
            raise ValueError("疾病/主题不能为空。")
        v = version.lower()
        only_latest: bool | None = None
        if v in {"latest", "new", "最新", "current"}:
            only_latest = True
        elif v in {"old", "旧版", "历史", "all", "全部"}:
            only_latest = None
        guidelines = sk.lookup_guidelines(disease, self.strict, only_latest=only_latest)
        structured = {"指南目录条目": guidelines}
        return self._run(
            "guideline",
            f"{disease} 指南 共识 推荐等级 证据级别 版本",
            {"disease": disease, "version": version},
            structured,
        )

    # ---------- 5. 病例文书辅助 ----------
    def document_summarize(self, text: str) -> dict[str, Any]:
        """从病程记录提炼要点：主诉/现病史/既往史/用药/阳性体征。"""
        if not text or not text.strip():
            raise ValueError("病程记录不能为空。")
        return self._run(
            "document",
            "病例要点提炼 病程记录摘要 主诉 现病史 既往史 用药 阳性体征",
            {"task": "要点提炼（主诉/现病史/既往史/用药/阳性体征/关键检查，原文没有填“未提及”）", "text": text},
            {},
        )

    def document_template(self, disease: str) -> dict[str, Any]:
        """给出某疾病现病史/首次病程记录的书写要点模板（禁止生成完整病历）。"""
        if not disease or not disease.strip():
            raise ValueError("疾病名称不能为空。")
        return self._run(
            "document",
            f"{disease} 现病史 首次病程记录 书写要点 模板",
            {"task": f"仅给出「{disease}」现病史与首次病程记录的书写要点模板（框架提示，禁止生成完整病历）",
             "text": f"疾病：{disease}（无病程原文，仅输出书写要点模板）"},
            {},
        )

    # ---------- 6. 临床路径 / 操作规范 ----------
    def pathway(self, subject: str, aspect: str = "全部") -> dict[str, Any]:
        """查询疾病标准临床路径、术前评估、术后并发症、护理、知情告知要点。"""
        if not subject or not subject.strip():
            raise ValueError("疾病/操作名称不能为空。")
        path = sk.lookup_pathway(subject, self.strict)
        structured = {"结构化路径信息": path} if path else {}
        return self._run(
            "pathway",
            f"{subject} 临床路径 术前评估 术后并发症 护理 知情告知 {aspect}",
            {"subject": subject, "aspect": aspect},
            structured,
        )

    # ---------- 7. 不良反应 / 过敏风险 ----------
    def allergy_risk(
        self,
        drugs: list[str] | str,
        allergies: list[str] | str = "",
        population: list[str] | str = "",
    ) -> dict[str, Any]:
        """输入用药+过敏史+特殊人群，检索过敏风险与用药提醒。"""
        def to_list(x: list[str] | str) -> list[str]:
            if isinstance(x, str):
                return [s.strip() for s in re.split(r"[,，、\n]", x) if s.strip()]
            return [str(i).strip() for i in x if str(i).strip()]

        drugs_l = to_list(drugs)
        allergies_l = to_list(allergies)
        population_l = to_list(population)
        if not drugs_l and not allergies_l:
            raise ValueError("用药或过敏史至少提供一项。")
        allergy_hits = sk.screen_allergy(allergies_l, drugs_l, self.strict)
        pop_hits = sk.population_reminders(population_l, self.strict)
        structured = {
            "用药": drugs_l,
            "过敏史": allergies_l,
            "过敏/交叉过敏风险条目": allergy_hits,
            "特殊人群提醒条目": pop_hits,
        }
        return self._run(
            "allergy",
            "用药 过敏史 不良反应 特殊人群 孕哺 儿童 老年 肝肾功能 风险提示 " + " ".join(drugs_l + allergies_l + population_l),
            {
                "drugs": "、".join(drugs_l) or "未提供",
                "allergies": "、".join(allergies_l) or "未提供",
                "population": "、".join(population_l) or "未提供",
            },
            structured,
        )

    # ---------- 统一入口与格式化 ----------
    def run(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """按操作名统一调用（供 API/CLI 使用）。"""
        dispatch: dict[str, Callable[..., dict[str, Any]]] = {
            "differential": self.differential,
            "lab_interpret": self.lab_interpret,
            "drug_info": self.drug_info,
            "drug_interactions": self.drug_interactions,
            "guideline_search": self.guideline_search,
            "document_summarize": self.document_summarize,
            "document_template": self.document_template,
            "pathway": self.pathway,
            "allergy_risk": self.allergy_risk,
        }
        fn = dispatch.get(operation)
        if not fn:
            raise ValueError(
                f"未知专项操作: {operation}；可选: {', '.join(dispatch)}"
            )
        return fn(**kwargs)

    @staticmethod
    def format_result(result: dict[str, Any]) -> str:
        """把专项结果格式化为可读文本。"""
        lines = [
            "=" * 64,
            f"【专项辅助 · {result['feature']}】",
            result["answer"],
            "",
        ]
        if result.get("diagnosis_warning"):
            lines += [result["diagnosis_warning"], ""]
        if result.get("safety_warning"):
            lines += [result["safety_warning"], ""]
        if result.get("structured"):
            lines += ["【结构化知识（本地知识库，须核对原文）】",
                      json.dumps(result["structured"], ensure_ascii=False, indent=2), ""]
        if result.get("citations"):
            lines.append("【引用来源（文档 / 章节 / 页码）】")
            for c in result["citations"]:
                loc = []
                if c.get("section"):
                    loc.append(f"章节: {c['section']}")
                if c.get("page", -1) > 0:
                    loc.append(f"第{c['page']}页")
                loc_str = "，".join(loc) if loc else c.get("source", "")
                lines.append(f"  [{c['index']}] {c['doc_title']}（{loc_str}） chunk={c['chunk_id']}")
        lines += ["", result["disclaimer"], "=" * 64]
        return "\n".join(lines)
