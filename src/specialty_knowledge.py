"""专项辅助功能的结构化本地参考知识（教学演示版）。

设计原则：
- 每条记录都带 source（来源）与 version（版本/年份），strict 模式下无来源条目不返回；
- 仅作为"线索/提醒"，最终以药品说明书、药典、最新版指南为准；
- 数据为教学用简化示例，生产必须替换为审核过的权威知识库。
"""
from __future__ import annotations

from typing import Any, Iterable


# ---------- 通用过滤：strict 模式只保留带来源的条目 ----------
def _usable(item: dict[str, Any], strict: bool) -> bool:
    return (not strict) or bool(str(item.get("source")))


# ---------- 2. 检验/检查参考范围 ----------
LAB_REFERENCES: list[dict[str, Any]] = [
    {
        "name": "白细胞计数 WBC",
        "reference_range": "成人 (3.5–9.5)×10⁹/L（不同实验室略有差异）",
        "meaning": "反映机体免疫/炎症状态，升高常见于感染、应激、血液系统疾病。",
        "high_hint": "可能提示细菌感染、炎症、应激或血液系统疾病（需结合中性粒细胞分类）。",
        "low_hint": "可能提示病毒感染、骨髓抑制、自身免疫等。",
        "context": "需结合症状（发热/咳嗽）、中性粒细胞比例、CRP、用药史（激素/免疫抑制剂）。",
        "source": "临床检验基础（教学示例）",
        "version": "2024 教学版",
    },
    {
        "name": "C 反应蛋白 CRP",
        "reference_range": "<10 mg/L（不同检测方法有差异）",
        "meaning": "急性时相反应蛋白，反映炎症/组织损伤程度，特异性不高。",
        "high_hint": "提示存在炎症或感染，动态变化比单次值更有意义。",
        "low_hint": "一般无临床意义。",
        "context": "需结合白细胞、降钙素原、症状与影像学，不能单凭 CRP 判断感染部位。",
        "source": "炎症指标临床应用（教学示例）",
        "version": "2024 教学版",
    },
    {
        "name": "血红蛋白 Hb",
        "reference_range": "成年男性 130–175 g/L；成年女性 115–150 g/L",
        "meaning": "反映贫血或红细胞增多状态。",
        "high_hint": "可见于脱水、慢性缺氧、红细胞增多症等。",
        "low_hint": "提示贫血，需进一步查平均红细胞体积、铁代谢等明确类型。",
        "context": "需结合性别、月经/出血史、营养状况、慢性病。",
        "source": "临床检验基础（教学示例）",
        "version": "2024 教学版",
    },
    {
        "name": "空腹血糖 FPG",
        "reference_range": "3.9–6.1 mmol/L（糖尿病诊断需另行标准）",
        "meaning": "反映基础血糖水平，用于糖代谢异常筛查。",
        "high_hint": "提示糖耐量异常或糖尿病可能，需复查或做 OGTT/HbA1c。",
        "low_hint": "可见于饥饿、胰岛素过量、某些内分泌疾病。",
        "context": "需结合餐后血糖、HbA1c、症状及用药（激素可升高血糖）。",
        "source": "糖尿病基础管理（示例教学资料）",
        "version": "2024 教学版",
    },
    {
        "name": "血清肌酐 Cr / eGFR",
        "reference_range": "肌酐成年男性 57–97 μmol/L，女性 41–73 μmol/L（因实验室/肌群而异）",
        "meaning": "反映肾小球滤过功能，eGFR 用于慢性肾病分期与用药剂量调整。",
        "high_hint": "提示肾功能下降，需结合尿素氮、尿常规、年龄与肌群。",
        "low_hint": "多无临床意义，可见于肌肉量低。",
        "context": "是多种药物（如二甲双胍、ACEI/ARB）剂量调整与禁忌判断的关键依据。",
        "source": "慢性肾病用药（教学示例）",
        "version": "2024 教学版",
    },
    {
        "name": "胸部 X 线报告术语：磨玻璃影",
        "reference_range": "影像学描述术语，非数值指标",
        "meaning": "CT 上密度轻度增高、仍能透过看到血管纹理的模糊影。",
        "high_hint": "可见于炎症、水肿、出血、间质性病变等，需结合临床与随访。",
        "low_hint": "不适用。",
        "context": "需结合症状、病程、炎症指标、既往影像对比；不能直接等同于某种疾病。",
        "source": "影像报告术语解读（教学示例）",
        "version": "2024 教学版",
    },
]


def lookup_labs(names: Iterable[str], strict: bool = True) -> list[dict[str, Any]]:
    """按名称/别名匹配检验参考条目。"""
    keys = ["name", "reference_range", "meaning", "high_hint", "low_hint", "context", "source", "version"]
    result: list[dict[str, Any]] = []
    for wanted in names:
        w = str(wanted).strip().lower()
        if not w:
            continue
        for item in LAB_REFERENCES:
            if w in item["name"].lower() or item["name"].lower() in w or w in item.get("aliases", ""):
                picked = {k: item[k] for k in keys if k in item}
                if _usable(item, strict) and picked not in result:
                    result.append(picked)
    return result


# ---------- 3. 药物信息 ----------
DRUGS: dict[str, dict[str, Any]] = {
    "二甲双胍": {
        "name": "二甲双胍 (Metformin)",
        "indication": "2 型糖尿病一线用药，尤其超重/肥胖患者。",
        "usage": "通常随餐服用以减少胃肠反应，剂量由医师根据血糖与肾功能调整。",
        "contraindication": "eGFR<30 禁用；严重感染、缺氧、脱水、造影剂使用时需评估暂停。",
        "adverse": "胃肠道不适、腹泻；罕见乳酸酸中毒（高危人群需警惕）。",
        "interaction": "与造影剂、酒精、利尿剂、糖皮质激素等联用需评估。",
        "renal": "eGFR<45 需减量/评估，<30 禁用（以最新版说明书为准）。",
        "hepatic": "严重肝功能不全慎用/禁用，以说明书为准。",
        "pregnancy": "妊娠期使用须由医师权衡，哺乳期权衡利弊（以最新分级为准）。",
        "source": "糖尿病基础管理（示例教学资料）",
        "version": "2024 教学版",
    },
    "美托洛尔": {
        "name": "美托洛尔 (Metoprolol)",
        "indication": "高血压、心绞痛、部分心律失常、慢性心衰（稳定期）。",
        "usage": "口服剂量个体化，须逐步滴定，不可突然停药。",
        "contraindication": "支气管哮喘、严重心动过缓、未控制的心衰、严重低血压禁用/慎用。",
        "adverse": "心动过缓、乏力、支气管痉挛、低血糖症状掩盖。",
        "interaction": "与维拉帕米、地高辛、胰岛素/降糖药联用需谨慎。",
        "renal": "肾功能不全一般无需调整，严重者监测。",
        "hepatic": "肝功能不全需减量（以说明书为准）。",
        "pregnancy": "妊娠/哺乳期使用须医师评估（以最新分级为准）。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    "依那普利": {
        "name": "依那普利 (Enalapril, ACEI)",
        "indication": "高血压、慢性肾病、心衰；合并糖尿病者常优先考虑。",
        "usage": "从小剂量开始，监测血压与血钾、肌酐。",
        "contraindication": "双侧肾动脉狭窄、妊娠、既往血管性水肿禁用。",
        "adverse": "干咳、高钾血症、血管性水肿、血肌酐升高。",
        "interaction": "与保钾利尿剂、NSAIDs、ARB 联用增加高钾/肾损伤风险。",
        "renal": "肾功能不全需减量并监测肌酐与血钾。",
        "hepatic": "严重肝功能不全慎用。",
        "pregnancy": "妊娠禁用（以最新分级为准）。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    "氨氯地平": {
        "name": "氨氯地平 (Amlodipine, CCB)",
        "indication": "高血压、慢性稳定型心绞痛，老年患者常用。",
        "usage": "每日一次，剂量个体化。",
        "contraindication": "严重低血压、对二氢吡啶类过敏禁用。",
        "adverse": "脚踝水肿、面部潮红、头痛、心悸。",
        "interaction": "与 CYP3A4 强抑制剂/诱导剂联用需评估（以说明书为准）。",
        "renal": "肾功能不全一般无需调整。",
        "hepatic": "肝功能不全慎用并减量。",
        "pregnancy": "妊娠期使用须医师评估（以最新分级为准）。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    "氢氯噻嗪": {
        "name": "氢氯噻嗪 (Hydrochlorothiazide, 利尿剂)",
        "indication": "高血压、老年单纯收缩期高血压，常与 ACEI/ARB 联用。",
        "usage": "剂量个体化，注意监测电解质。",
        "contraindication": "严重肾功能不全、低钾血症、痛风慎用/禁用。",
        "adverse": "低钾血症、低钠血症、尿酸升高、光敏。",
        "interaction": "与洋地黄、NSAIDs、糖皮质激素联用需监测电解质。",
        "renal": "严重肾功能不全疗效下降，需评估。",
        "hepatic": "严重肝病慎用（诱发肝性脑病）。",
        "pregnancy": "妊娠期使用须医师评估（以最新分级为准）。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
}


def lookup_drug(name: str, strict: bool = True) -> dict[str, Any] | None:
    """按药品名（支持别名/部分匹配）查结构化信息。"""
    n = str(name).strip().lower()
    for key, info in DRUGS.items():
        if n == key or n in info["name"].lower() or key in n:
            if _usable(info, strict):
                return dict(info)
    return None


# ---------- 3b. 联用风险筛查规则 ----------
INTERACTION_RULES: list[dict[str, Any]] = [
    {
        "pair": ("依那普利", "氢氯噻嗪"),
        "level": "联用需监测（机制互补，但需警惕电解质/血压）",
        "risk": "ACEI + 利尿剂为常见合理组合，但可能引起低血压、电解质紊乱；依那普利可部分抵消噻嗪类低钾。",
        "advice": "监测血压、血钾、肌酐；由医师确定剂量。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    {
        "pair": ("美托洛尔", "氨氯地平"),
        "level": "通常可联用，注意心率/血压监测",
        "risk": "β 阻滞剂 + CCB 联合降压通常安全，但均应从小剂量起始并监测心率与血压。",
        "advice": "监测心率、血压与水肿；避免突然停用 β 阻滞剂。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    {
        "pair": ("依那普利", "美托洛尔"),
        "level": "通常可联用，注意低血压/心动过缓",
        "risk": "两药均有降压作用，联用需警惕低血压；资料未提示显著药代动力学禁忌。",
        "advice": "从小剂量开始，监测血压与心率。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    {
        "pair": ("二甲双胍", "依那普利"),
        "level": "通常可联用，关注肾功能",
        "risk": "两者分别用于糖尿病与高血压/肾病，联用本身无明确禁忌；但肾功能变化时两药均需重新评估。",
        "advice": "定期监测 eGFR 与血糖；造影/脱水等应激状态下评估暂停。",
        "source": "糖尿病/高血压诊疗（示例教学资料）",
        "version": "2024 教学版",
    },
    {
        "pair": ("美托洛尔", "哮喘"),
        "level": "慎用/禁忌提示",
        "risk": "β 受体阻滞剂可诱发支气管痉挛，哮喘患者慎用（非两药配伍，属疾病-药物禁忌）。",
        "advice": "哮喘患者应避免或仅在专科医师严密评估下使用。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
]


def screen_interactions(drugs: list[str], strict: bool = True) -> list[dict[str, Any]]:
    """两两筛查配伍禁忌/相互作用。"""
    found: list[dict[str, Any]] = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = str(drugs[i]).strip(), str(drugs[j]).strip()
            for rule in INTERACTION_RULES:
                pa, pb = rule["pair"]
                if {a, b} == {pa, pb} and _usable(rule, strict):
                    found.append(
                        {
                            "drug_a": a,
                            "drug_b": b,
                            "level": rule["level"],
                            "risk": rule["risk"],
                            "advice": rule["advice"],
                            "source": rule["source"],
                            "version": rule["version"],
                        }
                    )
    return found


# ---------- 4. 指南/共识版本目录 ----------
GUIDELINES: list[dict[str, Any]] = [
    {
        "disease": "高血压",
        "title": "中国高血压防治指南（示例条目）",
        "organization": "示例高血压防治委员会（教学用）",
        "version": "2024 版（最新）",
        "year": 2024,
        "is_latest": True,
        "recommendation_class": "Ⅰ 类（示例：诊室血压测量诊断）",
        "evidence_level": "A（示例）",
        "status": "最新版，可用于当前诊疗参考（须核对原文）",
        "source": "data/raw/高血压诊疗指南.md",
    },
    {
        "disease": "高血压",
        "title": "高血压防治指南旧版（示例条目）",
        "organization": "示例高血压防治委员会（教学用）",
        "version": "2010 版（旧版）",
        "year": 2010,
        "is_latest": False,
        "recommendation_class": "资料未注明",
        "evidence_level": "资料未注明",
        "status": "旧版，已被新版替代，避免用于当前诊疗方案",
        "source": "data/raw/高血压诊疗指南.md",
    },
    {
        "disease": "2 型糖尿病",
        "title": "2 型糖尿病基础管理指南（示例条目）",
        "organization": "示例糖尿病学会（教学用）",
        "version": "2024 版（最新）",
        "year": 2024,
        "is_latest": True,
        "recommendation_class": "Ⅰ 类（示例：二甲双胍为一线）",
        "evidence_level": "A（示例）",
        "status": "最新版，可用于当前诊疗参考（须核对原文）",
        "source": "data/raw/糖尿病基础管理.md",
    },
    {
        "disease": "脑卒中",
        "title": "急性缺血性卒中诊治指南（示例条目）",
        "organization": "示例卒中学会（教学用）",
        "version": "2023 版（最新）",
        "year": 2023,
        "is_latest": True,
        "recommendation_class": "Ⅰ 类（示例：时间窗内评估溶栓）",
        "evidence_level": "A（示例）",
        "status": "最新版；溶栓时间窗等须按原文执行",
        "source": "data/raw/常见急症识别.md",
    },
]


def lookup_guidelines(disease: str, strict: bool = True, only_latest: bool | None = None) -> list[dict[str, Any]]:
    keys = ["disease", "title", "organization", "version", "year", "is_latest",
            "recommendation_class", "evidence_level", "status", "source"]
    d = str(disease).strip()
    result: list[dict[str, Any]] = []
    for g in GUIDELINES:
        if d and d not in g["disease"]:
            continue
        if only_latest is True and not g["is_latest"]:
            continue
        if only_latest is False and g["is_latest"]:
            continue
        picked = {k: g[k] for k in keys if k in g}
        if _usable(g, strict) and picked not in result:
            result.append(picked)
    return result


# ---------- 6. 临床路径 / 操作规范 ----------
PATHWAYS: dict[str, dict[str, Any]] = {
    "高血压": {
        "subject": "高血压",
        "pathway": "门诊评估 → 危险分层 → 生活方式干预 → 药物起始/联合 → 随访与达标评估。",
        "pre_op": "择期手术前评估血压水平、靶器官损害与用药（β 阻滞剂/ACEI/ARB 是否停用须医师决定）。",
        "post_op_complications": "术后监测血压波动、心脑血管事件、电解质紊乱（尤其利尿剂/ACEI/ARB）。",
        "nursing": "定时监测血压、记录出入量、低盐饮食指导、用药依从性教育。",
        "informed_consent": "说明治疗目标、药物可能不良反应、需长期随访及生活方式干预的重要性。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
    "2 型糖尿病": {
        "subject": "2 型糖尿病",
        "pathway": "初诊评估（血糖/HbA1c/并发症筛查）→ 生活方式 + 一线药物 → 随访调整 → 并发症监测。",
        "pre_op": "术前评估血糖控制、肾功能、心血管风险；评估二甲双胍等药物围术期管理。",
        "post_op_complications": "监测血糖波动、感染、伤口愈合、低血糖与酮症。",
        "nursing": "血糖监测、饮食指导、足部护理、低血糖识别与处理教育。",
        "informed_consent": "说明血糖目标、用药方案、低血糖风险与慢性并发症筛查计划。",
        "source": "糖尿病基础管理（示例教学资料）",
        "version": "2024 教学版",
    },
}


def lookup_pathway(subject: str, strict: bool = True) -> dict[str, Any] | None:
    s = str(subject).strip()
    for key, info in PATHWAYS.items():
        if key in s or s in key:
            if _usable(info, strict):
                return dict(info)
    return None


# ---------- 7. 过敏 / 特殊人群规则 ----------
ALLERGY_RULES: list[dict[str, Any]] = [
    {
        "trigger": "青霉素",
        "risk": "青霉素类存在交叉过敏风险，使用头孢类等 β-内酰胺药物前须详细询问过敏史并评估。",
        "action": "明确过敏者避免使用相关药物，必要时做皮试/脱敏由专科决定。",
        "source": "药物临床应用指南（教学示例）",
        "version": "2024 教学版",
    },
    {
        "trigger": "磺胺",
        "risk": "磺胺类过敏者使用含磺胺结构药物（部分利尿剂、磺酰脲类）需警惕交叉过敏。",
        "action": "避免使用同类结构药物，替代方案由医师评估。",
        "source": "药物临床应用指南（教学示例）",
        "version": "2024 教学版",
    },
    {
        "trigger": "哮喘",
        "risk": "哮喘患者使用 β 受体阻滞剂（如美托洛尔）可能诱发支气管痉挛。",
        "action": "慎用/避免，必要时专科严密评估。",
        "source": "高血压诊疗指南（示例教学资料）",
        "version": "2024 教学版",
    },
]

POPULATION_RULES: dict[str, list[dict[str, Any]]] = {
    "妊娠": [
        {"note": "ACEI/ARB（如依那普利）妊娠禁用；二甲双胍/β 阻滞剂需医师权衡。",
         "source": "药品说明书/指南（教学示例）", "version": "2024 教学版"},
    ],
    "哺乳": [
        {"note": "多数药物可经乳汁分泌，用药须权衡利弊并参考最新版说明书。",
         "source": "药品说明书（教学示例）", "version": "2024 教学版"},
    ],
    "儿童": [
        {"note": "剂量按体重/体表面积计算，肝肾功能发育未成熟，须按儿科说明书调整。",
         "source": "药物临床应用指南（教学示例）", "version": "2024 教学版"},
    ],
    "老年": [
        {"note": "起始剂量宜小、缓慢加量，关注多重用药与肝肾功能减退（starting low, go slow）。",
         "source": "高血压诊疗指南（示例教学资料）", "version": "2024 教学版"},
    ],
    "肾功能": [
        {"note": "eGFR<30 二甲双胍禁用；ACEI/ARB 需减量并监测肌酐/血钾；依据 eGFR 调整剂量。",
         "source": "慢性肾病用药（教学示例）", "version": "2024 教学版"},
    ],
    "肝功能": [
        {"note": "严重肝功能不全者多种药物需减量/禁用（如二甲双胍、美托洛尔、氨氯地平），以说明书为准。",
         "source": "药品说明书（教学示例）", "version": "2024 教学版"},
    ],
}


def screen_allergy(allergies: Iterable[str], drugs: Iterable[str], strict: bool = True) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    allergy_text = " ".join(str(a) for a in allergies)
    drug_text = " ".join(str(d) for d in drugs)
    for rule in ALLERGY_RULES:
        t = rule["trigger"]
        if (t in allergy_text or t in drug_text) and _usable(rule, strict):
            found.append({k: rule[k] for k in ("trigger", "risk", "action", "source", "version")})
    return found


def population_reminders(populations: Iterable[str], strict: bool = True) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    text = " ".join(str(p) for p in populations)
    for key, rules in POPULATION_RULES.items():
        if key in text:
            for r in rules:
                if _usable(r, strict) and r not in found:
                    found.append(dict(r))
    return found


# ---------- 1. 常见鉴别线索（用于症状鉴别的结构化提示） ----------
DIFFERENTIAL_HINTS: list[dict[str, Any]] = [
    {
        "syndrome": "发热伴干咳、长期使用激素、类风湿",
        "consider": ["感染（含机会性感染）", "间质性肺病/药物性肺损伤", "类风湿相关肺受累", "肿瘤"],
        "priority": "长期使用激素者感染风险升高，应先排除感染与机会性感染。",
        "workup": ["胸部 CT", "血常规+CRP/PCT", "痰检/病原学", "必要时支气管镜", "免疫指标复查"],
        "source": "临床鉴别思路（教学示例）",
        "version": "2024 教学版",
    },
    {
        "syndrome": "突发胸痛伴冷汗、放射痛",
        "consider": ["急性冠脉综合征/心肌梗死", "主动脉夹层", "肺栓塞", "气胸"],
        "priority": "按危急重症优先排查，立即心电图与心肌标志物。",
        "workup": ["心电图", "心肌标志物", "D-二聚体/CT 血管成像", "生命体征监测"],
        "source": "常见急症识别（示例教学资料）",
        "version": "2024 教学版",
    },
]


def lookup_differential_hints(presentation: str, strict: bool = True) -> dict[str, Any] | None:
    text = str(presentation)
    best: dict[str, Any] | None = None
    best_score = 0
    for hint in DIFFERENTIAL_HINTS:
        score = sum(1 for term in ["发热", "干咳", "激素", "类风湿", "胸痛", "冷汗"] if term in text)
        if score > best_score and _usable(hint, strict):
            best_score, best = score, dict(hint)
    return best
