# MedRAG — 基于 LangChain 的医疗临床智能诊疗问答系统

一个可落地的 RAG（检索增强生成）问答系统，面向医疗临床教育场景：把医学指南、教材、病历知识等文档构建为向量索引，用户提问时先检索相关证据，再由大模型基于证据生成**可溯源、带免责声明**的回答。

> ⚠️ **重要声明**：本系统仅用于健康教育与技术演示，**不能替代执业医师的诊断、处方或急救处置**。所有回答仅供参考。

---

## 1. 系统架构

```
┌──────────────┐   文档加载     ┌───────────────┐   切分     ┌──────────────┐
│  原始文档     │ ────────────▶ │ DocumentProcessor │ ──────▶ │  Chunks      │
│ PDF/DOCX/TXT │               └───────────────┘           └──────┬───────┘
└──────────────┘                                                   │ Embedding
                                                                   ▼
                                                           ┌──────────────┐
                                                           │  FAISS 向量库 │  持久化
                                                           └──────┬───────┘
                                                                  │ 检索
用户问题 ──▶ Retriever(相似度/MMR/混合) ──▶ 带编号上下文 ──▶ LLM(带 Prompt) ──▶ 答案 + 来源 + 安全护栏
```

**核心模块**（均在 `src/`）：

| 文件 | 职责 |
|------|------|
| `config.py` | YAML 默认 + `.env` 环境变量覆盖的配置管理 |
| `document_processor.py` | 多格式文档加载（PDF/DOCX/TXT/MD/CSV）与语义切分 |
| `embeddings.py` | OpenAI 兼容 / Ollama / Fake（测试）Embedding 工厂 |
| `vectorstore.py` | FAISS 索引构建、保存、加载、增量添加 |
| `retriever.py` | similarity / MMR / BM25 混合检索（RRF 融合） |
| `clinical_retriever.py` | **临床增强检索**：HyDE / MultiQuery / Rerank / 双模式 |
| `citation.py` | **引用溯源**：结构化引用、文档/章节/页码、原文片段跳转 |
| `specialty_prompts.py` | **功能二**：7 大专项 Prompt 模板 + 强制免责声明 |
| `specialty_knowledge.py` | **功能二**：带来源/版本标注的结构化知识库（检验参考、药品、相互作用、指南、路径、过敏规则） |
| `specialty_assistant.py` | **功能二**：专项辅助统一服务 `ClinicalSpecialtyAssistant`（9 个操作） |
| `memory.py` | **多轮会话记忆**：TTL 过期、摘要压缩、敏感会话清理 |
| `feedback.py` | **反馈机制**：正确/幻觉/过时/不完整，检索与提示词调优 |
| `clinical_chain.py` | 临床问答服务：整合上述增强模块 |
| `llm.py` | OpenAI 兼容 / Ollama / Fake LLM 工厂 |
| `prompts.py` | 中文医疗 Prompt 模板、双模式/多轮上下文、上下文格式化 |
| `chain.py` | 基础 LCEL 链：检索 → 上下文 → LLM → 后处理 |
| `rag_system.py` | 顶层系统类 `MedRAGSystem` |
| `api.py` | FastAPI 服务接口（含临床增强 + 安全合规端点） |
| `utils.py` | 日志、文本清洗、安全护栏 |
| `privacy.py` | **功能三**：隐私脱敏预处理链（姓名/手机/身份证/住址） |
| `content_filter.py` | **功能三**：前置内容过滤与意图分类（违规拦截） |
| `role_prompts.py` | **功能三**：医护/患者双角色 Prompt + 幻觉抑制铁律 |
| `audit.py` | **功能三**：全量审计留痕（JSONL，只增不改） |
| `safety_chain.py` | **功能三**：安全问答编排层（过滤→脱敏→检索→生成→审计） |
| `knowledge_base.py` | **功能四**：知识库管理后台（上传/增量入库、分类、版本、权限隔离、检索调试面板） |
| `ops.py` | **功能五**：后台运维统计、模型切换、RAG 开关、参数配置、会话管理 |
| `scripts/ops_admin.py` | **功能五**：后台运维管理 CLI（统计/模型/参数/会话/对比） |

---

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key 与模型：

```env
# 使用 DeepSeek / 通义 / 智谱等 OpenAI 兼容服务示例
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxxx

EMBEDDING_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxxx
```

> 若不填 API Key，可用测试模式（见第 5 节）跑通全流程。

### 2.3 构建索引

```bash
python scripts/ingest.py
# 或指定目录
python scripts/ingest.py --dir ./data/raw
```

### 2.4 提问

```bash
# 单次提问
python scripts/query.py "高血压的诊断标准是什么？"

# 交互模式
python scripts/query.py
```

### 2.5 启动 API 服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

接口：

- `GET  /health` — 健康检查
- `POST /ingest` — `{"directory": "./data/raw", "rebuild": true}`
- `POST /query` — `{"question": "二甲双胍怎么吃？"}`
- `POST /query/simple` — 只返回答案文本

Swagger 文档：<http://127.0.0.1:8000/docs>

---

## 3. 临床增强功能（核心新增）

### 3.1 基础 RAG 问答 + 双模式

支持医生的自然语言临床提问（症状鉴别、用药选择、诊疗流程、指南原文查询、并发症处理、禁忌症、检验指标解读），提供两种模式：

| 模式 | 说明 | LangChain 技术点 |
|------|------|------------------|
| `precise` 精准检索 | 优先引用指南/规范原文，逐字核对，不做跨文档推断 | 指南加权 Rerank、引用溯源 |
| `comprehensive` 综合推理 | 多文档交叉综合分析，给出鉴别诊断与推理过程并标注依据 | **HyDE 改写**、**MultiQuery 多查询生成**、**Rerank 重排** |

```bash
python scripts/clinical.py --mode precise "高血压的诊断标准是什么？"
python scripts/clinical.py --mode comprehensive "老年糖尿病合并肌酐升高如何选药？"
```

 HyDE / MultiQuery / Rerank 可在 `.env` 中开关：

```env
USE_HYDE=true
USE_MULTI_QUERY=true
MULTI_QUERY_COUNT=3
USE_RERANK=true
RERANK_MODEL=BAAI/bge-reranker-base   # CrossEncoder；需 pip install sentence-transformers
```

> Rerank 支持三级降级：CrossEncoder → LLM 打分 → 关键词/指南启发式，无额外依赖也能工作。

### 3.2 引用溯源与原文跳转

**每条回答都附带引用来源、文档名称、章节 / 页码、chunk_id**，医生可直接核对文献、防止幻觉：

```
[1] 高血压诊疗指南（章节: 一、定义与诊断标准，第 12 页） chunk=chunk-000030
```

- 结果中 `citations` 字段包含结构化引用，`citations_markdown` 是带 `<details>` 折叠块的 Markdown，前端可直接渲染“点击查看原文片段”。
- API: `POST /clinical/query` 返回引用；`GET /clinical/session/{id}/original/{chunk_id}` 按 chunk_id 获取**完整原文片段**。
- Python: `system.get_original(result, chunk_id)`。

### 3.3 多轮对话上下文记忆

保留本次病例的对话背景（例如先说“老年、糖尿病、肌酐升高”，后续提问自动带上）。实现参照 `ConversationBufferMemory` / `ConversationSummaryMemory`：

- 超过 `MEMORY_MAX_TURNS` 轮自动摘要压缩，保留关键临床背景
- `MEMORY_TTL_SECONDS` 会话过期自动清理
- `MEMORY_AUTO_PURGE_SENSITIVE=true` 时，含身份证/手机号等敏感信息的会话**不落盘并在清理时直接丢弃**
- 会话可持久化到 `SESSION_STORE_DIR`，重启后恢复

```python
sid = system.new_session()
system.clinical_query("患者老年、糖尿病、肌酐升高", session_id=sid)
system.clinical_query("那如何选药？", session_id=sid)   # 自动带上前面的病例背景
```

### 3.4 问答反馈机制

医生可对回答标记 `correct` / `hallucination`（幻觉）/ `outdated`（过时）/ `incomplete`（不完整）：

```python
system.submit_feedback(question, answer, "hallucination", result=result, note="剂量未注明出处")
```

反馈以 JSONL 持久化，并用于**检索调优**（按 chunk 负面反馈率调节权重）与**提示词优化**（幻觉率高时强调严格引用、过时反馈多时提示核对最新指南）：

```bash
# CLI 交互模式输入：feedback hallucination
# API：
POST /clinical/feedback   {"question": "...", "answer": "...", "label": "hallucination", "chunk_ids": [...]}
GET  /clinical/feedback/stats
```

### 3.5 临床增强 API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/clinical/session` | 新建会话，返回 session_id |
| `POST` | `/clinical/query` | 临床问答（双模式 + 记忆 + 引用 + Rerank） |
| `POST` | `/clinical/feedback` | 提交医生反馈 |
| `GET` | `/clinical/feedback/stats` | 反馈统计 |
| `GET` | `/clinical/session/{id}/original/{chunk_id}` | 查看完整原文片段 |

### 3.6 Python 用法

```python
from src.rag_system import MedRAGSystem

system = MedRAGSystem()
system.ingest("./data/raw")

sid = system.new_session()
result = system.clinical_query("老年糖尿病患者如何选降压药？", mode="comprehensive", session_id=sid)
print(system.clinical.format_result(result))
print(result["citations"])                # 结构化引用（文档/章节/页码）
frag = system.get_original(result, result["citations"][0]["chunk_id"])  # 原文跳转
system.submit_feedback("...", "...", "correct", result=result)
print(system.feedback_stats())
```

---

## 4. 面向临床医护的专项辅助功能（功能二 · 重点）

在基础 RAG 问答之上，提供 **7 大专项功能 / 9 个操作**，全部遵循「检索证据 → LLM → 结构化知识补充 → 安全护栏 → 强制免责声明」链路。所有结论**仅供鉴别/参考思路，不能作为确诊依据**，且结构化条目均带 `source`/`version` 标注、严格模式过滤无来源条目。

| # | 功能 | 操作名 | 说明 |
|---|------|--------|------|
| 1 | 症状鉴别辅助 | `differential` | 主诉+体征+基础病史 → 鉴别诊断列表 / 需排查疾病 / 检查项目 / 鉴别要点；强制标注“仅供鉴别思路，不能作为确诊依据” |
| 2 | 检验/检查报告解读 | `lab_interpret` | 血常规、生化、炎症指标、影像术语 → 意义 / 参考范围 / 异常提示 / 需结合的临床信息；**不下疾病结论** |
| 3 | 药物药学辅助 | `drug_info` / `drug_interactions` | 适应症、用法用量、禁忌、不良反应、相互作用、肝肾功能调整、妊娠分级；多药联用配伍禁忌/相互作用筛查。来源：药典/说明书/临床应用指南 |
| 4 | 指南共识快速检索 | `guideline_search` | 按疾病检索指南，含 Ⅰ/Ⅱ 类推荐与 A/B 证据级别，区分最新版/旧版（避免过时方案） |
| 5 | 病例文书辅助 | `document_summarize` / `document_template` | LangChain 摘要提取主诉/现病史/既往史/用药/阳性体征；给出书写要点模板。**禁止生成可直接复制提交的完整病历**，法律责任由医师承担 |
| 6 | 临床路径/操作规范 | `pathway` | 标准临床路径、术前评估、术后并发症观察、护理要点、知情告知要点 |
| 7 | 不良反应/过敏风险 | `allergy_risk` | 用药+过敏史 → 过敏/交叉过敏风险；孕哺、儿童、老年、肝肾功能损伤等特殊人群提醒 |

> 注意：项目内置的结构化知识与示例文档为**教学示例数据**，生产必须替换为审核过的权威知识库（药典、说明书、指南原文）。

### 4.1 CLI 用法

```bash
# 1. 症状鉴别（含强制免责声明）
python scripts/specialty.py differential --presentation "中年女性，发热伴干咳3天" --history "类风湿，长期用激素"

# 2. 报告解读
python scripts/specialty.py lab --items "白细胞计数" "CRP"

# 3. 单药信息 / 联用筛查
python scripts/specialty.py drug --drug 二甲双胍
python scripts/specialty.py interactions --drugs 依那普利 氢氯噻嗪

# 4. 指南检索（latest 最新 / all 全部版本）
python scripts/specialty.py guidelines --disease 高血压

# 5. 文书要点提炼 / 书写模板
python scripts/specialty.py summarize --text "患者女，45岁，发热伴干咳3天，既往类风湿…"
python scripts/specialty.py template --disease 高血压

# 6. 临床路径
python scripts/specialty.py pathway --subject 高血压

# 7. 过敏与特殊人群
python scripts/specialty.py allergy --drugs 美托洛尔 --allergies 哮喘 --population 老年 肾功能

# 无参数进入交互菜单
python scripts/specialty.py
```

### 4.2 API 端点

| 方法 | 端点 | 请求体示例 |
|------|------|-----------|
| `POST` | `/specialty/differential` | `{"presentation": "发热伴干咳3天", "history": "长期激素"}` |
| `POST` | `/specialty/lab-interpret` | `{"items": ["白细胞计数", "CRP"]}` |
| `POST` | `/specialty/drug-info` | `{"drug": "二甲双胍"}` |
| `POST` | `/specialty/drug-interactions` | `{"drugs": ["依那普利", "氢氯噻嗪"]}` |
| `POST` | `/specialty/guidelines` | `{"disease": "高血压", "version": "latest"}` |
| `POST` | `/specialty/document-summarize` | `{"text": "病程记录原文…"}` |
| `POST` | `/specialty/document-template` | `{"disease": "高血压"}` |
| `POST` | `/specialty/pathway` | `{"subject": "高血压"}` |
| `POST` | `/specialty/allergy-risk` | `{"drugs": ["美托洛尔"], "allergies": ["哮喘"], "population": ["老年"]}` |
| `GET` | `/specialty/operations` | 列出所有操作名 |
| `POST` | `/specialty/{operation}` | 通用入口，如 `/specialty/differential` + `{"presentation": "…"}` |

响应包含：`answer`、`disclaimer`（强制）、`citations`/`citations_markdown`（文档/章节/页码）、`structured`（带来源的本地结构化知识）、`safety_warning`、`diagnosis_warning`（疑似确诊式表述警示）。

### 4.3 Python 用法

```python
from src.rag_system import MedRAGSystem

system = MedRAGSystem()
r = system.specialty_run("differential", presentation="发热伴干咳3天", history="类风湿，长期用激素")
print(system.format_specialty_result(r))
# 所有专项操作都可经 specialty_run(op, **params) 调用
```

### 4.4 安全护栏

- 每个专项结果**强制追加**免责声明（服务层保证，不依赖模型是否遵守 Prompt）。
- 鉴别/报告/文书三类明确写入「不能作为确诊依据」「不能直接据此诊断疾病」「禁止生成完整病历」。
- 检测到「确诊为/诊断为/得了…病」等**确诊式表述**时额外给出 `diagnosis_warning`。
- 结构化知识严格模式只返回带 `source` 的条目；无索引时自动降级为仅结构化知识，不中断服务。

配置项（`.env`）：`SPECIALTY_ENABLED`、`SPECIALTY_KNOWLEDGE_STRICT`、`SPECIALTY_TOP_K`、`SPECIALTY_APPEND_DISCLAIMER`、`SPECIALTY_WARN_DIAGNOSIS_ASSERTION`。

---

## 5. 安全、合规、风控（功能三 · 重中之重）

面向医疗场景的强制安全层，所有问答经统一编排链：**前置过滤 → 隐私脱敏 → 检索 → 角色 Prompt → 审计留痕**。违规问题在检索/LLM 之前即被短路拦截，但仍写入审计。

### 5.1 前置内容过滤 + 医疗风控

`content_filter.py` 用确定性规则（关键词 + 正则，可离线测试、零额外依赖）做意图分类：

| 拦截类别 | 适用角色 | 说明 |
|----------|----------|------|
| 自杀/自残、堕胎、违禁药/毒品 | 全部 | 一律拦截，返回危机干预话术（心理援助热线 / 急诊） |
| 开处方、自行购药、自行调整剂量、确诊结论 | 患者端 | 严格拦截，引导线下就医 |
| 用药剂量/确诊式表述 | 医护端 | 不拦截，附加"仅供鉴别思路"风险提示 |

LangChain 集成：`build_content_filter_chain(role, raise_on_block=True)`，违规抛 `ContentBlockedError`；`classify_intent()` 可独立调用。

### 5.2 用户角色区分（两套问答口径）

`role_prompts.py` 提供两套 Prompt，未知角色一律按患者端（更安全）处理：

- **医护端 `clinician`**：规范医学术语、指南原文逐字引用、鉴别诊断思路（标注为"思路"而非确诊）、区分"证据支持"与"临床判断"。
- **患者端 `patient`**：通俗科普、大白话解释术语、绝不做诊断/不给剂量/不推荐自行购药、引导线下就医。

### 5.3 日志全留存（医疗审计）

`audit.py` 以 JSONL 只追加方式记录每条问答：**提问内容、回答内容、提问人（user_id）、角色、时间（UTC ISO + 时间戳）、召回文档（chunk_id/文档/章节/页码）、过滤结果**。支持按 user_id / role / session_id / 关键词 / 时间范围查询与统计。审计写入前**二次脱敏**（双保险），原始隐私绝不落盘。

### 5.4 隐私保护（前置脱敏链）

`privacy.py` 在入检索/入模型/入日志之前自动脱敏：身份证号、手机号、邮箱、银行卡号、住址、病历号、出生日期、姓名（称谓上下文 + 常见姓氏规则降低误伤）。命中即替换为 `*`（数字类定长掩码，不暴露尾号）。LangChain 前置链：`build_desensitize_chain()`。

### 5.5 幻觉抑制机制

`role_prompts.py` 的"证据铁律"强制约束：知识库无内容直接答"暂无相关权威资料，建议咨询执业医师"，**禁止编造**；不得虚构文献/页码/[来源N]；逐字引用加引号、推断标注"基于资料推断"；回答必须带真实存在的 [来源N] 引用（`citation.py` 提供文档/章节/页码原文跳转核对）。

### 5.6 用法示例

```python
from src.rag_system import MedRAGSystem

system = MedRAGSystem()
system.ingest("./data/raw")

# 患者端（严格口径 + 违规拦截 + 脱敏 + 审计）
r = system.safe_query(
    "我叫张三，身份证110101199001011234，二甲双胍一天吃几片？",
    role="patient", user_id="u1001",
)
print(r["blocked"], r["answer"])   # True（自行用药剂量+隐私命中）

# 医护端（专业口径）
r = system.safe_query("高血压的鉴别诊断思路？", role="clinician", user_id="doc01")

print(system.audit_entries(user_id="u1001"))  # 审计查询
print(system.audit_stats())                   # 审计统计
print(system.format_safe_result(r))           # 格式化输出
```

### 5.7 安全合规 API 端点

- `POST /safe/query` — 安全问答（`question` / `role` / `user_id` / `session_id`）
- `POST /safe/desensitize` — 独立脱敏端点（如粘贴病历预处理）
- `POST /safe/audit/query` — 审计日志查询（按提问人/角色/会话/关键词过滤）
- `GET  /safe/audit/stats` — 审计统计（总量、按角色、违规分类）

配置项（`.env`）：`SAFETY_FILTER_ENABLED`、`PRIVACY_DESENSITIZE_ENABLED`、`SAFETY_DEFAULT_ROLE`（patient/clinician）、`AUDIT_ENABLED`、`AUDIT_STORE_PATH`。

---

## 6. 知识库管理后台（功能四 · LangChain RAG 配套）

面向管理员的知识库运维后台，覆盖「入库 → 分类 → 版本 → 权限 → 检索调试」全生命周期。核心类 `src/knowledge_base.py::KnowledgeBaseManager`，挂载于 `MedRAGSystem.kb`。

### 6.1 文档上传与增量入库

- 支持 **PDF / Word(.docx) / Markdown / txt**（由 `DocumentProcessor` 按扩展名加载）。
- 分块器为 **RecursiveCharacterTextSplitter**，并针对医疗文档提供**章节 / 条款自定义分隔符**（内置 `MEDICAL_SEPARATORS`，从“第一章/第一条”等中文序号到章节标题、条款编号），上传时可覆盖 `separators` / `chunk_size` / `chunk_overlap`。
- 每个 chunk 打上 `doc_id` / `doc_status` / `visibility` / `kb_department` / `kb_category` 元数据，向量化后**增量写入 FAISS** 并落盘，无需全量重建。

```bash
python scripts/kb_admin.py upload data/raw/hypertension.md \
  --category 临床指南 --title 高血压指南 --version v2.0 \
  --separators 第 条 章节
```

### 6.2 批量导入 / 向量化

```bash
python scripts/kb_admin.py batch data/raw --category 内科
```

失败项逐条记录不中断；全量重建用 `python scripts/kb_admin.py reindex`。

### 6.3 分类目录

内置 9 类：**临床指南、药学、检验、内科、外科、急诊、儿科、护理、感染控制**（`config/config.yaml` 的 `knowledge_base.categories` 可扩展）。

```bash
python scripts/kb_admin.py categories
python scripts/kb_admin.py add-category 感染控制
python scripts/kb_admin.py stats        # 各类文档/chunk 统计
```

### 6.4 文档版本管理（防过时知识，关键）

- 同 `title` 再上传新版时，旧版自动 **`archived`（归档）**；元数据记录 `archived_versions` 链。
- 三态状态：`active` / `archived` / `invalid`，可手动 `archive` / `invalidate` / `reactivate`。
- **检索默认过滤失效文档**：`retriever.py` 与 `clinical_retriever.py` 在候选检索后剔除 `doc_status != active` 的 chunk；知识库检索还会先多召回 `KB_CANDIDATE_BUFFER` 条再过滤，避免 top-k 不足。`KB_FILTER_INVALID_DEFAULT=false` 可关闭（仅供调试）。

```bash
python scripts/kb_admin.py versions --title 高血压指南
python scripts/kb_admin.py invalidate doc-xxxxxxxx
python scripts/kb_admin.py reactivate  doc-xxxxxxxx
```

### 6.5 知识库权限隔离

三档可见性，检索时按 `account` / `department` 二次过滤（多召回缓冲保证过滤后仍有足够结果）：

| 可见性 | 规则 |
|--------|------|
| `public` | 全院通用，任意账号可检索 |
| `department` | 科室内部资料，仅 `kb_department` 匹配的科室可检索 |
| `confidential` | 涉密资料，仅登记表 `allowed_accounts` 中的账号可检索 |

```bash
python scripts/kb_admin.py upload secret.md --visibility confidential --allow alice bob
python scripts/kb_admin.py query "高血压用药" --account alice --department 心内科
```

### 6.6 检索测试面板

管理员输入测试问句，展示 **原始召回（含得分/状态）→ 权限+失效过滤 → 重排** 全流程，并可实时调 `chunk_size` / `top-k` / `recall_k` / `rerank` 参数对比：

```bash
python scripts/kb_admin.py debug "高血压的诊断标准是什么" \
  --top-k 3 --recall-k 10 --chunk-size 200 --account alice --rerank
```

返回字段：`recall`（召回清单，每条含 `rank`/`doc_id`/`title`/`score`/`status`/`content`）、`filtered`（过滤后）、`results`（重排后）、`filtered_count`、`params`（本次实际参数）、`chunking_preview`（分块预览）。CLI 无 rerank 依赖时自动降级为关键词/指南启发式重排。

### 6.7 Python / API 用法

```python
from src.rag_system import MedRAGSystem
system = MedRAGSystem()

system.kb.upload_file("data/raw/guide.md", category="临床指南", title="高血压指南", version="v2.0")
print(system.kb.overview())
report = system.kb.debug_retrieval("高血压诊断标准", top_k=3, use_rerank=True)
ans = system.kb_query("高血压诊断标准", account="alice", department="心内科")
```

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/kb/overview` | 知识库总览（文档数、分类、状态分布） |
| `GET` / `POST` | `/kb/categories` | 列出 / 新增分类 |
| `POST` | `/kb/upload` | 上传单篇（`title`/`category`/`visibility`/`allowed_accounts`/`version`/`separators`…） |
| `POST` | `/kb/upload/batch` | 批量导入 |
| `GET` | `/kb/documents` | 文档清单（可按 `title` / `category` 过滤） |
| `GET` | `/kb/documents/{doc_id}/versions` | 某文档版本历史 |
| `POST` | `/kb/documents/{doc_id}/status` | 设置状态（archive/invalidate/reactivate） |
| `POST` | `/kb/query` | 带权限+失效过滤的问答 |
| `POST` | `/kb/debug` | 检索测试面板（召回/过滤/重排/参数调优） |

配置项（`.env`）：`KB_ENABLED`、`KB_REGISTRY_PATH`、`KB_UPLOAD_DIR`、`KB_CANDIDATE_BUFFER`、`KB_DEFAULT_VISIBILITY`、`KB_RECALL_K`、`KB_FILTER_INVALID_DEFAULT`、`KB_DEBUG_CHUNK_SIZE`、`KB_DEBUG_TOP_K`、`KB_DEBUG_USE_RERANK`。

---

## 7. 后台运维 & 统计（功能五）

面向运维/管理员的后台能力，统一入口为 `src/ops.py::OpsManager`，挂载于 `MedRAGSystem.ops`（也可经 `system.ops_*()` 快捷方法调用）。所有问答自动写入只增不改的 JSONL 日志 `ops/query_log.jsonl`；运行时参数持久化到 `ops/runtime_params.json`，重启后自动恢复。

### 7.1 问答统计

统计高频查询病种（内置高血压/糖尿病/冠心病/脑卒中/哮喘/慢阻肺/慢性肾病/肿瘤/感染/过敏关键词表）、高频问题、用户反馈（幻觉标记数量），并区分 RAG / 纯 LLM 查询量。

```bash
python scripts/ops_admin.py stats --top-n 5            # 全量统计
python scripts/ops_admin.py stats --keyword 高血压      # 仅统计含关键词的问题
```

```python
system.ops.stats(top_n=10, keyword=None)
# => {total_queries, rag_queries, pure_llm_queries,
#     top_questions:[{question,count}], top_diseases:[{disease,count}],
#     feedback:{total, by_label, hallucination_count, negative_rate}}
```

### 7.2 模型切换（在线 / 本地医疗大模型）与 RAG 开关

支持在线模型（OpenAI 兼容 / DeepSeek）、本地医疗大模型（Ollama，如 `qwen2.5:7b`）、测试模型之间**热切换**；切换后重建所有下游链，无需重启服务。RAG 开关用于「纯 LLM 模式 vs RAG 检索增强模式」A/B 对比测试：关闭时跳过检索、注入空上下文与纯 LLM Prompt（禁止虚构出处）。

```bash
python scripts/ops_admin.py models                          # 列出可切换模型
python scripts/ops_admin.py switch-model --provider ollama --model qwen2.5:7b
python scripts/ops_admin.py switch-model --provider openai --model deepseek-chat --temperature 0.3
python scripts/ops_admin.py rag --off                       # 纯 LLM 模式
python scripts/ops_admin.py rag --on                        # RAG 模式
python scripts/ops_admin.py compare "高血压的诊断标准是什么？"  # A/B 对比
```

`compare` 返回同一问题的两组答案与命中来源数（`pure_llm.source_count` 通常为 0，`rag.source_count` 为召回数），测试后自动恢复原 RAG 状态，两次调用均记入统计日志。

### 7.3 参数可视化配置

运行时可调并持久化：`chunk_size`、`chunk_overlap`、`top_k`、`rerank_threshold`（0-1，批内归一化后过滤低分候选）、`use_hyde`、`temperature`、`rag_enabled`。`get_params()` 返回每个参数的当前值/类型/说明/默认值，便于前端渲染配置面板。

```bash
python scripts/ops_admin.py params
python scripts/ops_admin.py set-param --chunk-size 300 --chunk-overlap 50 \
  --top-k 5 --rerank-threshold 0.6 --use-hyde --temperature 0.3
python scripts/ops_admin.py reset-params
```

```python
system.ops.set_params(chunk_size=300, top_k=5, use_hyde=True, temperature=0.3)
system.ops.get_params()    # 可视化配置元数据
system.ops.reset_params()  # 恢复初始值
```

### 7.4 会话管理

查看历史会话元数据（不裸露完整正文）、查看单会话完整对话、手动删除敏感会话、批量清理敏感会话。含手机号/身份证等问题会自动标记为敏感会话。

```bash
python scripts/ops_admin.py sessions --include-expired
python scripts/ops_admin.py session <session_id>
python scripts/ops_admin.py delete-session <session_id>
python scripts/ops_admin.py purge-sensitive
```

### 7.5 API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/ops/stats` | 问答统计（高频病种/问题/幻觉数） |
| `GET` | `/ops/models` | 可切换模型列表 |
| `GET` | `/ops/model/current` | 当前模型与 RAG 状态 |
| `POST` | `/ops/model/switch` | 热切换 LLM |
| `POST` | `/ops/rag` | 开关 RAG（纯 LLM vs RAG） |
| `GET` / `POST` | `/ops/params` | 查看 / 设置运行时参数 |
| `POST` | `/ops/params/reset` | 重置参数 |
| `POST` | `/ops/compare` | 纯 LLM vs RAG A/B 对比 |
| `GET` | `/ops/sessions` | 会话列表 |
| `GET` / `DELETE` | `/ops/sessions/{id}` | 查看 / 删除会话 |
| `POST` | `/ops/sessions/purge-sensitive` | 批量清理敏感会话 |

配置项（`.env`）：`OPS_ENABLED`、`OPS_QUERY_LOG_PATH`、`OPS_PARAM_STORE_PATH`、`RAG_ENABLED`、`RERANK_THRESHOLD`。

---

## 7b. 统一管理后端（五个功能一站式管理 API）

新增 `/admin/*` 命名空间，把前面五个功能的管理能力收敛到一个**带鉴权的 HTTP API**，便于运维后台 / 脚本统一调用（不提供前端界面）。

### 鉴权

- 所有 `/admin` 端点要求在请求头携带 `Authorization: Bearer <ADMIN_API_KEY>`。
- `.env` 中 `ADMIN_API_KEY` 留空时，管理 API 直接返回 `503`（拒绝裸奔），请务必设置。
- 每次写操作都会记入 `ADMIN_ACTION_LOG_PATH`（默认 `./admin/action_log.jsonl`，JSONL 只增不改），可通过 `/admin/actions` 查询。
- 配置中的 API Key 等敏感字段**只回显掩码**（如 `****key`），绝不回传明文。

### 端点总览

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/admin/overview` | 系统总览 + 五个功能状态摘要 |
| `GET` | `/admin/features` | 五个功能目录 |
| `GET` | `/admin/features/{feature}` | 单个功能开关与运行态 |
| `GET` | `/admin/features/{feature}/stats` | 单个功能统计 |
| `GET` / `POST` | `/admin/features/{feature}/config` | 查看 / 更新功能配置（白名单字段，敏感值掩码） |
| `GET` | `/admin/actions` | 管理操作日志 |
| `POST` | `/admin/actions/rebuild-index` | 功能一：重建向量索引 |
| `GET` | `/admin/specialty/operations` | 功能二：9 个专项操作及参数清单 |
| `GET` | `/admin/specialty/knowledge` | 功能二：结构化知识目录概览 |
| `POST` | `/admin/safety/audit/query` | 功能三：审计日志查询 |
| `GET` | `/admin/safety/audit/stats` | 功能三：审计统计 |
| `GET` | `/admin/kb/documents` | 功能四：文档列表（可按状态/分类过滤） |
| `POST` | `/admin/kb/documents/{id}/status` | 功能四：变更文档状态 |
| `POST` | `/admin/kb/debug` | 功能四：检索调试 |
| `POST` | `/admin/ops/model/switch` | 功能五：热切换模型 |
| `POST` | `/admin/ops/rag` | 功能五：开关 RAG |
| `POST` | `/admin/ops/params` / `.../reset` | 功能五：设置 / 重置运行时参数 |
| `GET` / `DELETE` | `/admin/ops/sessions[/{id}]` | 功能五：会话管理 |
| `POST` | `/admin/ops/sessions/purge-sensitive` | 功能五：清理敏感会话 |

其中 `{feature}` 取值为 `rag` / `specialty` / `safety` / `kb` / `ops`。

### 调用示例

```bash
# 总览
curl -H "Authorization: Bearer $ADMIN_API_KEY" http://localhost:8000/admin/overview

# 查看 RAG 配置（API Key 自动掩码）
curl -H "Authorization: Bearer $ADMIN_API_KEY" http://localhost:8000/admin/features/rag/config

# 更新专项功能配置
curl -X POST -H "Authorization: Bearer $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"updates":{"specialty_top_k":8}}' \
  http://localhost:8000/admin/features/specialty/config

# 运维：关闭 RAG（纯 LLM 模式）
curl -X POST -H "Authorization: Bearer $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"enabled":false}' http://localhost:8000/admin/ops/rag
```

实现见 `src/admin.py`（管理门面 `AdminManager`）与 `src/admin_api.py`（HTTP 路由）。

### 管理后台网页（网站）

系统自带一个**零构建依赖的单页管理网站**，由 FastAPI 直接托管：

- 访问地址：<http://localhost:8000/admin/ui>
- 页面文件：`web/admin.html`（纯 HTML/CSS/JS，无需 npm/构建步骤）
- 登录：在页面输入 `ADMIN_API_KEY`，前端通过 `/admin/*` API 取数（密钥存于 sessionStorage）
- 功能：系统总览、五大功能各自的管理面板（状态/统计/配置/快捷操作）、操作日志查询

支持的操作示例：
- 功能一：重建向量索引、查看/修改 RAG 配置
- 功能二：查看 9 项专项操作、结构化知识目录、修改专项配置
- 功能三：审计日志关键词/角色查询、审计统计
- 功能四：文档列表与状态变更、检索调试
- 功能五：模型热切换、RAG 开关、运行时参数、会话管理/清理

> 安全提示：本网站仅应在内网/受控环境使用；生产环境请配合 HTTPS 与网络访问控制。

---

## 8. Python API 用法（基础）

```python
from src.rag_system import MedRAGSystem

system = MedRAGSystem()
system.ingest("./data/raw")                 # 构建索引（首次）

result = system.query("高血压的用药原则？")
print(result["answer"])                     # 答案文本
print(result["source_documents"])           # 检索到的来源文档
print(result["safety_warning"])             # 安全提示（如有）

print(system.format_result(result))         # 格式化完整输出
```

---

## 10. 检索策略

在 `.env` 中配置 `RETRIEVAL_STRATEGY`：

| 策略 | 说明 |
|------|------|
| `similarity` | 余弦相似度 top-k |
| `mmr` | 最大边际相关性，兼顾相关性与多样性（默认） |
| `similarity_score_threshold` | 带相似度阈值过滤 |
| `hybrid` | 向量 + BM25 关键词，RRF 融合（需 `pip install rank-bm25`） |

相关参数：`TOP_K`、`FETCH_K`、`MMR_LAMBDA`、`SCORE_THRESHOLD`。

---

## 11. 离线测试模式（无需 API Key）

把 `.env` 中设为：

```env
LLM_PROVIDER=fake
EMBEDDING_PROVIDER=fake
```

即可用确定性哈希向量和假模型跑通全流程，用于验证代码正确性（结果无真实语义）。

运行测试：

```bash
pytest tests/ -v
```

测试覆盖：文本清洗、安全护栏、文档切分、引用溯源、原文跳转、双模式检索、会话记忆（TTL/敏感清理）、反馈统计与调优、基础 RAG 链路、完整临床问答链路，功能二的 7 大专项功能（9 个操作、强制免责声明、来源标注、版本区分、联用筛查、确诊式表述警示、文书法律边界），功能三的安全合规（隐私脱敏、违规拦截与角色区分、双角色 Prompt/幻觉抑制、审计留痕、端到端安全问答），以及功能四的知识库管理（上传与增量入库、医疗分隔符、批量导入、分类目录、版本归档与失效过滤、权限隔离、带 ACL 问答、检索测试面板、系统整合与 API 路由），以及功能五的后台运维（问答统计/高频病种/幻觉标记、模型热切换、RAG 开关与 A/B 对比、参数配置与持久化恢复、会话查看/删除/敏感清理、API 端点）（共 79 项）。

---

## 12. 切换为本地模型（Ollama）

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
```

并安装 `pip install langchain-ollama`，启动 Ollama 服务即可，无需任何云 API Key。

---

## 13. 目录结构

```
rag/
├── config/config.yaml        # YAML 默认配置
├── data/raw/                 # 示例医学文档
├── scripts/
│   ├── ingest.py             # 构建索引脚本
│   ├── query.py              # 基础问答脚本
│   ├── clinical.py           # 临床增强问答脚本（双模式/记忆/反馈）
│   ├── specialty.py          # 临床医护专项辅助功能脚本（7 大功能/9 操作）
│   ├── kb_admin.py           # 知识库管理后台 CLI（功能四：上传/分类/版本/权限/调试）
│   └── ops_admin.py          # 后台运维管理 CLI（功能五：统计/模型/参数/会话/对比）
├── knowledge_base/           # 功能四运行时数据（registry.json + uploads/）
├── ops/                      # 功能五运行时数据（query_log.jsonl + runtime_params.json）
├── src/                      # 核心代码（见上表）
├── tests/                    # 测试（含 test_knowledge_base.py / test_ops_features.py）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 14. 生产化建议

1. **合规**：医疗数据需符合当地隐私法规（如 HIPAA / 个人信息保护法），避免上传敏感病历到公共 API。
2. **重排序**：安装 `sentence-transformers` 后使用 Cross-Encoder（如 `BAAI/bge-reranker-base`）提升检索精度；未安装时自动降级为 LLM 打分或关键词启发式。
3. **知识图谱**：结合医学知识图谱做多跳推理。
4. **评测**：引入 RAGAS 等指标做自动化质量评估。
5. **引用校验**：对模型输出做引用一致性校验，防止幻觉引用。

---

**免责声明**：本项目为技术演示，所有医学内容均为简化示例，不构成任何医疗建议。请使用正规、权威、经审核的医学知识库作为生产数据源。
