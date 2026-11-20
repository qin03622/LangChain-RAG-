"""FastAPI 服务：健康检查、索引构建、基础问答、临床增强问答、反馈。

启动:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

端点:
    GET  /health                    健康检查
    POST /ingest                    构建/增量更新索引
    POST /query                     基础问答（结构化结果）
    POST /query/simple              基础问答（仅答案文本）
    POST /clinical/query            临床问答（双模式 + 记忆 + 引用 + Rerank）
    POST /clinical/feedback         提交医生反馈
    GET  /clinical/session/{id}/original/{chunk_id}  查看原文片段
    GET  /clinical/feedback/stats   反馈统计
    POST /clinical/session          新建会话
    POST /specialty/{operation}     专项辅助统一入口（见 /specialty/operations）
    GET  /specialty/operations      列出全部专项操作及所需参数
    POST /specialty/differential    1. 症状鉴别辅助
    POST /specialty/lab-interpret   2. 检验/检查报告解读
    POST /specialty/drug-info       3a. 单药药学信息
    POST /specialty/drug-interactions 3b. 联用风险筛查
    POST /specialty/guidelines      4. 指南共识检索
    POST /specialty/document-summarize 5a. 病例要点提炼
    POST /specialty/document-template  5b. 书写要点模板
    POST /specialty/pathway         6. 临床路径/操作规范
    POST /specialty/allergy-risk    7. 不良反应/过敏风险
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_system import MedRAGSystem
from .utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="MedRAG 医疗临床智能诊疗问答系统",
    description="基于 LangChain 的 RAG 系统：双模式临床问答、引用溯源、多轮记忆、反馈调优，含统一管理后端 /admin。",
    version="1.2.0",
)

# 统一管理后端：把五个功能的管理能力收敛到带鉴权的 /admin 命名空间
from .admin_api import register as register_admin_routes  # noqa: E402

register_admin_routes(app)

_system: Optional[MedRAGSystem] = None


def get_system() -> MedRAGSystem:
    global _system
    if _system is None:
        _system = MedRAGSystem()
    return _system


# ---------- Schema ----------
class HealthResponse(BaseModel):
    status: str
    version: str
    index_ready: bool


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="文档目录，默认取配置 DATA_DIR")
    rebuild: bool = Field(default=True, description="是否重建索引")


class IngestResponse(BaseModel):
    chunks: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: Optional[int] = Field(default=None, description="覆盖配置的检索数量")


class ClinicalQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Optional[str] = Field(default=None, description="precise 精准检索 | comprehensive 综合推理")
    session_id: Optional[str] = Field(default=None, description="多轮会话 ID，不传则新建")
    use_history: bool = Field(default=True, description="是否带上多轮上下文")


class CitationItem(BaseModel):
    index: int
    doc_title: str
    source: str
    section: str
    page: int
    chunk_id: str
    content: str


class ClinicalQueryResponse(BaseModel):
    answer: str
    mode: str
    session_id: str
    citations: list[CitationItem]
    citations_markdown: str
    safety_passed: bool
    safety_warning: str
    disclaimer: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    label: str = Field(..., description="correct | hallucination | outdated | incomplete")
    session_id: Optional[str] = None
    mode: Optional[str] = None
    chunk_ids: list[str] = Field(default_factory=list)
    note: str = ""


class FeedbackResponse(BaseModel):
    status: str
    id: Optional[str] = None
    message: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str


# ---------- 功能二：专项辅助 Schema ----------
class SpecialtyBaseRequest(BaseModel):
    """专项辅助请求公共字段。"""
    pass


class DifferentialRequest(SpecialtyBaseRequest):
    presentation: str = Field(..., min_length=1, description="主诉 + 体征")
    history: str = Field(default="", description="基础病史")


class LabInterpretRequest(SpecialtyBaseRequest):
    items: list[str] = Field(..., min_length=1, description="待解读指标/术语列表")


class DrugInfoRequest(SpecialtyBaseRequest):
    drug: str = Field(..., min_length=1, description="药品名称")
    question: str = Field(default="", description="具体关注点（可选）")


class DrugInteractionsRequest(SpecialtyBaseRequest):
    drugs: list[str] = Field(..., min_length=2, description="待筛查药品列表（至少 2 种）")


class GuidelineSearchRequest(SpecialtyBaseRequest):
    disease: str = Field(..., min_length=1, description="疾病/主题")
    version: str = Field(default="latest", description="latest 最新版 | all 全部版本")


class DocumentSummarizeRequest(SpecialtyBaseRequest):
    text: str = Field(..., min_length=1, description="病程记录原文")


class DocumentTemplateRequest(SpecialtyBaseRequest):
    disease: str = Field(..., min_length=1, description="疾病名称")


class PathwayRequest(SpecialtyBaseRequest):
    subject: str = Field(..., min_length=1, description="疾病/操作名称")
    aspect: str = Field(default="全部", description="关注环节")


class AllergyRiskRequest(SpecialtyBaseRequest):
    drugs: list[str] = Field(default_factory=list, description="用药列表")
    allergies: list[str] = Field(default_factory=list, description="过敏史")
    population: list[str] = Field(default_factory=list, description="特殊人群：孕哺/儿童/老年/肾功能/肝功能")


class SpecialtyOperationRequest(BaseModel):
    operation: str = Field(..., description="操作名")
    params: dict[str, Any] = Field(default_factory=dict, description="操作参数")


class SpecialtyResponse(BaseModel):
    feature: str
    answer: str
    disclaimer: str
    citations: list[CitationItem]
    citations_markdown: str
    structured: dict[str, Any]
    safety_warning: str
    diagnosis_warning: str


# ---------- 功能三：安全 / 合规 / 风控 Schema ----------
class SafeQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    role: str = Field(default="patient", description="patient 患者端（严格）| clinician 医护端（专业）")
    user_id: str = Field(default="anonymous", description="提问人标识（写入审计）")
    session_id: str = Field(default="", description="多轮会话 ID")
    use_history: bool = Field(default=False, description="是否附带历史（由 session_id + 记忆提供）")


class SafeQueryResponse(BaseModel):
    answer: str
    blocked: bool
    role: str
    filter: dict[str, Any]
    privacy_hits: list[str]
    audit_id: str = ""
    citations: list[CitationItem]
    citations_markdown: str
    safety_warning: str
    disclaimer: str


class AuditQueryRequest(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    session_id: Optional[str] = None
    keyword: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)


class DesensitizeRequest(BaseModel):
    text: str = Field(..., description="待脱敏文本（如粘贴的病历）")


class DesensitizeResponse(BaseModel):
    text: str
    hits: list[str]
    changed: bool
    mask_count: int


# ---------- 基础端点 ----------
@app.get("/health", response_model=HealthResponse)
def health():
    system = get_system()
    return HealthResponse(status="ok", version="1.1.0", index_ready=system.vsm.exists())


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    system = get_system()
    count = system.ingest(req.directory, rebuild=req.rebuild)
    return IngestResponse(chunks=count, message=f"已处理 {count} 个文本块。")


@app.post("/query")
def query(req: QueryRequest):
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先调用 /ingest。")
    result = system.query(req.question)
    sources = []
    for i, doc in enumerate(result.get("source_documents") or [], 1):
        sources.append(
            {
                "index": i,
                "source": doc.metadata.get("doc_title") or doc.metadata.get("filename", "未知"),
                "section": doc.metadata.get("section", ""),
                "page": doc.metadata.get("page", -1),
                "content": doc.page_content,
            }
        )
    return {
        "answer": result.get("answer", ""),
        "sources": sources,
        "safety_passed": result.get("safety_passed", True),
        "safety_warning": result.get("safety_warning", ""),
        "disclaimer": result.get("disclaimer", ""),
    }


@app.post("/query/simple")
def query_simple(req: QueryRequest):
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先调用 /ingest。")
    return {"answer": system.query(req.question)["answer"]}


# ---------- 临床增强端点 ----------
@app.post("/clinical/session", response_model=SessionResponse)
def new_session():
    """新建多轮会话，返回 session_id。"""
    return SessionResponse(session_id=get_system().new_session())


@app.post("/clinical/query", response_model=ClinicalQueryResponse)
def clinical_query(req: ClinicalQueryRequest):
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先调用 /ingest。")
    result = system.clinical_query(
        req.question, mode=req.mode, session_id=req.session_id, use_history=req.use_history
    )
    return ClinicalQueryResponse(
        answer=result["answer"],
        mode=result["mode"],
        session_id=result["session_id"],
        citations=[CitationItem(**c) for c in result["citations"]],
        citations_markdown=result["citations_markdown"],
        safety_passed=result["safety_passed"],
        safety_warning=result["safety_warning"],
        disclaimer=result["disclaimer"],
    )


@app.post("/clinical/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    system = get_system()
    result = None
    if req.session_id or req.chunk_ids:
        result = {
            "citations": [{"chunk_id": c} for c in req.chunk_ids],
            "mode": req.mode or "",
            "session_id": req.session_id or "",
        }
    try:
        fb = system.submit_feedback(
            req.question, req.answer, req.label, result=result, note=req.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if fb.get("status") == "disabled":
        return FeedbackResponse(status="disabled", message="反馈功能未启用")
    return FeedbackResponse(status="ok", id=fb.get("id"))


@app.get("/clinical/feedback/stats")
def feedback_stats():
    return get_system().feedback_stats()


@app.get("/clinical/session/{session_id}/original/{chunk_id}")
def get_original(session_id: str, chunk_id: str):
    """点击引用跳转：查看该 chunk 的完整原文片段。

    由于原文片段存在检索结果中而非会话里，这里直接从向量库按 chunk_id 查找。
    """
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先调用 /ingest。")
    from langchain_core.documents import Document

    all_docs = list(system.vsm.store.docstore._dict.values())  # type: ignore[attr-defined]
    from .citation import get_original_fragment

    frag = get_original_fragment(all_docs, chunk_id)
    if not frag:
        raise HTTPException(status_code=404, detail="未找到该原文片段。")
    return frag


# ---------- 功能二：面向临床医护的专项辅助端点 ----------
def _specialty_response(result: dict[str, Any]) -> SpecialtyResponse:
    return SpecialtyResponse(
        feature=result["feature"],
        answer=result["answer"],
        disclaimer=result["disclaimer"],
        citations=[CitationItem(**c) for c in result.get("citations", [])],
        citations_markdown=result.get("citations_markdown", ""),
        structured=result.get("structured", {}),
        safety_warning=result.get("safety_warning", ""),
        diagnosis_warning=result.get("diagnosis_warning", ""),
    )


def _run_specialty(operation: str, params: dict[str, Any]) -> SpecialtyResponse:
    system = get_system()
    try:
        result = system.specialty_run(operation, **params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return _specialty_response(result)


SPECIALTY_OPERATIONS: dict[str, list[str]] = {
    "differential": ["presentation", "history"],
    "lab_interpret": ["items"],
    "drug_info": ["drug", "question"],
    "drug_interactions": ["drugs"],
    "guideline_search": ["disease", "version"],
    "document_summarize": ["text"],
    "document_template": ["disease"],
    "pathway": ["subject", "aspect"],
    "allergy_risk": ["drugs", "allergies", "population"],
}


@app.get("/specialty/operations")
def specialty_operations():
    """列出全部专项功能操作及其参数名。"""
    return {
        "operations": [
            {"operation": op, "params": params} for op, params in SPECIALTY_OPERATIONS.items()
        ],
        "disclaimer": "所有专项功能仅供鉴别/参考思路，不能作为确诊依据，须由执业医师确认。",
    }


@app.post("/specialty/differential", response_model=SpecialtyResponse)
def specialty_differential(req: DifferentialRequest):
    """1. 症状鉴别辅助（强制标注“仅供鉴别思路，不能作为确诊依据”）。"""
    return _run_specialty("differential", req.model_dump())


@app.post("/specialty/lab-interpret", response_model=SpecialtyResponse)
def specialty_lab_interpret(req: LabInterpretRequest):
    """2. 检验/检查报告解读（不下疾病结论）。"""
    return _run_specialty("lab_interpret", {"items": req.items})


@app.post("/specialty/drug-info", response_model=SpecialtyResponse)
def specialty_drug_info(req: DrugInfoRequest):
    """3a. 药品药学信息。"""
    return _run_specialty("drug_info", req.model_dump())


@app.post("/specialty/drug-interactions", response_model=SpecialtyResponse)
def specialty_drug_interactions(req: DrugInteractionsRequest):
    """3b. 联用风险筛查。"""
    return _run_specialty("drug_interactions", {"drugs": req.drugs})


@app.post("/specialty/guidelines", response_model=SpecialtyResponse)
def specialty_guidelines(req: GuidelineSearchRequest):
    """4. 指南/共识检索（含版本与推荐等级）。"""
    return _run_specialty("guideline_search", req.model_dump())


@app.post("/specialty/document-summarize", response_model=SpecialtyResponse)
def specialty_document_summarize(req: DocumentSummarizeRequest):
    """5a. 病例要点提炼。"""
    return _run_specialty("document_summarize", {"text": req.text})


@app.post("/specialty/document-template", response_model=SpecialtyResponse)
def specialty_document_template(req: DocumentTemplateRequest):
    """5b. 书写要点模板（禁止生成完整病历）。"""
    return _run_specialty("document_template", {"disease": req.disease})


@app.post("/specialty/pathway", response_model=SpecialtyResponse)
def specialty_pathway(req: PathwayRequest):
    """6. 临床路径/操作规范查询。"""
    return _run_specialty("pathway", req.model_dump())


@app.post("/specialty/allergy-risk", response_model=SpecialtyResponse)
def specialty_allergy_risk(req: AllergyRiskRequest):
    """7. 不良反应/过敏与特殊人群风险提示。"""
    return _run_specialty("allergy_risk", req.model_dump())


# 通用入口放在最后，避免 /specialty/{operation} 抢占上面具体路径
@app.post("/specialty/{operation}")
def specialty_generic(operation: str, params: dict[str, Any] | None = Body(default=None)):
    """专项辅助统一入口：路径传 operation，请求体直接给参数字典。"""
    if operation not in SPECIALTY_OPERATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"未知专项操作: {operation}；可选: {', '.join(SPECIALTY_OPERATIONS)}",
        )
    return _run_specialty(operation, params or {})


# ---------- 功能三：安全 / 合规 / 风控端点 ----------
@app.post("/safe/query", response_model=SafeQueryResponse)
def safe_query(req: SafeQueryRequest):
    """安全问答：前置过滤 → 脱敏 → 角色口径 → 引用溯源 → 审计留痕。

    患者端（patient）严格拦截开处方/确诊/自行用药剂量及高危问题；
    医护端（clinician）使用专业口径，但仍不下确诊结论、不开处方。
    """
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先调用 /ingest。")
    if req.role not in ("patient", "clinician"):
        raise HTTPException(status_code=400, detail="role 只能是 patient 或 clinician。")

    history_text = ""
    if req.use_history and req.session_id:
        history_text = system.memory.get_context_text(req.session_id)

    result = system.safe_query(
        req.question,
        role=req.role,
        user_id=req.user_id,
        session_id=req.session_id,
        history_text=history_text,
    )
    return SafeQueryResponse(
        answer=result["answer"],
        blocked=result["blocked"],
        role=result["role"],
        filter=result["filter"],
        privacy_hits=result["privacy_hits"],
        audit_id=result.get("audit_id", ""),
        citations=[CitationItem(**c) for c in result.get("citations", [])],
        citations_markdown=result.get("citations_markdown", ""),
        safety_warning=result.get("safety_warning", ""),
        disclaimer=result.get("disclaimer", ""),
    )


@app.post("/safe/desensitize", response_model=DesensitizeResponse)
def desensitize_text(req: DesensitizeRequest):
    """隐私脱敏独立端点：把病历/提问中的姓名、手机、身份证、住址替换为掩码。"""
    from .privacy import desensitize

    result = desensitize(req.text)
    return DesensitizeResponse(
        text=result.text,
        hits=result.hits,
        changed=result.changed,
        mask_count=len(result.details),
    )


@app.post("/safe/audit/query")
def audit_query(req: AuditQueryRequest):
    """审计日志查询（支持按提问人/角色/会话/关键词过滤）。"""
    return {
        "records": get_system().audit_entries(
            user_id=req.user_id,
            role=req.role,
            session_id=req.session_id,
            keyword=req.keyword,
            limit=req.limit,
        )
    }


@app.get("/safe/audit/stats")
def audit_stats():
    """审计统计：总量、按角色分布、违规拦截分类。"""
    return get_system().audit_stats()


# ---------- 功能四：知识库管理后台端点 ----------
class KBUploadRequest(BaseModel):
    """文档上传（路径方式）请求。"""
    path: str = Field(..., description="服务器端文档绝对路径")
    category: str = Field(default="临床指南", description="知识分类")
    title: Optional[str] = Field(default=None, description="文档标题，默认取文件名")
    version: str = Field(default="v1.0", description="版本号")
    visibility: str = Field(default="public", description="public | department | confidential")
    allowed_accounts: list[str] = Field(default_factory=list, description="可见账号/科室白名单")
    replace_latest: bool = Field(default=True, description="是否归档同名旧版")
    separators: Optional[list[str]] = Field(default=None, description="自定义分块分隔符")
    chunk_size: Optional[int] = Field(default=None, description="本次分块大小")
    chunk_overlap: Optional[int] = Field(default=None, description="本次块重叠")


class KBStatusRequest(BaseModel):
    status: str = Field(..., description="active | archived | invalid")


class KBQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    account: Optional[str] = Field(default=None, description="检索账号（权限隔离）")
    department: Optional[str] = Field(default=None, description="检索科室（权限隔离）")
    category: Optional[str] = Field(default=None, description="限定分类")
    top_k: Optional[int] = Field(default=None)


class KBDebugRequest(BaseModel):
    question: str = Field(..., min_length=1)
    account: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    top_k: Optional[int] = None
    recall_k: Optional[int] = None
    use_rerank: Optional[bool] = None
    chunk_size: Optional[int] = None
    include_invalid: bool = False


class KBCategoryRequest(BaseModel):
    name: str = Field(..., min_length=1)


def _kb():
    return get_system().kb


@app.get("/kb/overview")
def kb_overview():
    """知识库概览：文档数、分块数、分类分布、索引状态。"""
    return _kb().overview()


@app.get("/kb/categories")
def kb_categories():
    return {"categories": _kb().category_stats()}


@app.post("/kb/categories")
def kb_add_category(req: KBCategoryRequest):
    return {"name": req.name, "added": _kb().add_category(req.name)}


@app.post("/kb/upload")
def kb_upload(req: KBUploadRequest):
    """文档上传与增量入库（自动分块 + 向量化 + 旧版归档）。"""
    try:
        return _kb().upload_file(
            req.path,
            category=req.category,
            title=req.title,
            version=req.version,
            visibility=req.visibility,
            allowed_accounts=req.allowed_accounts,
            replace_latest=req.replace_latest,
            separators=req.separators,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/kb/upload/batch")
def kb_upload_batch(reqs: list[KBUploadRequest]):
    """批量导入多份文档。"""
    files = [
        (r.path, {
            "category": r.category, "title": r.title, "version": r.version,
            "visibility": r.visibility, "allowed_accounts": r.allowed_accounts,
            "replace_latest": r.replace_latest, "separators": r.separators,
            "chunk_size": r.chunk_size, "chunk_overlap": r.chunk_overlap,
        })
        for r in reqs
    ]
    return {"results": _kb().upload_batch(files)}


@app.get("/kb/documents")
def kb_list_documents(status: Optional[str] = None, category: Optional[str] = None):
    """列出全部登记文档，可按状态/分类过滤。"""
    docs = list(_kb().documents.values())
    if status:
        docs = [d for d in docs if d["status"] == status]
    if category:
        docs = [d for d in docs if d["category"] == category]
    return {"documents": docs, "total": len(docs)}


@app.get("/kb/documents/{doc_id}/versions")
def kb_versions(doc_id: str):
    """查看某文档的全部版本。"""
    return {"versions": _kb().list_versions(doc_id=doc_id)}


@app.post("/kb/documents/{doc_id}/status")
def kb_set_status(doc_id: str, req: KBStatusRequest):
    """归档 / 失效 / 重新激活文档（检索默认过滤失效文档）。"""
    try:
        return _kb().set_status(doc_id, req.status)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/kb/query")
def kb_query(req: KBQueryRequest):
    """带版本失效过滤 + 权限隔离的知识库问答。"""
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先入库。")
    return system.kb_query(
        req.question, account=req.account, department=req.department,
        category=req.category, top_k=req.top_k,
    )


@app.post("/kb/debug")
def kb_debug(req: KBDebugRequest):
    """检索测试面板：召回片段 → 过滤 → 重排，调试 chunk/top-k/rerank。"""
    if not get_system().vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先入库。")
    return _kb().debug_retrieval(
        req.question,
        account=req.account,
        department=req.department,
        category=req.category,
        top_k=req.top_k,
        recall_k=req.recall_k,
        use_rerank=req.use_rerank,
        chunk_size=req.chunk_size,
        include_invalid=req.include_invalid,
    )


# =====================================================================
# 功能五：后台运维 & 统计
# =====================================================================

class OpsStatsRequest(BaseModel):
    top_n: int = Field(default=10, ge=1, le=100, description="高频榜返回条数")
    keyword: Optional[str] = Field(default=None, description="仅统计含该关键词的问题")


class OpsSwitchModelRequest(BaseModel):
    provider: Optional[str] = Field(default=None, description="openai / ollama / fake")
    model: Optional[str] = Field(default=None, description="模型名")
    temperature: Optional[float] = Field(default=None, ge=0, le=2, description="生成温度")


class OpsRagRequest(BaseModel):
    enabled: bool = Field(description="true=RAG 检索增强；false=纯 LLM 模式")


class OpsParamsRequest(BaseModel):
    params: dict[str, Any] = Field(
        description="参数字典：chunk_size/chunk_overlap/top_k/rerank_threshold/use_hyde/temperature"
    )


class OpsCompareRequest(BaseModel):
    question: str = Field(min_length=1, description="对比测试问题")
    mode: Optional[str] = Field(default=None, description="临床检索模式 precise/comprehensive")
    session_id: Optional[str] = Field(default=None, description="可选会话 ID")


def _ops():
    return get_system().ops


@app.get("/ops/stats")
def ops_stats(top_n: int = 10, keyword: Optional[str] = None):
    """问答统计：高频查询病种 / 高频问题 / 用户反馈（幻觉标记数量）。"""
    return _ops().stats(top_n=top_n, keyword=keyword)


@app.get("/ops/models")
def ops_list_models():
    """列出可切换的模型（在线 / 本地医疗大模型 / 测试）。"""
    return _ops().list_models()


@app.get("/ops/model/current")
def ops_current_model():
    """查看当前生效的模型与 RAG 开关状态。"""
    return _ops().current_model()


@app.post("/ops/model/switch")
def ops_switch_model(req: OpsSwitchModelRequest):
    """热切换 LLM（provider/model/temperature 至少给一个）。"""
    try:
        return _ops().switch_model(
            req.provider, req.model, temperature=req.temperature
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ops/rag")
def ops_set_rag(req: OpsRagRequest):
    """开关 RAG：关闭后进入纯 LLM 模式。"""
    return _ops().set_rag(req.enabled)


@app.get("/ops/params")
def ops_get_params():
    """查看当前可调参数及元数据（供前端可视化配置）。"""
    return _ops().get_params()


@app.post("/ops/params")
def ops_set_params(req: OpsParamsRequest):
    """运行时设置参数（仅白名单字段），校验后生效并持久化。"""
    try:
        return _ops().set_params(**req.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ops/params/reset")
def ops_reset_params():
    """重置所有运行时参数覆盖，恢复默认值。"""
    return _ops().reset_params()


@app.post("/ops/compare")
def ops_compare(req: OpsCompareRequest):
    """A/B 对比测试：同一问题分别用纯 LLM 与 RAG 模式作答。"""
    system = get_system()
    if not system.vsm.exists():
        raise HTTPException(status_code=409, detail="向量索引不存在，请先 ingest。")
    try:
        return _ops().compare(
            req.question, mode=req.mode, session_id=req.session_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/ops/sessions")
def ops_list_sessions(include_expired: bool = False):
    """查看历史会话列表（不返回完整消息正文，避免敏感内容裸露）。"""
    return _ops().list_sessions(include_expired=include_expired)


@app.get("/ops/sessions/{session_id}")
def ops_get_session(session_id: str):
    """查看单个会话完整对话记录（运维审计用）。"""
    try:
        return _ops().get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/ops/sessions/{session_id}")
def ops_delete_session(session_id: str):
    """手动删除会话（敏感会话立即丢弃并删除磁盘持久化）。"""
    try:
        return _ops().delete_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/ops/sessions/purge-sensitive")
def ops_purge_sensitive():
    """批量清理所有被标记为敏感的会话。"""
    return _ops().purge_sensitive_sessions()
