"""统一管理后端 HTTP 路由（/admin/*）。

把五个功能的管理能力收敛到一个带鉴权的命名空间，供运维后台 / 脚本统一调用：

  鉴权：所有 /admin 端点要求请求头 ``Authorization: Bearer <ADMIN_API_KEY>``。
        ADMIN_API_KEY 为空时服务返回 503，避免裸奔。

  路由分组：
    /admin/overview                     总览
    /admin/features                     五个功能目录
    /admin/features/{feature}           单个功能开关与运行态
    /admin/features/{feature}/stats     单个功能统计
    /admin/features/{feature}/config    单个功能配置（读/写，敏感值掩码）
    /admin/actions                      管理操作日志
    /admin/actions/rebuild-index        功能一：重建索引
    /admin/specialty/operations         功能二：9 个专项操作清单
    /admin/specialty/knowledge          功能二：结构化知识目录
    /admin/safety/audit                 功能三：审计查询/统计
    /admin/kb/documents                 功能四：文档列表/状态变更
    /admin/kb/debug                     功能四：检索调试
    /admin/ops/*                        功能五：模型/RAG/参数/会话管理

注册方式：``register(app)`` 把路由**平铺**注册到 FastAPI app（而非 include_router），
    这样 app.routes 中每条都是普通 APIRoute，便于现有测试按 ``r.path`` 内省。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .admin import FEATURE_IDS, AdminManager
from .config import ROOT_DIR, get_settings
from .utils import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False, description="管理 API 访问密钥（Bearer Token）")


# ---------- 鉴权与依赖 ----------
def require_admin_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """校验管理 API Key；返回调用方标识（actor）用于操作日志。"""
    expected = get_settings().admin_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理 API 未配置密钥（ADMIN_API_KEY 为空），请先设置 ADMIN_API_KEY。",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少管理 API 密钥，请使用 Authorization: Bearer <key>。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="管理 API 密钥无效。"
        )
    return f"key:***{expected[-4:]}"


def get_admin(actor: str = Depends(require_admin_key)) -> AdminManager:
    # 延迟导入 get_system，避免与 api.py 的路由注册形成循环导入
    from .api import get_system

    return AdminManager(get_system(), get_settings())


def _handle(error: Exception) -> HTTPException:
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    logger.exception("管理操作失败: %s", error)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="管理操作内部错误"
    )


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data}


# ---------- 请求 Schema ----------
class ConfigUpdateRequest(BaseModel):
    updates: dict[str, Any] = Field(..., description="字段名 -> 新值（白名单字段）")


class RebuildIndexRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="文档目录，默认取配置 DATA_DIR")


class AuditQueryRequest(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    session_id: Optional[str] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    keyword: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)


class KBStatusRequest(BaseModel):
    status: str = Field(..., description="目标状态，如 active/archived/invalid")


class KBDebugRequest(BaseModel):
    question: str
    account: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None
    top_k: Optional[int] = None
    recall_k: Optional[int] = None
    use_rerank: Optional[bool] = None
    chunk_size: Optional[int] = None
    include_invalid: Optional[bool] = None


class SwitchModelRequest(BaseModel):
    provider: str
    model: str
    temperature: Optional[float] = None


class RagRequest(BaseModel):
    enabled: bool


class ParamsRequest(BaseModel):
    params: dict[str, Any] = Field(..., description="参数名 -> 新值（OpsManager 白名单）")


# ---------- 处理函数 ----------
def admin_overview(admin: AdminManager = Depends(get_admin)):
    return _ok(admin.overview())


def admin_features(admin: AdminManager = Depends(get_admin)):
    return _ok({"features": [{"id": f} for f in FEATURE_IDS]})


def admin_feature_status(feature: str, admin: AdminManager = Depends(get_admin)):
    try:
        return _ok(admin.feature_status(feature))
    except ValueError as exc:
        raise _handle(exc)


def admin_feature_stats(
    feature: str,
    top_n: int = Query(default=10, ge=1, le=100),
    keyword: Optional[str] = None,
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.feature_stats(feature, top_n=top_n, keyword=keyword))
    except ValueError as exc:
        raise _handle(exc)


def admin_get_config(feature: str, admin: AdminManager = Depends(get_admin)):
    try:
        return _ok(admin.feature_config(feature))
    except ValueError as exc:
        raise _handle(exc)


def admin_update_config(
    feature: str,
    payload: ConfigUpdateRequest,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.update_config(feature, payload.updates, actor=actor))
    except ValueError as exc:
        raise _handle(exc)


def admin_actions(
    limit: int = Query(default=50, ge=1, le=500),
    actor: Optional[str] = None,
    admin: AdminManager = Depends(get_admin),
):
    return _ok({"actions": admin.action_logs(limit=limit, actor=actor)})


def admin_rebuild_index(
    payload: RebuildIndexRequest = RebuildIndexRequest(),
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.rebuild_index(payload.directory, actor=actor))
    except Exception as exc:
        raise _handle(exc)


def admin_specialty_operations(admin: AdminManager = Depends(get_admin)):
    return _ok({"operations": admin.specialty_operations()})


def admin_specialty_knowledge(admin: AdminManager = Depends(get_admin)):
    return _ok(admin.specialty_knowledge_catalog())


def admin_audit_query(payload: AuditQueryRequest, admin: AdminManager = Depends(get_admin)):
    return _ok({"entries": admin.audit_query(**payload.model_dump(exclude_none=True))})


def admin_audit_stats(admin: AdminManager = Depends(get_admin)):
    return _ok(admin.feature_stats("safety"))


def admin_kb_documents(
    status: Optional[str] = None,
    category: Optional[str] = None,
    admin: AdminManager = Depends(get_admin),
):
    return _ok({"documents": admin.kb_documents(status=status, category=category)})


def admin_kb_set_status(
    doc_id: str,
    payload: KBStatusRequest,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.kb_set_status(doc_id, payload.status, actor=actor))
    except ValueError as exc:
        raise _handle(exc)


def admin_kb_debug(payload: KBDebugRequest, admin: AdminManager = Depends(get_admin)):
    try:
        return _ok(admin.kb_debug(
            payload.question,
            **payload.model_dump(exclude_none=True, exclude={"question"}),
        ))
    except ValueError as exc:
        raise _handle(exc)


def admin_ops_switch_model(
    payload: SwitchModelRequest,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.ops_switch_model(
            payload.provider, payload.model, payload.temperature, actor=actor
        ))
    except ValueError as exc:
        raise _handle(exc)


def admin_ops_set_rag(
    payload: RagRequest,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    return _ok(admin.ops_set_rag(payload.enabled, actor=actor))


def admin_ops_set_params(
    payload: ParamsRequest,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.ops_set_params(payload.params, actor=actor))
    except ValueError as exc:
        raise _handle(exc)


def admin_ops_reset_params(
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    return _ok(admin.ops_reset_params(actor=actor))


def admin_ops_list_sessions(
    include_expired: bool = False, admin: AdminManager = Depends(get_admin)
):
    return _ok({"sessions": admin.system.ops.list_sessions(include_expired)})


def admin_ops_delete_session(
    session_id: str,
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    try:
        return _ok(admin.ops_delete_session(session_id, actor=actor))
    except ValueError as exc:
        raise _handle(exc)


def admin_ops_purge_sensitive(
    actor: str = Depends(require_admin_key),
    admin: AdminManager = Depends(get_admin),
):
    return _ok(admin.ops_purge_sensitive(actor=actor))


# ---------- 路由表（平铺注册） ----------
ROUTES: list[tuple[str, str, Any]] = [
    ("GET", "/admin/overview", admin_overview),
    ("GET", "/admin/features", admin_features),
    ("GET", "/admin/features/{feature}", admin_feature_status),
    ("GET", "/admin/features/{feature}/stats", admin_feature_stats),
    ("GET", "/admin/features/{feature}/config", admin_get_config),
    ("POST", "/admin/features/{feature}/config", admin_update_config),
    ("GET", "/admin/actions", admin_actions),
    ("POST", "/admin/actions/rebuild-index", admin_rebuild_index),
    ("GET", "/admin/specialty/operations", admin_specialty_operations),
    ("GET", "/admin/specialty/knowledge", admin_specialty_knowledge),
    ("POST", "/admin/safety/audit/query", admin_audit_query),
    ("GET", "/admin/safety/audit/stats", admin_audit_stats),
    ("GET", "/admin/kb/documents", admin_kb_documents),
    ("POST", "/admin/kb/documents/{doc_id}/status", admin_kb_set_status),
    ("POST", "/admin/kb/debug", admin_kb_debug),
    ("POST", "/admin/ops/model/switch", admin_ops_switch_model),
    ("POST", "/admin/ops/rag", admin_ops_set_rag),
    ("POST", "/admin/ops/params", admin_ops_set_params),
    ("POST", "/admin/ops/params/reset", admin_ops_reset_params),
    ("GET", "/admin/ops/sessions", admin_ops_list_sessions),
    ("DELETE", "/admin/ops/sessions/{session_id}", admin_ops_delete_session),
    ("POST", "/admin/ops/sessions/purge-sensitive", admin_ops_purge_sensitive),
]


def admin_ui() -> HTMLResponse:
    """返回管理后台单页应用（纯静态 HTML，无构建依赖）。"""
    ui_path = ROOT_DIR / "web" / "admin.html"
    if not ui_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="管理后台页面未找到")
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


def admin_ui_slash() -> RedirectResponse:
    """/admin/ui/ 重定向到 /admin/ui。"""
    return RedirectResponse(url="/admin/ui", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


def register(app: Any) -> None:
    """把 /admin 路由平铺注册到 FastAPI app。"""
    for methods, path, handler in ROUTES:
        app.add_api_route(
            path,
            handler,
            methods=[methods],
            tags=["统一管理后端"],
            name=getattr(handler, "__name__", path),
        )
    # 管理后台网页（公开页面，登录由前端通过 /admin/* API 完成）
    app.add_api_route("/admin/ui", admin_ui, methods=["GET"], include_in_schema=False)
    app.add_api_route("/admin/ui/", admin_ui_slash, methods=["GET"], include_in_schema=False)


__all__ = ["register", "require_admin_key", "get_admin", "ROUTES"]
