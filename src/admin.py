"""统一管理后端（Admin Backend）：五个功能的一站式管理门面。

本模块是"管理面"，不重复实现业务逻辑，而是把现有五个功能的管理能力
统一收敛到一个入口，供 ``src/admin_api.py`` 的 HTTP 路由调用：

  - 功能一 基础 RAG / 临床问答：索引状态、反馈统计、重建索引
  - 功能二 专项辅助：开关与配置、9 个操作清单、结构化知识目录
  - 功能三 安全 / 合规 / 风控：开关与配置、审计统计与查询
  - 功能四 知识库管理：概览、分类、文档状态、检索调试
  - 功能五 后台运维：问答统计、模型、RAG 开关、参数、会话管理

设计原则：
  1. 所有配置项带安全标记，API Key 等敏感字段只回显掩码，绝不读取明文到响应。
  2. 每项管理操作（写操作）记录到操作日志（JSONL，只增不改），便于追责。
  3. 配置分组更新采用白名单，非法字段拒绝；运维参数复用 OpsManager 的校验。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import ROOT_DIR, Settings
from .utils import get_logger

logger = get_logger(__name__)

# 五个功能的固定标识（供前端导航 / API 分组）
FEATURE_IDS = ("rag", "specialty", "safety", "kb", "ops")

FEATURE_LABELS: dict[str, str] = {
    "rag": "功能一 · 基础 RAG 与临床问答",
    "specialty": "功能二 · 面向临床医护的专项辅助",
    "safety": "功能三 · 安全 / 合规 / 风控",
    "kb": "功能四 · 知识库管理后台",
    "ops": "功能五 · 后台运维与统计",
}

# 各功能对外可见的配置项：{配置字段: (类型, 说明, 是否敏感)}
CONFIG_SPECS: dict[str, dict[str, tuple[str, str, bool]]] = {
    "rag": {
        "llm_provider": ("str", "LLM 提供方（openai/ollama/fake）", False),
        "llm_model": ("str", "LLM 模型名", False),
        "llm_api_key": ("str", "LLM API Key", True),
        "llm_base_url": ("str", "LLM Base URL", False),
        "llm_temperature": ("float", "生成温度", False),
        "embedding_provider": ("str", "Embedding 提供方", False),
        "embedding_model": ("str", "Embedding 模型名", False),
        "embedding_api_key": ("str", "Embedding API Key", True),
        "retrieval_strategy": ("str", "检索策略 similarity/mmr/hybrid", False),
        "top_k": ("int", "默认召回数量", False),
        "fetch_k": ("int", "MMR 候选池大小", False),
        "mmr_lambda": ("float", "MMR 多样性权重", False),
        "use_rerank": ("bool", "是否启用重排", False),
        "rerank_model": ("str", "重排模型", False),
        "clinical_mode": ("str", "临床问答默认模式 precise/comprehensive", False),
        "use_hyde": ("bool", "是否启用 HyDE", False),
        "use_multi_query": ("bool", "是否启用多查询改写", False),
        "multi_query_count": ("int", "多查询数量", False),
        "precise_top_k": ("int", "精准模式召回数", False),
        "comprehensive_top_k": ("int", "综合模式召回数", False),
        "guideline_boost": ("float", "指南文档加权系数", False),
        "memory_enabled": ("bool", "多轮会话记忆总开关", False),
        "memory_ttl_seconds": ("int", "会话过期时间（秒）", False),
        "memory_max_turns": ("int", "自动摘要压缩轮数阈值", False),
        "memory_auto_purge_sensitive": ("bool", "自动清理敏感会话", False),
        "feedback_enabled": ("bool", "反馈机制总开关", False),
        "feedback_tuning_enabled": ("bool", "用反馈调优检索/提示词", False),
    },
    "specialty": {
        "specialty_enabled": ("bool", "专项辅助功能总开关", False),
        "specialty_knowledge_strict": ("bool", "结构化知识仅采用带来源条目", False),
        "specialty_top_k": ("int", "专项功能检索文档数", False),
        "specialty_append_disclaimer": ("bool", "强制追加免责声明", False),
        "specialty_warn_diagnosis_assertion": ("bool", "确诊式表述警示", False),
    },
    "safety": {
        "safety_filter_enabled": ("bool", "前置内容过滤/违规拦截", False),
        "privacy_desensitize_enabled": ("bool", "隐私脱敏预处理", False),
        "safety_default_role": ("str", "默认问答角色 patient/clinician", False),
        "audit_enabled": ("bool", "全量审计留痕", False),
        "audit_store_path": ("str", "审计日志路径", False),
        "enable_safety_guard": ("bool", "输出安全护栏", False),
    },
    "kb": {
        "kb_enabled": ("bool", "知识库管理模块总开关", False),
        "kb_registry_path": ("str", "文档登记库路径", False),
        "kb_upload_dir": ("str", "上传暂存目录", False),
        "kb_candidate_buffer": ("int", "过滤前多召回的候选缓冲数", False),
        "kb_default_visibility": ("str", "默认可见性 public/department/confidential", False),
        "kb_recall_k": ("int", "检索面板默认召回数", False),
        "kb_debug_chunk_size": ("int", "面板临时 chunk 大小（0=沿用全局）", False),
        "kb_debug_top_k": ("int", "面板默认 top_k", False),
        "kb_debug_use_rerank": ("bool", "面板默认启用重排", False),
        "kb_filter_invalid_default": ("bool", "检索默认过滤失效文档", False),
    },
    "ops": {
        "ops_enabled": ("bool", "后台运维模块总开关", False),
        "ops_query_log_path": ("str", "问答统计日志路径", False),
        "ops_param_store_path": ("str", "运行时参数持久化路径", False),
        "rag_enabled": ("bool", "RAG 检索增强开关（关闭=纯 LLM）", False),
        "rerank_threshold": ("float", "Rerank 最低得分阈值", False),
    },
}


def _cast(raw: Any, typ: str, field: str) -> Any:
    """按声明类型转换配置值（bool 单独处理，避免 True→1 误转）。"""
    if typ == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            v = raw.strip().lower()
            if v in ("true", "1", "yes", "on"):
                return True
            if v in ("false", "0", "no", "off"):
                return False
        raise ValueError(f"配置项 {field} 需要布尔值，收到 {raw!r}")
    if isinstance(raw, bool):
        raise ValueError(f"配置项 {field} 需要 {typ} 类型，收到布尔值")
    try:
        if typ == "int":
            return int(raw)
        if typ == "float":
            return float(raw)
        return str(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 {field}={raw!r} 无法转换为 {typ}") from exc


def _mask(value: Any) -> str:
    """敏感值掩码：只显示长度与尾号特征，绝不回显明文。"""
    text = "" if value is None else str(value)
    if not text:
        return ""
    if len(text) <= 4:
        return "*" * len(text)
    return f"{'*' * (len(text) - 4)}{text[-4:]}"


class AdminManager:
    """统一管理门面。持有对 ``MedRAGSystem`` 的引用，所有方法线程安全。"""

    def __init__(self, system: Any, settings: Optional[Settings] = None):
        self.system = system
        self.settings: Settings = settings or system.settings
        self._lock = threading.RLock()
        log_path = Path(self.settings.admin_action_log_path)
        if not log_path.is_absolute():
            log_path = ROOT_DIR / log_path
        self._action_log_path: Path = log_path
        self._action_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ==================== 操作审计 ====================
    def _record_action(
        self, actor: str, action: str, feature: str = "", detail: Any = None
    ) -> dict[str, Any]:
        rec = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "actor": actor or "anonymous",
            "action": action,
            "feature": feature,
            "detail": detail,
        }
        with self._lock:
            with self._action_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    def action_logs(self, limit: int = 50, actor: Optional[str] = None) -> list[dict[str, Any]]:
        """查询管理操作日志（最新的 limit 条）。"""
        if not self._action_log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self._lock:
            for line in self._action_log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if actor and rec.get("actor") != actor:
                    continue
                rows.append(rec)
        return rows[-limit:]

    # ==================== 总览 ====================
    def overview(self) -> dict[str, Any]:
        """管理后台总览：系统状态 + 五个功能的状态摘要。"""
        s = self.settings
        return {
            "app": "MedRAG 统一管理后端",
            "version": "1.0.0",
            "server_time": time.time(),
            "system": {
                "llm_provider": s.llm_provider,
                "llm_model": s.llm_model,
                "embedding_provider": s.embedding_provider,
                "embedding_model": s.embedding_model,
                "retrieval_strategy": s.retrieval_strategy,
                "vector_store": s.vector_store,
                "index_ready": self.system.vsm.exists(),
            },
            "features": {
                "rag": self.feature_status("rag"),
                "specialty": self.feature_status("specialty"),
                "safety": self.feature_status("safety"),
                "kb": self.feature_status("kb"),
                "ops": self.feature_status("ops"),
            },
            "features_catalog": [
                {"id": fid, "label": FEATURE_LABELS[fid]} for fid in FEATURE_IDS
            ],
        }

    # ==================== 功能状态 / 配置 / 统计 ====================
    def feature_status(self, feature: str) -> dict[str, Any]:
        """单个功能的开关与关键运行态（只读，不含统计聚合）。"""
        self._validate_feature(feature)
        s = self.settings
        if feature == "rag":
            return {
                "enabled": True,
                "index_ready": self.system.vsm.exists(),
                "clinical_mode": s.clinical_mode,
                "rag_enabled": getattr(s, "rag_enabled", True),
                "memory_enabled": s.memory_enabled,
                "feedback_enabled": s.feedback_enabled,
            }
        if feature == "specialty":
            return {
                "enabled": s.specialty_enabled,
                "operations": len(self.specialty_operations()),
                "strict": s.specialty_knowledge_strict,
                "append_disclaimer": s.specialty_append_disclaimer,
            }
        if feature == "safety":
            return {
                "enabled": s.safety_filter_enabled or s.privacy_desensitize_enabled,
                "filter_enabled": s.safety_filter_enabled,
                "privacy_enabled": s.privacy_desensitize_enabled,
                "audit_enabled": s.audit_enabled,
                "default_role": s.safety_default_role,
            }
        if feature == "kb":
            ov = self.system.kb.overview()
            return {
                "enabled": s.kb_enabled,
                "total_documents": ov.get("total_documents", 0),
                "active_documents": ov.get("active_documents", 0),
                "vector_index_ready": ov.get("vector_index_ready", False),
            }
        # ops
        return {
            "enabled": s.ops_enabled,
            "rag_enabled": self.system.ops.current_model().get("rag_enabled", True),
            "current_model": self.system.ops.current_model(),
            "params_count": len(self.system.ops.get_params()),
        }

    def feature_stats(self, feature: str, **kwargs: Any) -> dict[str, Any]:
        """单个功能的统计摘要（只读聚合）。"""
        self._validate_feature(feature)
        if feature == "rag":
            return {
                "feedback": self.system.feedback_stats(),
                "index_ready": self.system.vsm.exists(),
            }
        if feature == "specialty":
            return {"operations": self.specialty_operations()}
        if feature == "safety":
            return self.system.audit_stats()
        if feature == "kb":
            return self.system.kb.overview()
        return self.system.ops.stats(
            top_n=int(kwargs.get("top_n", 10)), keyword=kwargs.get("keyword")
        )

    def feature_config(self, feature: str) -> dict[str, Any]:
        """返回某功能的全部可见配置项（敏感值掩码）。"""
        self._validate_feature(feature)
        result: dict[str, Any] = {}
        for name, (typ, desc, sensitive) in CONFIG_SPECS[feature].items():
            raw = getattr(self.settings, name, None)
            result[name] = {
                "value": _mask(raw) if sensitive else self._json_value(raw),
                "type": typ,
                "description": desc,
                "sensitive": sensitive,
            }
        return {"feature": feature, "label": FEATURE_LABELS[feature], "config": result}

    def update_config(
        self, feature: str, updates: dict[str, Any], actor: str = "anonymous"
    ) -> dict[str, Any]:
        """按功能分组更新配置（白名单字段，敏感字段也只接受新值不回显）。"""
        self._validate_feature(feature)
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates 必须是非空对象")
        spec = CONFIG_SPECS[feature]
        unknown = set(updates) - set(spec)
        if unknown:
            raise ValueError(f"功能 {feature} 不支持的配置项: {', '.join(sorted(unknown))}")
        applied: dict[str, Any] = {}
        with self._lock:
            for name, raw in updates.items():
                typ, _, sensitive = spec[name]
                value = _cast(raw, typ, name)
                setattr(self.settings, name, value)
                applied[name] = _mask(value) if sensitive else value
            # 与 OpsManager 的运行时参数联动
            if feature == "rag" and "rag_enabled" in updates:
                self.system.ops.set_rag(bool(self.settings.rag_enabled))
            if feature == "ops" and "rerank_threshold" in updates:
                # rerank_threshold 属于 ops 配置但非 OpsManager 白名单参数，直接写 settings 即可
                pass
            self._record_action(
                actor, "update_config", feature, {"fields": list(updates.keys())}
            )
        return {"feature": feature, "applied": applied}

    # ==================== 功能一：RAG / 临床问答 ====================
    def rebuild_index(self, directory: Optional[str] = None, actor: str = "anonymous") -> dict[str, Any]:
        """重建（或增量更新）向量索引，管理端维护入口。"""
        with self._lock:
            count = self.system.ingest(directory, rebuild=True)
            self._record_action(actor, "rebuild_index", "rag", {"directory": directory, "chunks": count})
        return {"chunks": count, "index_ready": self.system.vsm.exists()}

    # ==================== 功能二：专项辅助 ====================
    def specialty_operations(self) -> list[dict[str, Any]]:
        """列出功能二的 9 个操作及参数（复用 API 层常量，避免循环导入）。"""
        from .api import SPECIALTY_OPERATIONS

        return [
            {"operation": op, "params": params} for op, params in SPECIALTY_OPERATIONS.items()
        ]

    def specialty_knowledge_catalog(self) -> dict[str, Any]:
        """结构化知识目录概览：各表条目数、来源/版本样本（教学数据说明）。"""
        from . import specialty_knowledge as sk

        catalog: dict[str, Any] = {}
        for name in dir(sk):
            if not name.isupper() or name.startswith("_"):
                continue
            value = getattr(sk, name)
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
                sample = value[0]
                catalog[name] = {
                    "count": len(value),
                    "sample_source": sample.get("source", ""),
                    "sample_version": sample.get("version", ""),
                }
        return {
            "tables": catalog,
            "note": "教学示例数据；生产必须替换为审核过的权威知识库。",
        }

    # ==================== 功能三：安全 / 合规 ====================
    def audit_query(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.system.audit_entries(**kwargs)

    # ==================== 功能四：知识库 ====================
    def kb_documents(
        self, status: Optional[str] = None, category: Optional[str] = None
    ) -> list[dict[str, Any]]:
        docs = list(self.system.kb.documents.values())
        if status:
            docs = [d for d in docs if d.get("status") == status]
        if category:
            docs = [d for d in docs if d.get("category") == category]
        return docs

    def kb_set_status(
        self, doc_id: str, status: str, actor: str = "anonymous"
    ) -> dict[str, Any]:
        with self._lock:
            result = self.system.kb.set_status(doc_id, status)
            self._record_action(actor, "kb_set_status", "kb", {"doc_id": doc_id, "status": status})
        return result

    def kb_debug(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.system.kb.debug_retrieval(question, **kwargs)

    # ==================== 功能五：运维 ====================
    def ops_switch_model(self, provider: str, model: str, temperature: Optional[float] = None, actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.switch_model(provider, model, temperature=temperature)
            self._record_action(actor, "switch_model", "ops", result)
        return result

    def ops_set_rag(self, enabled: bool, actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.set_rag(enabled)
            self._record_action(actor, "set_rag", "ops", result)
        return result

    def ops_set_params(self, params: dict[str, Any], actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.set_params(**params)
            self._record_action(actor, "set_params", "ops", {"fields": list(params)})
        return result

    def ops_reset_params(self, actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.reset_params()
            self._record_action(actor, "reset_params", "ops")
        return result

    def ops_delete_session(self, session_id: str, actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.delete_session(session_id)
            self._record_action(actor, "delete_session", "ops", {"session_id": session_id})
        return result

    def ops_purge_sensitive(self, actor: str = "anonymous") -> dict[str, Any]:
        with self._lock:
            result = self.system.ops.purge_sensitive_sessions()
            self._record_action(actor, "purge_sensitive", "ops", result)
        return result

    # ==================== 工具 ====================
    @staticmethod
    def _validate_feature(feature: str) -> None:
        if feature not in FEATURE_IDS:
            raise ValueError(f"未知功能: {feature}；可选: {', '.join(FEATURE_IDS)}")

    @staticmethod
    def _json_value(value: Any) -> Any:
        """把配置原始值转为 JSON 可序列化值。"""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


__all__ = ["AdminManager", "FEATURE_IDS", "FEATURE_LABELS", "CONFIG_SPECS"]
