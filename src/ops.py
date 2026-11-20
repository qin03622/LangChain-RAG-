"""功能五：后台运维 & 统计。

提供四类能力（全部经 ``MedRAGSystem.ops`` 暴露）：

1. 问答统计：高频查询病种 / 高频问题 / 用户反馈（幻觉标记数量）。
2. 模型切换：在线模型 / 本地医疗大模型 / 测试模型之间热切换；
   RAG 开关（纯 LLM 模式 vs RAG 检索增强模式）用于 A/B 对比测试。
3. 参数可视化配置：chunk 大小 / 重叠量 / 召回 top-k / rerank 阈值 /
   HyDE 开关 / temperature，运行时可调并持久化。
4. 会话管理：查看历史对话列表、手动删除敏感会话。

所有统计源自只增不改的 JSONL 问答日志（``ops/query_log.jsonl``）；
运行时参数持久化到 ``ops/runtime_params.json``，重启后自动恢复。
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import Settings
from .llm import build_llm
from .utils import get_logger

if TYPE_CHECKING:  # 仅类型检查时导入，避免循环依赖
    from .rag_system import MedRAGSystem

logger = get_logger(__name__)

# 常见医疗病种关键词（用于高频病种统计，命中即归入该类；可扩充）。
DISEASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "高血压": ("高血压", "血压高", "降压", "收缩压", "舒张压"),
    "糖尿病": ("糖尿病", "血糖", "二甲双胍", "胰岛素", "糖化血红蛋白"),
    "冠心病": ("冠心病", "心绞痛", "心肌梗死", "支架", "冠脉"),
    "脑卒中": ("脑卒中", "中风", "脑梗", "脑出血", "偏瘫"),
    "哮喘": ("哮喘", "支气管哮喘", "喘息"),
    "慢阻肺": ("慢阻肺", "copd", "肺气肿"),
    "慢性肾病": ("慢性肾病", "肌酐", "肾功能不全", "尿毒症"),
    "肿瘤": ("肿瘤", "癌症", "癌", "化疗", "放疗"),
    "感染": ("感染", "发热", "炎症", "抗生素"),
    "过敏": ("过敏", "皮疹", "荨麻疹"),
}

# 可调参数白名单：{名称: (设置属性, 类型, 说明)}
PARAM_SPEC: dict[str, tuple[str, Callable[[Any], Any], str]] = {
    "chunk_size": ("chunk_size", int, "文档切分块大小（字符）"),
    "chunk_overlap": ("chunk_overlap", int, "切分块重叠量（字符）"),
    "top_k": ("top_k", int, "召回数量 top-k"),
    "rerank_threshold": ("rerank_threshold", float, "Rerank 最低得分阈值（0-1，归一化后）"),
    "use_hyde": ("use_hyde", bool, "是否启用 HyDE 查询改写"),
    "temperature": ("llm_temperature", float, "LLM 生成温度"),
}


class OpsManager:
    """后台运维管理器。所有方法线程安全，可被 FastAPI 并发调用。"""

    def __init__(self, system: "MedRAGSystem"):
        self.system = system
        self.settings: Settings = system.settings
        self._lock = threading.RLock()
        self._log_path: Path = self.settings.ops_query_log_file
        self._param_path: Path = self.settings.ops_param_store_file
        # 运行时状态（可持久化参数）
        default_model = {
            "provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
        }
        self._current_model: dict = dict(default_model)
        self._rag_enabled: bool = bool(getattr(self.settings, "rag_enabled", True))
        # 记录初始值（在恢复持久化参数之前），供 reset_params 恢复
        self._defaults: dict[str, Any] = {
            name: getattr(self.settings, spec[0]) for name, spec in PARAM_SPEC.items()
        }
        self._default_rag_enabled: bool = self._rag_enabled
        self._default_model: dict = dict(default_model)
        self._params: dict[str, Any] = {}
        self._load_params()
        # 确保日志/参数目录存在
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._param_path.parent.mkdir(parents=True, exist_ok=True)
        # 重启后恢复上次持久化的运行时配置（模型不可用时安全降级）
        try:
            self._apply_params_to_settings()
            self.settings.rag_enabled = self._rag_enabled
            if self._current_model != default_model or "temperature" in self._params:
                temp = self.settings.model_copy(
                    update={
                        "llm_provider": self._current_model["provider"],
                        "llm_model": self._current_model["model"],
                    }
                )
                self.system.rebind_llm(build_llm(temp))
        except Exception as exc:  # noqa: BLE001
            logger.warning("恢复运维运行时配置失败，使用默认值: %s", exc)

    # ==================== 内部工具 ====================
    def _load_params(self) -> None:
        if not self._param_path.exists():
            return
        try:
            data = json.loads(self._param_path.read_text(encoding="utf-8"))
            # 参数
            for name, raw in (data.get("params") or {}).items():
                if name in PARAM_SPEC:
                    _, cast, _ = PARAM_SPEC[name]
                    self._params[name] = cast(raw)
            # 模型 & RAG 开关
            cur = data.get("current_model") or {}
            if cur.get("provider") and cur.get("model"):
                self._current_model = {
                    "provider": str(cur["provider"]),
                    "model": str(cur["model"]),
                }
            if "rag_enabled" in data:
                self._rag_enabled = bool(data["rag_enabled"])
            logger.info("已载入运维运行时参数: %s", self._param_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("载入运维参数失败(%s)，使用默认值。", exc)

    def _persist(self) -> None:
        data = {
            "params": self._params,
            "current_model": self._current_model,
            "rag_enabled": self._rag_enabled,
            "updated_at": time.time(),
        }
        self._param_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _apply_params_to_settings(self) -> None:
        """把持久化的参数写回当前 Settings 对象（供检索/切分/模型使用）。"""
        for name, value in self._params.items():
            attr = PARAM_SPEC[name][0]
            setattr(self.settings, attr, value)
        # chunk 参数需要重建切分器
        if "chunk_size" in self._params or "chunk_overlap" in self._params:
            self.system.processor.reconfigure(
                chunk_size=getattr(self.settings, "chunk_size"),
                chunk_overlap=getattr(self.settings, "chunk_overlap"),
            )
        # rerank 阈值由 ClinicalRetriever 读取 settings.rerank_threshold，已同步。

    def _log_query(
        self,
        question: str,
        answer: str,
        *,
        mode: str,
        rag_enabled: bool,
        model: dict,
        session_id: Optional[str] = None,
        chunk_ids: Optional[list[str]] = None,
    ) -> None:
        if not self.settings.ops_enabled:
            return
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "question": question,
            "answer": answer[:500],
            "mode": mode,
            "rag_enabled": bool(rag_enabled),
            "model": model,
            "session_id": session_id or "",
            "chunk_ids": chunk_ids or [],
        }
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_log(self) -> list[dict]:
        if not self._log_path.exists():
            return []
        records: list[dict] = []
        with self._lock:
            for line in self._log_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # ==================== 1. 问答统计 ====================
    def stats(self, top_n: int = 10, keyword: str | None = None) -> dict[str, Any]:
        """问答统计总览：高频问题、高频病种、反馈/幻觉数量、RAG 使用情况。"""
        records = self._read_log()
        if keyword:
            kw = keyword.strip().lower()
            records = [r for r in records if kw in r.get("question", "").lower()]

        # 高频问题（规范化：去空白，按原文计数）
        q_counter = Counter(
            re.sub(r"\s+", "", r.get("question", "")).strip()
            for r in records
            if r.get("question", "").strip()
        )
        # 高频病种
        d_counter: Counter[str] = Counter()
        for r in records:
            q = r.get("question", "").lower()
            for disease, keywords in DISEASE_KEYWORDS.items():
                if any(k.lower() in q for k in keywords):
                    d_counter[disease] += 1

        # 反馈统计（来自 FeedbackStore）
        fb_stats: dict[str, Any] = {}
        try:
            fb_stats = self.system.feedback.stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取反馈统计失败: %s", exc)

        total = len(records)
        rag_count = sum(1 for r in records if r.get("rag_enabled"))
        by_label = fb_stats.get("by_label", {}) or {}
        return {
            "total_queries": total,
            "rag_queries": rag_count,
            "pure_llm_queries": total - rag_count,
            "top_questions": [
                {"question": q, "count": c} for q, c in q_counter.most_common(top_n)
            ],
            "top_diseases": [
                {"disease": d, "count": c} for d, c in d_counter.most_common(top_n)
            ],
            "feedback": {
                "total": fb_stats.get("total", 0),
                "by_label": by_label,
                "hallucination_count": int(by_label.get("hallucination", 0)),
                "negative_rate": fb_stats.get("negative_rate", 0.0),
            },
        }

    # ==================== 2. 模型切换 & RAG 开关 ====================
    def list_models(self) -> list[dict]:
        """列出预设可切换模型（在线 / 本地医疗大模型 / 测试）。"""
        return [
            {
                "key": "deepseek",
                "provider": "openai",
                "model": "deepseek-chat",
                "label": "在线模型 · DeepSeek",
                "description": "OpenAI 兼容在线服务，需配置 API Key 与 BASE_URL",
            },
            {
                "key": "online-default",
                "provider": "openai",
                "model": self.settings.llm_model,
                "label": "在线模型 · 当前配置",
                "description": "使用 .env 中现有的 LLM_PROVIDER / LLM_MODEL 配置",
            },
            {
                "key": "local-ollama",
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "label": "本地医疗大模型 · Ollama/qwen2.5",
                "description": "本地 Ollama 服务，无需云 API Key（需安装 langchain-ollama）",
            },
            {
                "key": "fake",
                "provider": "fake",
                "model": "fake",
                "label": "测试模型 · Fake",
                "description": "离线确定性假模型，用于跑通流程与 A/B 测试",
            },
        ]

    def switch_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        temperature: float | None = None,
    ) -> dict:
        """热切换 LLM：重建模型实例并让所有下游服务重新绑定。

        provider/model 二选一；缺省时沿用当前配置。切换后重建链与临床服务，
        但不重建向量索引（embedding 与 LLM 解耦）。
        """
        if not provider and not model and temperature is None:
            raise ValueError("请至少提供 provider / model / temperature 之一")

        new_provider = (provider or self._current_model["provider"]).lower()
        new_model = model or self._current_model["model"]

        # 用 model_copy 生成临时 Settings 构造新 LLM，避免污染其他无关配置
        update: dict[str, Any] = {"llm_provider": new_provider, "llm_model": new_model}
        if temperature is not None:
            update["llm_temperature"] = float(temperature)
        temp_settings = self.settings.model_copy(update=update)
        try:
            new_llm = build_llm(temp_settings)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"切换模型失败：{exc}") from exc

        with self._lock:
            # 同步到真实 settings（temperature 也写回，供可视化配置展示）
            self.settings.llm_provider = new_provider
            self.settings.llm_model = new_model
            if temperature is not None:
                self.settings.llm_temperature = float(temperature)
                self._params["temperature"] = float(temperature)
            self._current_model = {"provider": new_provider, "model": new_model}

            # 重新绑定到所有持有 llm 引用的组件（含 LCEL 链）
            self.system.rebind_llm(new_llm)
            self._persist()
        logger.info("模型已切换: %s / %s", new_provider, new_model)
        return self.current_model()

    def current_model(self) -> dict:
        return {
            **self._current_model,
            "rag_enabled": self._rag_enabled,
            "temperature": getattr(self.settings, "llm_temperature", None),
        }

    def set_rag(self, enabled: bool) -> dict:
        """开关 RAG：关闭后系统走纯 LLM 模式（不注入检索证据）。"""
        with self._lock:
            self._rag_enabled = bool(enabled)
            self.settings.rag_enabled = bool(enabled)
            self.system._invalidate_chain()
            self._persist()
        logger.info("RAG 已%s", "开启" if enabled else "关闭（纯 LLM 模式）")
        return {"rag_enabled": self._rag_enabled}

    # ==================== 3. 参数可视化配置 ====================
    def get_params(self) -> dict[str, Any]:
        """返回当前生效的可调参数及其元数据（用于前端可视化渲染）。"""
        result: dict[str, Any] = {}
        for name, (attr, cast, desc) in PARAM_SPEC.items():
            raw = self._params.get(name, getattr(self.settings, attr))
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                value = raw
            # bool 转 JSON 友好值
            result[name] = {
                "value": value,
                "type": cast.__name__,
                "description": desc,
                "default": cast(getattr(self.settings, attr)),
            }
        result["rag_enabled"] = {
            "value": self._rag_enabled,
            "type": "bool",
            "description": "RAG 检索增强开关（关闭=纯 LLM 模式）",
            "default": True,
        }
        return result

    def set_params(self, **params: Any) -> dict:
        """运行时设置参数（仅白名单内字段），校验类型后写回并持久化。

        支持字段：chunk_size / chunk_overlap / top_k / rerank_threshold /
        use_hyde / temperature。未知字段会被拒绝。
        """
        unknown = set(params) - set(PARAM_SPEC)
        if unknown:
            raise ValueError(f"不支持的参数: {', '.join(sorted(unknown))}")
        applied: dict[str, Any] = {}
        with self._lock:
            for name, raw in params.items():
                attr, cast, _ = PARAM_SPEC[name]
                if isinstance(raw, bool) and cast is not bool:
                    # bool 是 int 子类，避免把 True 误当 1
                    raise ValueError(f"参数 {name} 类型不正确，期望 {cast.__name__}")
                try:
                    value = cast(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"参数 {name}={raw!r} 无法转换为 {cast.__name__}") from exc
                self._params[name] = value
                setattr(self.settings, attr, value)
                applied[name] = value

            if "chunk_size" in applied or "chunk_overlap" in applied:
                self.system.processor.reconfigure(
                    chunk_size=getattr(self.settings, "chunk_size"),
                    chunk_overlap=getattr(self.settings, "chunk_overlap"),
                )
            if "temperature" in applied:
                # temperature 需重建 LLM 实例才会生效
                temp = self.settings.model_copy(
                    update={
                        "llm_provider": self._current_model["provider"],
                        "llm_model": self._current_model["model"],
                        "llm_temperature": applied["temperature"],
                    }
                )
                self.system.rebind_llm(build_llm(temp))
            # top_k / use_hyde / RAG 开关影响检索器与链，需使其重建
            self.system._invalidate_chain()
            self._persist()
        logger.info("运维参数已更新: %s", applied)
        return applied

    def reset_params(self) -> dict:
        """清空所有运行时参数覆盖，恢复初始 Settings 默认值。"""
        with self._lock:
            self._params.clear()
            for name, value in self._defaults.items():
                attr = PARAM_SPEC[name][0]
                setattr(self.settings, attr, value)
            self._rag_enabled = self._default_rag_enabled
            self.settings.rag_enabled = self._default_rag_enabled
            self._current_model = dict(self._default_model)
            self.settings.llm_provider = self._default_model["provider"]
            self.settings.llm_model = self._default_model["model"]
            self.system.processor.reconfigure(
                chunk_size=getattr(self.settings, "chunk_size"),
                chunk_overlap=getattr(self.settings, "chunk_overlap"),
            )
            temp = self.settings.model_copy(
                update={
                    "llm_provider": self._current_model["provider"],
                    "llm_model": self._current_model["model"],
                }
            )
            from .llm import build_llm as _build_llm

            self.system.rebind_llm(_build_llm(temp))
            self._persist()
        self.system._invalidate_chain()
        return self.get_params()

    def compare(
        self,
        question: str,
        *,
        mode: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """A/B 对比测试：对同一问题分别以【纯 LLM】和【RAG】模式作答。

        两次调用都记录进问答日志（rag_enabled 分别标记），便于统计对比。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空。")
        original = self._rag_enabled
        try:
            self.set_rag(False)
            llm_result = self.system.query(question, mode=mode, session_id=session_id)
            llm_result = dict(llm_result)
            self.set_rag(True)
            rag_result = self.system.query(question, mode=mode, session_id=session_id)
            rag_result = dict(rag_result)
        finally:
            self.set_rag(original)
        return {
            "question": question,
            "pure_llm": {
                "answer": llm_result.get("answer", ""),
                "source_count": len(llm_result.get("source_documents", [])),
            },
            "rag": {
                "answer": rag_result.get("answer", ""),
                "source_count": len(rag_result.get("source_documents", [])),
            },
        }

    # ==================== 4. 会话管理 ====================
    def list_sessions(self, include_expired: bool = False) -> list[dict]:
        """查看历史会话列表（不返回完整消息正文，避免敏感内容裸露）。"""
        return self.system.memory.list_sessions(include_expired=include_expired)

    def get_session(self, session_id: str) -> dict:
        """查看单个会话的完整对话记录（运维审计用）。"""
        s = self.system.memory.get(session_id)
        if s is None:
            raise KeyError(f"会话不存在或已过期: {session_id}")
        return s.as_dict()

    def delete_session(self, session_id: str) -> dict:
        """手动删除会话（敏感会话立即丢弃，同步删除磁盘持久化）。"""
        s = self.system.memory.get(session_id)
        if s is None:
            raise KeyError(f"会话不存在或已过期: {session_id}")
        was_sensitive = bool(s.sensitive)
        self.system.memory.clear(session_id)
        logger.info("会话已删除: %s (敏感=%s)", session_id, was_sensitive)
        return {"session_id": session_id, "deleted": True, "sensitive": was_sensitive}

    def purge_sensitive_sessions(self) -> dict:
        """批量清理所有被标记为敏感的会话。"""
        sids = [s["session_id"] for s in self.list_sessions() if s.get("sensitive")]
        for sid in sids:
            self.system.memory.clear(sid)
        return {"purged": len(sids), "session_ids": sids}


__all__ = ["OpsManager", "DISEASE_KEYWORDS", "PARAM_SPEC"]
