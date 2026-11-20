"""配置管理：YAML 默认值 + 环境变量覆盖。

所有配置可通过项目根目录的 .env 文件或进程环境变量覆盖，
环境变量优先级高于 config/config.yaml。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（src/ 的上一级）
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """系统全局配置。"""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_provider: str = Field(default="openai", description="LLM 提供方: openai | ollama")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_api_key: str = Field(default="")
    llm_base_url: Optional[str] = Field(default=None)
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=2048)

    # Embedding
    embedding_provider: str = Field(default="openai", description="openai | ollama | huggingface")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_api_key: str = Field(default="")
    embedding_base_url: Optional[str] = Field(default=None)

    # 向量库
    vector_store: str = Field(default="faiss", description="faiss | chroma")
    vector_store_dir: str = Field(default="./vectorstore")
    collection_name: str = Field(default="medical_rag")

    # 检索
    retrieval_strategy: str = Field(default="mmr", description="similarity | mmr | hybrid")
    top_k: int = Field(default=5)
    fetch_k: int = Field(default=20)
    mmr_lambda: float = Field(default=0.5)
    score_threshold: float = Field(default=0.0)
    use_rerank: bool = Field(default=False)

    # 文本切分
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)

    # ===== 临床问答增强 =====
    # 检索增强：HyDE / MultiQuery / Rerank
    clinical_mode: str = Field(default="precise", description="precise 精准检索 | comprehensive 综合推理")
    use_hyde: bool = Field(default=False, description="HyDE：先生成假设答案再检索")
    use_multi_query: bool = Field(default=False, description="多查询生成后融合检索")
    multi_query_count: int = Field(default=3)
    use_rerank: bool = Field(default=True, description="CrossEncoder / LLM 重排")
    rerank_model: str = Field(default="BAAI/bge-reranker-base")
    comprehensive_top_k: int = Field(default=10, description="综合推理模式检索文档数")
    precise_top_k: int = Field(default=4, description="精准检索模式检索文档数")
    guideline_boost: float = Field(default=1.3, description="指南/规范类文档权重")

    # 会话记忆
    memory_enabled: bool = Field(default=True)
    memory_ttl_seconds: int = Field(default=3600, description="会话过期时间（秒）")
    memory_max_turns: int = Field(default=12, description="超过轮数后自动摘要压缩")
    memory_auto_purge_sensitive: bool = Field(default=True, description="敏感会话自动清理")
    session_store_dir: str = Field(default="./sessions")

    # 反馈
    feedback_enabled: bool = Field(default=True)
    feedback_store_path: str = Field(default="./feedback.jsonl")
    feedback_tuning_enabled: bool = Field(default=True, description="用反馈调节检索与提示词")

    # ===== 面向临床医护的专项辅助功能（功能二） =====
    specialty_enabled: bool = Field(default=True, description="是否启用专项辅助功能")
    specialty_knowledge_strict: bool = Field(
        default=True, description="结构化知识仅采用已标注来源的条目，避免无来源推断"
    )
    specialty_top_k: int = Field(default=6, description="专项功能检索文档数")
    specialty_append_disclaimer: bool = Field(
        default=True, description="无论模型输出如何，强制追加专项免责声明"
    )
    specialty_warn_diagnosis_assertion: bool = Field(
        default=True, description="检测并警示“确诊式”表述，防止越权下结论"
    )

    # ===== 安全 / 合规 / 风控（功能三） =====
    safety_filter_enabled: bool = Field(default=True, description="前置内容过滤与违规拦截")
    privacy_desensitize_enabled: bool = Field(default=True, description="隐私脱敏预处理")
    safety_default_role: str = Field(default="patient", description="默认问答角色: patient | clinician")
    audit_enabled: bool = Field(default=True, description="全量审计留痕")
    audit_store_path: str = Field(default="./audit.jsonl", description="审计日志 JSONL 路径")

    # ===== 功能四：知识库管理（LangChain RAG 配套管理后台） =====
    kb_enabled: bool = Field(default=True, description="是否启用知识库管理后台")
    kb_registry_path: str = Field(default="./knowledge_base/registry.json", description="文档登记库(JSON)")
    kb_upload_dir: str = Field(default="./knowledge_base/uploads", description="上传文件暂存目录")
    kb_candidate_buffer: int = Field(
        default=15, description="检索时多召回的候选数，供失效/权限过滤后裁剪"
    )
    kb_default_visibility: str = Field(default="public", description="public | department | confidential")
    kb_categories: str = Field(
        default="临床指南,药学,检验,内科,外科,急诊,儿科,护理,感染控制",
        description="预置知识库分类目录（逗号分隔）",
    )
    kb_recall_k: int = Field(default=20, description="检索测试面板默认召回候选数")
    kb_debug_chunk_size: int = Field(default=0, description="测试面板临时 chunk_size，0 表示沿用全局配置")
    kb_debug_top_k: int = Field(default=5, description="检索测试面板默认 top_k")
    kb_debug_use_rerank: bool = Field(default=True, description="检索测试面板默认是否重排")
    kb_filter_invalid_default: bool = Field(
        default=True, description="通用检索默认过滤已归档/失效文档，防止过时知识被引用"
    )

    # ===== 功能五：后台运维 & 统计 =====
    ops_enabled: bool = Field(default=True, description="是否启用后台运维与统计功能")
    ops_query_log_path: str = Field(
        default="./ops/query_log.jsonl", description="问答统计日志 JSONL 路径"
    )
    ops_param_store_path: str = Field(
        default="./ops/runtime_params.json", description="运行时可调参数持久化路径"
    )
    rag_enabled: bool = Field(default=True, description="RAG 检索增强开关；关闭=纯 LLM 模式")
    rerank_threshold: float = Field(
        default=0.0, description="Rerank 最低得分阈值；低于阈值的候选被剔除（0=不过滤）"
    )

    # ===== 统一管理后端（五个功能的管理 API） =====
    admin_api_key: str = Field(
        default="", description="管理 API 访问密钥；为空时拒绝所有 /admin 请求"
    )
    admin_action_log_path: str = Field(
        default="./admin/action_log.jsonl", description="管理操作审计日志 JSONL 路径"
    )

    # 路径 / 安全
    data_dir: str = Field(default="./data/raw")
    log_level: str = Field(default="INFO")
    enable_safety_guard: bool = Field(default=True)
    disclaimer: str = Field(
        default=(
            "以上内容由 AI 辅助生成，仅供参考，不能替代执业医师的诊断与处方。"
            "请务必咨询专业医疗机构。"
        )
    )

    @property
    def vector_store_path(self) -> Path:
        p = Path(self.vector_store_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def session_store_path(self) -> Path:
        p = Path(self.session_store_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def feedback_path(self) -> Path:
        p = Path(self.feedback_store_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def kb_registry_file(self) -> Path:
        p = Path(self.kb_registry_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def kb_upload_path(self) -> Path:
        p = Path(self.kb_upload_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def ops_query_log_file(self) -> Path:
        p = Path(self.ops_query_log_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def ops_param_store_file(self) -> Path:
        p = Path(self.ops_param_store_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def admin_action_log_file(self) -> Path:
        p = Path(self.admin_action_log_path)
        return p if p.is_absolute() else ROOT_DIR / p


def load_yaml_config(path: Optional[Path] = None) -> dict[str, Any]:
    """读取 YAML 配置文件（仅作参考/默认值来源）。"""
    cfg_path = path or (ROOT_DIR / "config" / "config.yaml")
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_settings() -> Settings:
    """构造配置对象。"""
    return Settings()


# 全局单例
settings = get_settings()
