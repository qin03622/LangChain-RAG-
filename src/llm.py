"""LLM 工厂：OpenAI 兼容 API / Ollama 本地模型。"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings
from .utils import get_logger

logger = get_logger(__name__)


class FakeChatModel(BaseChatModel):
    """离线测试用假模型（回显上下文片段）。"""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage

        text = " ".join(str(m.content) for m in messages)
        msg = AIMessage(content=f"[测试模型响应] 已收到输入（{len(text)} 字符）。请配置真实 API Key 后使用。")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def build_llm(settings: Settings) -> BaseChatModel:
    """根据配置构造聊天模型。"""
    provider = settings.llm_provider.lower()

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("请安装 langchain-openai: pip install langchain-openai") from exc
        logger.info("使用 OpenAI 兼容 LLM: %s @ %s", settings.llm_model, settings.llm_base_url)
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key or "EMPTY",
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("请安装 langchain-ollama: pip install langchain-ollama") from exc
        logger.info("使用 Ollama LLM: %s", settings.llm_model)
        return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)

    if provider in {"fake", "test"}:
        logger.warning("使用 FakeChatModel（仅测试用）")
        return FakeChatModel()

    raise ValueError(f"不支持的 LLM 提供方: {provider}")
