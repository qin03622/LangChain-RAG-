"""RAG 链：LCEL 组合 检索 → 构造上下文 → LLM → 后处理。"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.retrievers import BaseRetriever

from .config import Settings
from .prompts import BASIC_QA_PROMPT, PURE_LLM_PROMPT, format_context
from .utils import safety_check


def build_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    settings: Settings,
) -> Runnable:
    """构建基础 RAG 链。

    输出为 dict: {answer, source_documents, safety_warning}
    """

    def retrieve_with_context(question: str) -> dict[str, Any]:
        # (功能五) RAG 关闭时为纯 LLM 模式：跳过检索，注入空上下文与调优提示。
        if not getattr(settings, "rag_enabled", True):
            return {"context": "", "question": question, "docs": []}
        docs: list[Document] = retriever.invoke(question)
        return {"context": format_context(docs), "question": question, "docs": docs}

    rag_enabled = getattr(settings, "rag_enabled", True)
    prompt = BASIC_QA_PROMPT if rag_enabled else PURE_LLM_PROMPT

    # 纯 LLM 提示词额外需要 tuning_hint 变量；RAG 提示词不接受该变量。
    def build_prompt_step():
        if rag_enabled:
            return prompt | llm | StrOutputParser()
        return (
            RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "tuning_hint": _tuning_hint(settings),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    chain = (
        RunnableLambda(retrieve_with_context)
        | {
            "answer": build_prompt_step(),
            "docs": RunnableLambda(lambda x: x["docs"]),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | RunnableLambda(lambda x: _postprocess(x, settings))
    )
    return chain


def _tuning_hint(settings: Settings) -> str:
    """(功能五) 把当前反馈调优与参数摘要拼成提示词提示，供纯 LLM 模式参考。"""
    hint = "请谨慎作答，不确定的内容不要编造。"
    if getattr(settings, "llm_temperature", None) is not None:
        hint += f"（当前温度 {settings.llm_temperature}）"
    return hint


def _postprocess(data: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """后处理：安全护栏 + 免责声明 + 标准化字段。"""
    question = data.get("question", "")
    answer = data.get("answer", "")
    warning = ""
    if settings.enable_safety_guard:
        passed, warning = safety_check(question, answer)
        data["safety_passed"] = passed
    else:
        data["safety_passed"] = True

    data["safety_warning"] = warning
    data["disclaimer"] = settings.disclaimer
    # 统一字段名，便于 API 与展示层使用
    data["source_documents"] = data.get("docs", [])
    return data
