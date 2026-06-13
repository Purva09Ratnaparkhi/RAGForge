"""
LLM Factory — Configurable LLM provider with automatic fallback.
=================================================================

Returns a LangChain-compatible LLM object configured for either
**Groq** (primary — fast inference of open-source models) or
**OpenAI** (fallback — higher reliability and model variety).

Why Groq primary?
-----------------
Groq's LPU inference engine provides the fastest token generation for
Llama 3 70B, making it ideal for real-time RAG where users expect
sub-second streaming.  OpenAI serves as a fallback if Groq is down
or rate-limited.

Connects to
-----------
* Every component that needs an LLM: query rewriter, compressor,
  multi-vector summariser, guardrails, agentic RAG, etc.
* ``config.settings`` — provider, model, temperature, max_tokens, keys.
"""

from __future__ import annotations

from loguru import logger

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def get_llm(
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
):
    """Create and return a LangChain LLM object.

    Parameters
    ----------
    provider : str | None
        ``"groq"`` or ``"openai"``.  Defaults to ``LLM_PROVIDER`` from
        settings.
    temperature : float | None
        Sampling temperature.  Defaults to ``LLM_TEMPERATURE``.
    max_tokens : int | None
        Maximum tokens to generate.  Defaults to ``LLM_MAX_TOKENS``.
    streaming : bool
        Enable streaming mode for SSE endpoints.

    Returns
    -------
    BaseChatModel
        A LangChain chat model (``ChatGroq`` or ``ChatOpenAI``).

    Raises
    ------
    ValueError
        If the requested provider's API key is missing.

    Example
    -------
    >>> llm = get_llm()
    >>> response = llm.invoke("Hello")
    """
    provider = provider or LLM_PROVIDER
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or LLM_MAX_TOKENS

    if provider == "groq":
        return _get_groq(temperature, max_tokens, streaming)
    elif provider == "openai":
        return _get_openai(temperature, max_tokens, streaming)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_with_fallback(
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
):
    """Get primary LLM with automatic fallback to secondary provider.

    Tries Groq first; if it fails (missing key, import error), falls
    back to OpenAI.

    Parameters
    ----------
    temperature : float | None
    max_tokens : int | None
    streaming : bool

    Returns
    -------
    BaseChatModel

    Example
    -------
    >>> llm = get_llm_with_fallback(streaming=True)
    """
    try:
        llm = get_llm("groq", temperature, max_tokens, streaming)
        logger.info("[LLMFactory] Using Groq ({})", GROQ_MODEL)
        return llm
    except (ValueError, ImportError) as exc:
        logger.warning("[LLMFactory] Groq unavailable ({}), falling back to OpenAI", exc)

    try:
        llm = get_llm("openai", temperature, max_tokens, streaming)
        logger.info("[LLMFactory] Using OpenAI ({})", OPENAI_MODEL)
        return llm
    except (ValueError, ImportError) as exc:
        logger.error("[LLMFactory] Both providers failed: {}", exc)
        raise ValueError(
            "No LLM provider available. Set GROQ_API_KEY or OPENAI_API_KEY "
            "in your .env file."
        ) from exc


# ── Internal constructors ─────────────────────────────────────────────────


def _get_groq(
    temperature: float, max_tokens: int, streaming: bool
):
    """Build a ChatGroq instance.

    Parameters
    ----------
    temperature : float
    max_tokens : int
    streaming : bool

    Returns
    -------
    ChatGroq
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=GROQ_API_KEY,
        streaming=streaming,
    )


def _get_openai(
    temperature: float, max_tokens: int, streaming: bool
):
    """Build a ChatOpenAI instance.

    Parameters
    ----------
    temperature : float
    max_tokens : int
    streaming : bool

    Returns
    -------
    ChatOpenAI
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        streaming=streaming,
    )
