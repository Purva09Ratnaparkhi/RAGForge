"""
Conversation Memory — Component 10: Summary buffer memory.
============================================================

Manages multi-turn conversation history with a simple message buffer
and LLM-based summarisation for older turns.

Since ``ConversationSummaryBufferMemory`` was removed in LangChain v1.x,
this module implements the same concept directly:
* Recent turns are kept verbatim as ``BaseMessage`` objects.
* When the buffer exceeds ``max_token_limit``, older turns are
  summarised by the LLM and replaced with a single summary message.

Why summary buffer?
--------------------
* **Window memory** (fixed number of turns) loses early context.
* **Full history** blows the context window for long sessions.
* **Summary buffer** is the best of both: recent turns verbatim,
  older turns compressed to a summary.  The 2 000-token limit keeps
  the memory footprint predictable.

Connects to
-----------
* ``src.agents.rag_agent``     — reads history before generation.
* ``src.serving.streamlit_app`` — manages per-session memory.
* ``src.generation.llm_factory`` — provides the LLM for summarisation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger


class ConversationMemoryManager:
    """Manages multi-turn conversation memory with LLM summarisation.

    Holds the LLM used for summarisation and the message buffer.  A class
    is justified because the memory is stateful across the session
    lifetime.

    Parameters
    ----------
    llm
        LangChain LLM for summarising older conversation turns.
    max_token_limit : int
        Approximate maximum characters to keep in the raw buffer
        before summarising older turns.

    Example
    -------
    >>> mgr = ConversationMemoryManager(llm)
    >>> mgr.add_turn("What is BERT?", "BERT is a ...")
    >>> mgr.get_history_string()
    'Human: What is BERT?\\nAI: BERT is a ...'
    """

    def __init__(self, llm, max_token_limit: int = 2000) -> None:
        logger.info(
            "[Memory] Initialising with max_token_limit={}",
            max_token_limit,
        )
        self._llm = llm
        self._max_token_limit = max_token_limit
        self._messages: list[BaseMessage] = []
        self._summary: str = ""

    # ── Public API ────────────────────────────────────────────────────

    def add_turn(self, human_msg: str, ai_msg: str) -> None:
        """Record a single conversation turn.

        Parameters
        ----------
        human_msg : str
            The user's message.
        ai_msg : str
            The AI's response.

        Example
        -------
        >>> mgr.add_turn("What is attention?", "Attention is ...")
        """
        self._messages.append(HumanMessage(content=human_msg))
        self._messages.append(AIMessage(content=ai_msg))
        logger.debug("[Memory] Turn added ({} chars → {} chars)",
                     len(human_msg), len(ai_msg))

        # Check if we need to summarise
        total_chars = sum(len(m.content) for m in self._messages)
        if total_chars > self._max_token_limit:
            self._summarise_older_turns()

    def get_history(self) -> list[BaseMessage]:
        """Return the full message history (summary + recent raw).

        Returns
        -------
        list[BaseMessage]
            LangChain message objects.

        Example
        -------
        >>> msgs = mgr.get_history()
        >>> msgs[0].content
        'What is BERT?'
        """
        result: list[BaseMessage] = []
        if self._summary:
            result.append(SystemMessage(content=f"Previous conversation summary: {self._summary}"))
        result.extend(self._messages)
        return result

    def get_history_string(self) -> str:
        """Return the conversation history as a formatted string.

        Returns
        -------
        str
            Formatted history suitable for prompt injection.
        """
        messages = self.get_history()
        if not messages:
            return ""
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"AI: {msg.content}")
            elif isinstance(msg, SystemMessage):
                lines.append(f"Summary: {msg.content}")
            else:
                lines.append(f"Unknown: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the memory for a new session.

        Example
        -------
        >>> mgr.clear()
        >>> mgr.get_history()
        []
        """
        self._messages.clear()
        self._summary = ""
        logger.info("[Memory] Conversation history cleared")

    def get_summary(self) -> str:
        """Return the current LLM-generated summary of older turns.

        Returns
        -------
        str
            Summary string (empty if no summarisation has occurred).

        Example
        -------
        >>> mgr.get_summary()
        'The user asked about BERT and its applications...'
        """
        return self._summary

    # ── Internal ──────────────────────────────────────────────────────

    def _summarise_older_turns(self) -> None:
        """Summarise the oldest messages and keep only recent ones.

        Keeps the last 4 messages (2 turns) verbatim and summarises
        everything before that.
        """
        if len(self._messages) <= 4:
            return

        # Split: older messages to summarise, recent to keep
        older = self._messages[:-4]
        recent = self._messages[-4:]

        # Format older messages for summarisation
        older_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in older
        )

        prompt = (
            "Progressively summarise the conversation, adding to the "
            "previous summary with new information.\n\n"
            f"Current summary:\n{self._summary}\n\n"
            f"New lines of conversation:\n{older_text}\n\n"
            "Updated summary:"
        )

        try:
            result = self._llm.invoke(prompt)
            self._summary = result.content if hasattr(result, "content") else str(result)
            self._messages = list(recent)
            logger.info("[Memory] Summarised {} older messages", len(older))
        except Exception as exc:
            logger.warning("[Memory] Summarisation failed: {}. Trimming instead.", exc)
            # Fallback: just trim to recent
            self._messages = list(recent)
