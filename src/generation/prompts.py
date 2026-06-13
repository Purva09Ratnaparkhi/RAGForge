"""
Prompt Templates — all LLM prompts used across the RAG pipeline.
=================================================================

Every prompt template in RAGForge is defined here as a module-level
constant.  This centralisation makes it easy to audit, tune, and A/B
test prompts without touching component logic.

Categories
----------
* **QA** — main question-answering prompt (used in generation node).
* **Classification** — query type classification for agentic routing.
* **Rewriting** — query expansion, reformulation, and HyDE prompts.
* **Summarisation** — chunk summarisation for multi-vector retrieval.
* **Guardrails** — groundedness, hallucination, and confidence prompts.
* **Memory** — conversation summary prompt.

Connects to
-----------
* Every component that uses LLM prompts imports from this module.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ── QA Generation ─────────────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_template(
    "You are a research paper expert. Answer the question based ONLY on "
    "the provided context. If the context does not contain enough "
    "information to answer, say 'I don't have enough information to "
    "answer this question based on the uploaded papers.'\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{history}\n\n"
    "Question: {question}\n\n"
    "Provide a detailed, well-structured answer with specific references "
    "to the source material when possible."
)

QA_PROMPT_SIMPLE = ChatPromptTemplate.from_template(
    "You are a research paper expert. Answer the question based ONLY on "
    "the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# ── Query Classification ─────────────────────────────────────────────────

CLASSIFY_QUERY_PROMPT = ChatPromptTemplate.from_template(
    'Classify the following query into exactly one category:\n'
    '- "simple": single factual question answerable from one chunk\n'
    '- "complex": multi-part question requiring synthesis across chunks\n'
    '- "conversational": refers to previous messages (contains "it", '
    '"that", "this", "they", etc. referring to prior context)\n'
    '- "out_of_scope": not related to the uploaded research papers\n\n'
    'Query: "{query}"\n\n'
    'Respond with JSON only: {{"type": "simple|complex|conversational|out_of_scope"}}'
)

# ── Query Rewriting ──────────────────────────────────────────────────────

EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    'Given the query: "{query}"\n'
    "Generate 3 alternative phrasings that mean the same thing but use "
    "different vocabulary. Return them as a numbered list.\n"
    "1."
)

REFORMULATION_PROMPT = ChatPromptTemplate.from_template(
    "Given conversation history:\n{history}\n\n"
    'And the latest query: "{query}"\n\n'
    "Rewrite the query as a complete standalone question with no pronouns "
    "or references to previous messages. Return ONLY the rewritten question."
)

HYDE_PROMPT = ChatPromptTemplate.from_template(
    'Write a detailed paragraph that would be the ideal answer to: "{query}"\n'
    "Base it on academic research paper content. Be specific and technical."
)

# ── Summarisation ────────────────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarise this text in 2-3 sentences capturing the key technical "
    'concepts:\n\n"{text}"'
)

# ── Guardrails ───────────────────────────────────────────────────────────

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_template(
    "Context: {context}\n"
    "Answer: {answer}\n\n"
    "Is every factual claim in the answer directly supported by the context?\n"
    'Respond with JSON: {{"grounded": true/false, "unsupported_claims": ["..."]}}'
)

CONFIDENCE_PROMPT = ChatPromptTemplate.from_template(
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "On a scale of 0-10, how well does the context support this answer?\n"
    'Respond with JSON: {{"confidence": 8, "reasoning": "..."}}'
)

# ── Out of Scope ─────────────────────────────────────────────────────────

OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, but this question doesn't appear to be related to the "
    "uploaded research papers. I can only answer questions about the "
    "content of the documents that have been ingested into the system. "
    "Please ask a question about the research papers you've uploaded."
)

# ── Memory / Conversation ────────────────────────────────────────────────

CONVERSATION_SUMMARY_PROMPT = PromptTemplate.from_template(
    "Progressively summarise the conversation, adding to the previous "
    "summary with new information from the latest exchange.\n\n"
    "Current summary:\n{summary}\n\n"
    "New lines of conversation:\n{new_lines}\n\n"
    "Updated summary:"
)
