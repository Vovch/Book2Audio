"""Web search helpers for character / book research.

Uses **Exa** when ``EXA_API_KEY`` is set (https://exa.ai), otherwise **DuckDuckGo** text
search via the ``ddgs`` package (no API key; less reliable).
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)


def _search_exa(query: str, num_results: int) -> list[str]:
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        return []

    payload = {
        "query": query,
        "numResults": min(max(num_results, 1), 25),
        "contents": {"text": {"maxCharacters": 2000}},
    }
    try:
        r = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Exa search failed: %s", exc)
        return []

    chunks: list[str] = []
    for item in data.get("results", [])[:num_results]:
        title = item.get("title") or ""
        url = item.get("url") or ""
        text = ""
        inner = item.get("text")
        if isinstance(inner, str):
            text = inner
        elif isinstance(item.get("contents"), dict):
            text = item["contents"].get("text") or ""
        if not text and isinstance(item.get("highlights"), list):
            text = "\n".join(str(h) for h in item["highlights"] if h)
        chunks.append(f"## {title}\nURL: {url}\n{text}".strip())
    return chunks


def _search_duckduckgo(query: str, num_results: int) -> list[str]:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed; web search unavailable.")
        return []

    chunks: list[str] = []
    try:
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max(1, num_results)):
                title = row.get("title") or ""
                body = row.get("body") or ""
                href = row.get("href") or ""
                chunks.append(f"## {title}\nURL: {href}\n{body}".strip())
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
    return chunks


def search_snippets(query: str, *, num_results: int = 6) -> list[str]:
    """Return short text snippets from the public web (best-effort)."""
    q = (query or "").strip()
    if not q:
        return []

    exa = _search_exa(q, num_results)
    if exa:
        logger.info("Search backend: Exa (%d results)", len(exa))
        return exa

    ddg = _search_duckduckgo(q, num_results)
    if ddg:
        logger.info("Search backend: DuckDuckGo (%d results)", len(ddg))
    return ddg


def format_snippets_for_llm(snippets: Sequence[str]) -> str:
    if not snippets:
        return "(No web results; rely on the book title and author only.)"
    parts = []
    for i, s in enumerate(snippets, start=1):
        parts.append(f"[Result {i}]\n{s}")
    return "\n\n".join(parts)
