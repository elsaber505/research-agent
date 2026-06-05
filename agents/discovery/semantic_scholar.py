import asyncio
from dataclasses import dataclass
from random import random

import httpx

_SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,abstract,externalIds,openAccessPdf,publicationDate"

_MAX_ATTEMPTS = 20

SEARCH_SEMANTIC_SCHOLAR_TOOL = {
    "type": "function",
    "function": {
        "name": "search_semantic_scholar",
        "description": (
            "Search Semantic Scholar for papers matching a query. "
            "Returns only papers with open-access PDFs, including metadata "
            "such as title, authors, abstract, and publication date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
}


@dataclass
class SemanticScholarPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int | None
    published: str  # YYYY-MM-DD, empty string if unknown
    arxiv_id: str   # empty string if not on arXiv
    doi: str        # empty string if unavailable
    pdf_url: str


async def search_semantic_scholar(
    query: str, max_results: int = 10
) -> list[SemanticScholarPaper]:
    params = {
        "query": query,
        "fields": _FIELDS,
        "limit": min(max_results, 100),
        "openAccessPdf": "",
    }
    headers = {"User-Agent": "research-helper/0.1.0"}
    async with httpx.AsyncClient(headers=headers) as client:
        for attempt in range(_MAX_ATTEMPTS):
            response = await client.get(_SS_API, params=params, timeout=15.0)
            if response.status_code != 429:
                break
            await asyncio.sleep(2 + random())
        response.raise_for_status()

    papers = []
    for item in response.json().get("data", []):
        oa = item.get("openAccessPdf") or {}
        external_ids = item.get("externalIds") or {}
        authors = [a["name"] for a in (item.get("authors") or [])]

        papers.append(
            SemanticScholarPaper(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                authors=authors,
                abstract=item.get("abstract") or "",
                year=item.get("year"),
                published=item.get("publicationDate") or "",
                arxiv_id=external_ids.get("ArXiv") or "",
                doi=external_ids.get("DOI") or "",
                pdf_url=oa.get("url", ""),
            )
        )

    return papers
