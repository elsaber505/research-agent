import asyncio
from random import random
from agents.discovery.paper import Paper

import httpx

_SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,publicationDate,abstract,openAccessPdf"

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

async def search_semantic_scholar(
    query: str, max_results: int = 10
) -> list[Paper]:
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
            await asyncio.sleep(1 + random())
        response.raise_for_status()

    papers = []
    for item in response.json().get("data", []):
        # need title, abstract, and PDF to proceed
        if not (title := item.get("title")):
            continue
        if not (abstract := item.get("abstract")):
            continue
        if not (oa := item.get("openAccessPdf")):
            continue
        if not (pdf_url := oa.get("url")):
            continue

        authors = [a["name"] for a in item.get("authors", [])]
        published = item.get("publicationDate")
        if not published:
            year = item.get("year")
            if year:
                published = str(year)
            else:
                published = "unknown"

        papers.append(
            Paper(
                paper_id=item.get("paperId") or "",
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                pdf_url=pdf_url
            )
        )

    return papers
