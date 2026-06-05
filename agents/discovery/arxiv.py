import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

_ARXIV_API = "https://export.arxiv.org/api/query"
_NS = "http://www.w3.org/2005/Atom"

SEARCH_ARXIV_TOOL = {
    "type": "function",
    "function": {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for papers matching a query. "
            "Returns metadata including title, authors, abstract, and PDF URL."
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
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str  # YYYY-MM-DD
    pdf_url: str = ""


async def search_arxiv(query: str, max_results: int = 10) -> list[ArxivPaper]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    headers = {"User-Agent": "research-helper/0.1.0"}
    async with httpx.AsyncClient(headers=headers) as client:
        response = await client.get(_ARXIV_API, params=params, timeout=15.0)
        # debug
        print(response.url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall(f"{{{_NS}}}entry"):
        raw_id = entry.findtext(f"{{{_NS}}}id", "")
        arxiv_id = raw_id.split("/abs/")[-1]

        title = (entry.findtext(f"{{{_NS}}}title") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{{{_NS}}}summary") or "").strip().replace("\n", " ")
        published = (entry.findtext(f"{{{_NS}}}published") or "")[:10]

        authors = [
            author.findtext(f"{{{_NS}}}name") or ""
            for author in entry.findall(f"{{{_NS}}}author")
        ]

        pdf_url = ""
        for link in entry.findall(f"{{{_NS}}}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                pdf_url=pdf_url,
            )
        )

    return papers
