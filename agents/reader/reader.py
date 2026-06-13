import json

import pymupdf
import httpx
import litellm

from config import STRONG_MODEL
from agents.discovery.paper import Paper
from agents.reader.paper_summary import PaperSummary

_MAX_CHARS = 80_000

_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_summary",
        "description": "Submit the structured summary of the research paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "doi": {
                    "type": "string",
                    "description": "DOI found in the paper text. Empty string if not present.",
                },
                "core_claim": {
                    "type": "string",
                    "description": "One sentence stating the paper's main contribution or finding.",
                },
                "methodology": {
                    "type": "string",
                    "description": "2–3 sentences describing the methods, datasets, or experimental setup.",
                },
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 7,
                    "description": "The paper's main results or conclusions.",
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Acknowledged limitations or weaknesses of the work.",
                },
                "relevant_quotes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "page": {"type": "string"},
                        },
                        "required": ["text", "page"],
                    },
                    "description": "Verbatim quotes from the paper that directly address the research query, with page numbers.",
                },
                "relevance_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "How relevant is this paper to the research query? 1=not relevant, 5=directly and substantially answers it.",
                },
            },
            "required": [
                "doi",
                "core_claim",
                "methodology",
                "key_findings",
                "limitations",
                "relevant_quotes",
                "relevance_score",
            ],
        },
    },
}


def _build_system_prompt(query: str) -> str:
    return f"""\
You are a research paper analyst. Read the paper text provided and produce a structured summary.

Research query you are evaluating against: "{query}"

Rules:
- Base all fields strictly on what is written in the paper. Do not infer or hallucinate.
- If the text was truncated, base your analysis only on what you can see.
- For relevant_quotes: include only verbatim excerpts that directly address the research query above.
- For relevance_score: score 1 if the paper does not address the query even if otherwise interesting, \
and 5 only if it directly and substantially answers the query.
- For doi: extract from the paper text if present (usually on the first page); otherwise return "".
"""


async def _fetch_pdf(url: str) -> bytes | None:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.content
    except Exception:
        return None


def _extract_text(pdf_bytes: bytes) -> str | None:
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                parts.append(f"--- Page {page_num} ---\n{text}")
        full_text = "\n".join(parts)
        return full_text if full_text.strip() else None
    except Exception:
        return None


async def read_paper(paper: Paper, query: str) -> PaperSummary:
    pdf_bytes = await _fetch_pdf(paper.pdf_url)
    text = _extract_text(pdf_bytes) if pdf_bytes is not None else None
    used_full_text = text is not None

    if used_full_text:
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n[... text truncated ...]"
    else:
        text = paper.abstract

    user_message = (
        f"Title: {paper.title}\n"
        f"Published: {paper.published}\n\n"
        f"{'Full text' if used_full_text else 'Abstract (PDF unavailable)'}:\n\n{text}"
    )

    response = await litellm.acompletion(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt(query)},
            {"role": "user", "content": user_message},
        ],
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_summary"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    relevant_quotes = [
        (q["text"], q["page"]) for q in args.get("relevant_quotes", [])
    ]

    return PaperSummary(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=paper.authors,
        published=paper.published,
        doi=args.get("doi", ""),
        core_claim=args["core_claim"],
        methodology=args["methodology"],
        key_findings=args["key_findings"],
        limitations=args["limitations"],
        relevant_quotes=relevant_quotes,
        relevance_score=args["relevance_score"],
        used_full_text=used_full_text,
    )
