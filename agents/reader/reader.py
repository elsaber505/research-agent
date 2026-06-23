import json

import pymupdf
import httpx
import litellm

from config import STRONG_MODEL, STRONG_API_BASE
from agents.discovery.paper import Paper
from agents.reader.paper_summary import PaperSummary

_MAX_CHARS = 80_000
_NUM_CTX = 65_536  # Ollama default is 2048; override to handle full PDF text

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

_FORMAT_EXAMPLE = """\
{
    "doi": "10.48550/arXiv.1706.03762",
    "core_claim": "The Transformer architecture, relying solely on attention mechanisms without recurrence or convolution, achieves superior translation quality while being more parallelizable and faster to train.",
    "methodology": "The authors propose an encoder-decoder architecture using stacked self-attention and feed-forward layers. They evaluate on WMT 2014 English-to-German and English-to-French translation benchmarks using BLEU scores, training on 8 P100 GPUs for 3.5 days.",
    "key_findings": [
        "Transformer achieves 28.4 BLEU on EN-DE translation, outperforming all prior models including ensembles.",
        "Achieves 41.0 BLEU on EN-FR, establishing a new single-model state of the art.",
        "Training cost is significantly lower than recurrent or convolutional alternatives.",
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces."
    ],
    "limitations": [
        "Evaluated only on translation tasks; generalization to other sequence tasks is not fully demonstrated.",
        "Quadratic memory complexity with respect to sequence length limits applicability to very long sequences."
    ],
    "relevant_quotes": [
        {
            "text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "page": "1"
        },
        {
            "text": "The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
            "page": "2"
        }
    ],
    "relevance_score": 5
}\
"""

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

You MUST call the submit_summary function with your response. Include all required fields: \
DOI, core claim, methodology, key findings, limitations, relevant quotes, and relevance score.
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


def _parse_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


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
        f"\n\nExample of the required submit_summary format:\n{_FORMAT_EXAMPLE}"
    )

    response = await litellm.acompletion(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt(query)},
            {"role": "user", "content": user_message},
        ],
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_summary"}},
        api_base=STRONG_API_BASE,
        num_ctx=_NUM_CTX,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError(f"Model did not return a tool call for submit_summary on '{paper.title}'")
    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    key_findings = _parse_str_list(args.get("key_findings", []))
    limitations = _parse_str_list(args.get("limitations", []))

    raw_quotes = args.get("relevant_quotes", [])
    if isinstance(raw_quotes, str):
        try:
            raw_quotes = json.loads(raw_quotes)
        except (json.JSONDecodeError, ValueError):
            raw_quotes = []
    relevant_quotes = []
    for q in raw_quotes:
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(q, dict) and "text" in q:
            relevant_quotes.append((q["text"], q.get("page", "")))

    raw_score = args.get("relevance_score", 1)
    try:
        relevance_score = max(1, min(5, int(raw_score)))
    except (TypeError, ValueError):
        relevance_score = 1

    return PaperSummary(
        paper_id=paper.paper_id,
        title=paper.title,
        authors=paper.authors,
        published=paper.published,
        doi=args.get("doi", ""),
        core_claim=args.get("core_claim", ""),
        methodology=args.get("methodology", ""),
        key_findings=key_findings,
        limitations=limitations,
        relevant_quotes=relevant_quotes,
        relevance_score=relevance_score,
        used_full_text=used_full_text,
    )
