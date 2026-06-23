import litellm

from config import STRONG_MODEL, STRONG_API_BASE
from agents.reader.paper_summary import PaperSummary

_NUM_CTX = 65_536

SYSTEM_PROMPT = """\
You are a research synthesis writer. Given a set of paper summaries and a research query, \
write a structured Markdown report that synthesizes the findings across all papers.

Rules:
- Organize findings THEMATICALLY, not paper-by-paper. Group papers that address the same aspect together.
- Note conflicts where multiple papers' findings disagree.
- Flag open questions and areas where evidence is limited or mixed.
- Do NOT add any information that is not present in the summaries provided.
- Cite papers inline as [N] where N is the paper number in the bibliography.
- Be precise and concise.

Report structure (you must use exactly these sections):
1. # Research Report: {query}
2. ## Executive Summary  (3–5 sentences covering the main takeaways)
3. ## [Theme 1 title]  (one section per major thematic cluster — you decide the themes)
4. ... (additional theme sections as needed)
5. ## Open Questions & Future Work
6. ## Bibliography  (one entry per paper, format: [N] Authors (year). Title. DOI or paper_id)
"""


def _format_summaries(summaries: list[PaperSummary]) -> str:
    parts = []
    for i, s in enumerate(summaries, start=1):
        authors = ", ".join(s.authors[:3])
        if len(s.authors) > 3:
            authors += " et al."

        quotes = ""
        if s.relevant_quotes:
            quote_lines = "\n".join(
                f'  - "{q}" (p. {p})' for q, p in s.relevant_quotes
            )
            quotes = f"\nRelevant quotes:\n{quote_lines}"

        findings = "\n".join(f"  - {f}" for f in s.key_findings)
        limitations = "\n".join(f"  - {l}" for l in s.limitations) or "  - None reported"

        parts.append(
            f"### Paper [{i}]: {s.title}\n"
            f"Authors: {authors}\n"
            f"Published: {s.published}\n"
            f"Relevance: {s.relevance_score}/5\n"
            f"Source quality: {'full text' if s.used_full_text else 'abstract only'}\n\n"
            f"Core claim: {s.core_claim}\n\n"
            f"Methodology: {s.methodology}\n\n"
            f"Key findings:\n{findings}\n\n"
            f"Limitations:\n{limitations}"
            f"{quotes}"
        )

    return "\n\n---\n\n".join(parts)


async def write_report(
    query: str,
    summaries: list[PaperSummary],
    fact_check_report=None,  # reserved for future FactCheckReport
) -> str:
    if not summaries:
        return f"# Research Report: {query}\n\nNo relevant papers were found for this query."

    summaries = sorted(summaries, key=lambda s: int(s.relevance_score), reverse=True)

    system_prompt = SYSTEM_PROMPT.replace("{query}", query)

    user_message = (
        f"Research query: {query}\n\n"
        f"Papers to synthesize ({len(summaries)} total):\n\n"
        f"{_format_summaries(summaries)}"
    )

    response = await litellm.acompletion(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        api_base=STRONG_API_BASE,
        num_ctx=_NUM_CTX,
    )

    return response.choices[0].message.content