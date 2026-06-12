from dataclasses import dataclass


@dataclass
class PaperSummary:
    paper_id: str
    title: str
    authors: list[str]
    published: str
    doi: str                           # empty if not found in text
    core_claim: str                    # one sentence
    methodology: str                   # 2–3 sentences
    key_findings: list[str]
    limitations: list[str]
    relevant_quotes: list[tuple[str, str]]  # (verbatim quote, page reference)
    relevance_score: int               # 1–5
    used_full_text: bool               # False if fell back to abstract only
