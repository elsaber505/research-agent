from dataclasses import dataclass

@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str  # YYYY-MM-DD format; month and day may be excluded if unavailable
    pdf_url: str = ""
