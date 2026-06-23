import json

import litellm

from config import FAST_MODEL
from agents.decomposer.decomposer import SubQuery
from agents.discovery.paper import Paper

_MAX_ITEMS = 5

SYSTEM_PROMPT = f"""\
You are a research paper filter. You will be given a query and a list of papers retrieved \
from Semantic Scholar. Select only the papers whose abstracts directly address the query. \
Discard papers that are off-topic or only tangentially related. If more than {_MAX_ITEMS} \
on-topic papers are found, keep only the {_MAX_ITEMS} most relevant ones. 
""" + """
You MUST call the select_papers function with your response.
Example of the required format:
{
    "paper_ids": [
        "abcd1234",
        "nj23nd87",
        "jn2001nk"
    ]
}
Make sure that each ID you submit is actually a paper id from the list you were given.
"""

_SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_papers",
        "description": "Submit the IDs of papers that are relevant to the sub-query.",
        "parameters": {
            "type": "object",
            "maxItems": _MAX_ITEMS,
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paper IDs of the relevant papers",
                }
            },
            "required": ["paper_ids"],
        },
    },
}


def _format_papers(papers: list[Paper]) -> str:
    parts = []
    for p in papers:
        abstract = p.abstract[:500] + ("..." if len(p.abstract) > 500 else "")
        parts.append(
            f"ID: {p.paper_id}\n"
            f"Title: {p.title}\n"
            f"Published: {p.published}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(parts)


async def filter_papers(
    sub_query: SubQuery,
    papers: list[Paper],
) -> list[Paper]:
    if not papers:
        return []

    paper_index = {p.paper_id: p for p in papers}

    response = await litellm.acompletion(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Sub-query: {sub_query.query}\n"
                    f"Rationale: {sub_query.rationale}\n\n"
                    f"Papers:\n\n{_format_papers(papers)}"
                ),
            },
        ],
        tools=[_SELECT_TOOL],
        tool_choice={"type": "function", "function": {"name": "select_papers"}},
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return papers[:_MAX_ITEMS]
    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    paper_ids = args.get("paper_ids", [])
    if isinstance(paper_ids, str):
        print("debug")
        paper_ids = json.loads(paper_ids)

    return [paper_index[pid] for pid in paper_ids if pid in paper_index]
