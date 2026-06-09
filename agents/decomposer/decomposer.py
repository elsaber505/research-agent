import json
from dataclasses import dataclass

import litellm

from config import STRONG_MODEL

SYSTEM_PROMPT = """\
You are a research query decomposer. Given a user's research question, break it into \
3–5 focused sub-queries that map to distinct research areas or approaches. Each sub-query \
should be phrased as a search string for academic APIs like arXiv and Semantic Scholar.

Rules:
- Each sub-query covers a non-overlapping aspect of the overall question
- Use terminology found in academic paper titles and abstracts
- Priority 1 = most central to the question, 3 = supplementary context
"""

_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_sub_queries",
        "description": "Submit the decomposed sub-queries for the research pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "sub_queries": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search string optimized for academic search APIs",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "One sentence explaining what aspect this covers",
                            },
                            "priority": {
                                "type": "integer",
                                "description": "1–3 where 1 is most important",
                                "minimum": 1,
                                "maximum": 3,
                            },
                        },
                        "required": ["query", "rationale", "priority"],
                    },
                }
            },
            "required": ["sub_queries"],
        },
    },
}


@dataclass
class SubQuery:
    query: str
    rationale: str
    priority: int


async def decompose(query: str) -> list[SubQuery]:
    response = await litellm.acompletion(
        model=STRONG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_sub_queries"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    return [
        SubQuery(
            query=sq["query"],
            rationale=sq["rationale"],
            priority=sq["priority"],
        )
        for sq in args["sub_queries"]
    ]
