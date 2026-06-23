import json
from dataclasses import dataclass

import litellm

from config import FAST_MODEL, FAST_API_BASE

SYSTEM_PROMPT = """\
You are a research query decomposer. Given a user's research question, break it into \
3–5 focused sub-queries that map to distinct research areas or approaches. Each sub-query \
should be phrased as a search string for academic APIs like arXiv and Semantic Scholar.

Rules:
- Each sub-query covers a non-overlapping aspect of the overall question
- Use terminology found in academic paper titles and abstracts
- Priority 1 = most central to the question, 3 = supplementary context

You MUST always call the submit_sub_queries function with your response. For each sub-query \
you provide, the `query`, `rationale`, and `priority` properties are REQUIRED.
Example of the required format:
{
    "sub_queries": [
        {
            "query": "transformer attention mechanism self-supervised learning",
            "rationale": "Covers the core architectural innovation being researched.",
            "priority": 1
        },
        {
            "query": "BERT GPT language model pretraining NLP",
            "rationale": "Covers major implementations of the architecture in practice.",
            "priority": 1
        },
        {
            "query": "vision transformer image classification ViT",
            "rationale": "Covers application of transformers beyond NLP.",
            "priority": 2
        },
        {
            "query": "transformer architecture limitations alternatives survey",
            "rationale": "Provides supplementary context on known weaknesses.",
            "priority": 3
        }
    ]
}
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
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=[_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_sub_queries"}},
        api_base="http://localhost:11434",
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("Model did not return a tool call for submit_sub_queries")
    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    sub_queries = args.get("sub_queries")
    if not sub_queries:
        raise RuntimeError("Model did not return list of sub-queries")
    if isinstance(sub_queries, str):
        print("debug")
        sub_queries = json.loads(sub_queries)

    parsed_sub_queries = []
    for sq in sub_queries:
        query = sq.get("query")
        if not query:
            continue
        rationale = sq.get("rationale", "")
        priority = int(sq.get("priority", 3))
        parsed_sub_queries.append(SubQuery(
            query=query,
            rationale=rationale,
            priority = priority,
        ))
    if not parsed_sub_queries:
        raise RuntimeError("Model returned no valid sub-queries")

    return parsed_sub_queries
