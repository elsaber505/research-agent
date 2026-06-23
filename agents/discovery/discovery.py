import json

import litellm

from config import FAST_MODEL, FAST_API_BASE
from agents.decomposer.decomposer import SubQuery
from agents.discovery.semantic_scholar import (
    search_semantic_scholar,
    SEARCH_SEMANTIC_SCHOLAR_TOOL,
)
from agents.discovery.paper import Paper

SYSTEM_PROMPT = """\
You are a research paper discovery agent. Given a search query, your job is to collect \
papers using the search API tools provided to you.
After searching with the provided query, search multiple times with varied queries \
and terminology to maximize coverage. When you have made enough searches, \
call finish_search.
"""

_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_search",
        "description": "Signal that you have finished searching and are done collecting papers.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOLS = [SEARCH_SEMANTIC_SCHOLAR_TOOL, _FINISH_TOOL]
_MAX_TURNS = 6


async def discover(sub_query: SubQuery) -> list[Paper]:
    paper_index: dict[str, Paper] = {}  # keys are paper ids

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sub-query: {sub_query.query}\n"
                f"Rationale: {sub_query.rationale}"
            ),
        },
    ]

    for _ in range(_MAX_TURNS):
        response = await litellm.acompletion(
            model=FAST_MODEL,
            messages=messages,
            tools=_TOOLS,
            api_base=FAST_API_BASE,
        )

        message = response.choices[0].message

        assistant_msg: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            break

        done = False
        tool_results = []

        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "finish_search":
                done = True
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Search complete.",
                })

            elif name == "search_semantic_scholar":
                results = await search_semantic_scholar(
                    query=args["query"],
                    max_results=int(args.get("max_results", 10)),
                )
                for p in results:
                    paper_index[p.paper_id] = p
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Found {len(results)} papers. Total collected: {len(paper_index)}.",
                })

        messages.extend(tool_results)

        if done:
            break

    return list(paper_index.values())
