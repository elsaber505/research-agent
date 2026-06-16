import asyncio

from agents.decomposer import SubQuery, decompose
from agents.discovery import Paper, discover, filter_papers
from agents.reader import PaperSummary, read_paper
from agents.writer import write_report

_MAX_RETRIES = 2


async def _discover_with_retry(sub_query: SubQuery, errors: list[str]) -> list[Paper]:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await discover(sub_query)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                print(f"  [retry {attempt + 1}/{_MAX_RETRIES}] discovery error: {e}", flush=True)
            else:
                errors.append(f"Discovery failed for '{sub_query.query}': {e}")
                return []
    return []


async def _read_with_retry(
    paper: Paper, query: str, errors: list[str]
) -> PaperSummary | None:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await read_paper(paper, query)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                print(
                    f"  [retry {attempt + 1}/{_MAX_RETRIES}] reader error for "
                    f"'{paper.title[:60]}': {e}",
                    flush=True,
                )
            else:
                errors.append(f"Reader failed for '{paper.title}': {e}")
                return None
    return None


async def run_pipeline(query: str) -> tuple[list[PaperSummary], str]:
    """Run the full research pipeline and return (summaries, report)."""
    errors: list[str] = []

    # Stage 1: Decompose
    print("[Decomposer] Breaking query into sub-queries...", flush=True)
    sub_queries = await decompose(query)
    print(f"[Decomposer] Done — {len(sub_queries)} sub-queries:", flush=True)
    for sq in sorted(sub_queries, key=lambda s: s.priority):
        print(f"  [{sq.priority}] {sq.query}", flush=True)

    # Stage 2: Paper discovery -- sequential to respect API rate limits
    print("\n[Discovery] Searching for papers...", flush=True)
    paper_index: dict[str, Paper] = {}
    for i, sq in enumerate(sub_queries, start=1):
        print(f"  ({i}/{len(sub_queries)}) {sq.query}", flush=True)
        raw_papers = await _discover_with_retry(sq, errors)
        filtered = await filter_papers(sq, raw_papers) if raw_papers else []
        print(
            f"    → {len(raw_papers)} found, {len(filtered)} kept after filtering",
            flush=True,
        )
        for p in filtered:
            paper_index[p.paper_id] = p

    all_papers = list(paper_index.values())
    print(f"[Discovery] Done — {len(all_papers)} unique papers total\n", flush=True)

    # Stage 3: Read papers in parallel
    print(f"[Reader] Reading {len(all_papers)} papers in parallel...", flush=True)
    read_tasks = [_read_with_retry(p, query, errors) for p in all_papers]
    results = await asyncio.gather(*read_tasks)
    summaries = [s for s in results if s is not None]
    print(f"[Reader] Done — {len(summaries)}/{len(all_papers)} papers summarized\n", flush=True)

    # Stage 4: Write report
    print("[Writer] Synthesizing final report...", flush=True)
    report = await write_report(query, summaries)
    print("[Writer] Done\n", flush=True)

    if errors:
        print(f"[Errors] {len(errors)} non-fatal error(s) during run:", flush=True)
        for err in errors:
            print(f"  - {err}", flush=True)

    return summaries, report