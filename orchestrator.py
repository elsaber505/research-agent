import asyncio
from collections.abc import Callable

from agents.decomposer import SubQuery, decompose
from agents.discovery import Paper, discover, filter_papers
from agents.reader import PaperSummary, read_paper
from agents.writer import write_report
from config import READER_BATCH_SIZE

_MAX_RETRIES = 2


async def _discover_with_retry(
    sub_query: SubQuery, errors: list[str], log: Callable[[str], None]
) -> list[Paper]:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await discover(sub_query)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                log(f"  [retry {attempt + 1}/{_MAX_RETRIES}] discovery error: {e}")
            else:
                errors.append(f"Discovery failed for '{sub_query.query}': {e}")
                return []
    return []


async def _read_with_retry(
    paper: Paper, query: str, errors: list[str], log: Callable[[str], None]
) -> PaperSummary | None:
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await read_paper(paper, query)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                log(
                    f"  [retry {attempt + 1}/{_MAX_RETRIES}] reader error for "
                    f"'{paper.title[:60]}': {e}"
                )
            else:
                errors.append(f"Reader failed for '{paper.title}': {e}")
                return None
    return None


async def run_pipeline(
    query: str, log: Callable[[str], None] = print
) -> tuple[list[PaperSummary], str]:
    """Run the full research pipeline and return (summaries, report)."""
    errors: list[str] = []

    # Stage 1: Decompose
    log("[Decomposer] Breaking query into sub-queries...")
    sub_queries = await decompose(query)
    log(f"[Decomposer] Done — {len(sub_queries)} sub-queries:")
    for sq in sorted(sub_queries, key=lambda s: int(s.priority)):
        log(f"  [{sq.priority}] {sq.query}")

    # Stage 2: Paper discovery -- sequential to respect API rate limits
    log("\n[Discovery] Searching for papers...")
    paper_index: dict[str, Paper] = {}
    for i, sq in enumerate(sub_queries, start=1):
        log(f"  ({i}/{len(sub_queries)}) {sq.query}")
        raw_papers = await _discover_with_retry(sq, errors, log)
        if not raw_papers:
            log(f"    → No papers found")
            continue
        filtered = await filter_papers(sq, raw_papers) if raw_papers else []
        log(f"    → {len(raw_papers)} found, {len(filtered)} kept after filtering")
        for p in filtered:
            paper_index[p.paper_id] = p

    all_papers = list(paper_index.values())
    log(f"[Discovery] Done — {len(all_papers)} unique papers total\n")

    # Stage 3: Read papers in batches
    batch_size = READER_BATCH_SIZE or len(all_papers)
    batches = [all_papers[i:i + batch_size] for i in range(0, len(all_papers), batch_size)]
    log(
        f"[Reader] Reading {len(all_papers)} papers"
        f" ({len(batches)} batch(es) of up to {batch_size})..."
    )
    results: list[PaperSummary | None] = []
    for i, batch in enumerate(batches, start=1):
        log(f"  Batch {i}/{len(batches)} ({len(batch)} papers)...")
        batch_results = await asyncio.gather(
            *[_read_with_retry(p, query, errors, log) for p in batch]
        )
        results.extend(batch_results)
    summaries = [s for s in results if s is not None]
    log(f"[Reader] Done — {len(summaries)}/{len(all_papers)} papers summarized\n")

    # Stage 4: Write report
    log("[Writer] Synthesizing final report...")
    report = await write_report(query, summaries)
    log("[Writer] Done\n")

    if errors:
        log(f"[Errors] {len(errors)} non-fatal error(s) during run:")
        for err in errors:
            log(f"  - {err}")

    return summaries, report
