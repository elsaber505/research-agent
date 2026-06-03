# Research Agent — Design Document

## Overview

The Research Agent is an autonomous multi-agent system that answers complex research queries by discovering relevant academic papers, deeply reading and understanding them, verifying factual claims, and producing a structured summary report. The project serves two goals: (1) deliver genuinely useful research synthesis for users, and (2) showcase state-of-the-art agentic platform capabilities — hierarchical orchestration, parallel fan-out, adversarial verification, structured tool use, and resumable runs.

---

## User Experience

The application is a single-page web app. The user types a research query, clicks Run, and watches the agent pipeline execute in real time. When complete, the report renders inline as formatted Markdown with a download button.

**Query input page:**
- Text area for the research query
- Optional: clarifying questions (depth, focus area, max papers)
- Run button

**Live progress view (streams while running):**
```
[Decomposer]   ▶ Thinking...  (click to expand)
               Breaking query into sub-topics... done (4 sub-queries)
[Discovery]    🔍 search_arxiv: "RAG hallucination reduction" → 12 papers
               🔍 search_arxiv: "RLHF factuality" → 8 papers
               Searching complete — 23 candidates found
[Reader]       ▶ Thinking...  (click to expand)
               Reading papers in parallel... 8/12 complete
[Fact Checker] ▶ Thinking...  (click to expand)
               Verifying claims... 2 minor issues flagged
[Writer]       Writing report... done
```

**Report view:**
- Rendered Markdown report with section navigation
- Inline citations linked to source papers
- Download as Markdown or PDF button
- "Run again with different settings" link

**Past runs page:**
- List of previous queries with timestamps
- Click to reload any past report instantly (served from cache)

---

## Architecture

The system has two layers: a **web application layer** (FastAPI backend + React frontend) and an **agent pipeline layer** (the existing orchestrator + specialist agents). They communicate via Server-Sent Events (SSE) for real-time streaming.

```
Browser (React)
    │  POST /api/runs          (submit query)
    │  GET  /api/runs/{id}/stream  (SSE stream for progress)
    │  GET  /api/runs/{id}     (fetch completed report)
    ▼
FastAPI Backend
    │
    ▼
┌─────────────────┐
│   Orchestrator  │  runs as asyncio task, emits progress events
└────────┬────────┘
         │
    ┌────┴──────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   ▼
┌──────────────┐                             ┌──────────────────┐
│  Decomposer  │                             │  Report Writer   │
│  Agent       │                             │  Agent           │
└──────┬───────┘                             └──────────────────┘
       │ sub-queries
       ▼
┌──────────────────────────────────────────────┐
│         Discovery Agents (parallel)          │
│  DiscoveryAgent x N (one per sub-query)      │
└──────────────────┬───────────────────────────┘
                   │ paper candidates
                   ▼
┌──────────────────────────────────────────────┐
│         Reader Agents (parallel)             │
│  ReaderAgent x M (one per paper)             │
└──────────────────┬───────────────────────────┘
                   │ structured summaries
                   ▼
┌──────────────────────────────────────────────┐
│           Fact Checker Agent                 │
│  (adversarial reviewer, fresh context)       │
└──────────────────────────────────────────────┘
```

---

## Agent Specifications

### 1. Orchestrator

**Model:** `STRONG_MODEL` (config)
**Role:** Top-level coordinator. Never does research work directly.

Responsibilities:
- Invokes agents in sequence: Decomposer → Discovery (parallel) → Reader (parallel) → Fact Checker → Report Writer
- Maintains and persists `RunContext` after each stage (checkpoint)
- Emits structured progress events to an asyncio Queue, consumed by the SSE stream endpoint
- Handles retries for failed Discovery or Reader tasks (up to 2 retries each)
- If Fact Checker raises critical concerns, loops back to Reader or Discovery for remediation

### 2. Decomposer Agent

**Model:** `STRONG_MODEL` (config)
**Role:** Break the user query into 3–5 focused sub-queries that map to distinct research areas.

Input: raw user query string
Output: list of `SubQuery` objects, each with:
- `query`: search string optimized for academic search APIs
- `rationale`: one sentence explaining what aspect this covers
- `priority`: 1–3 (1 = most important)

Uses structured tool use (JSON schema output) to guarantee parseable output.

### 3. Discovery Agent

**Model:** `FAST_MODEL` (config)
**Role:** Search for relevant papers given a single sub-query.

One instance spawned per sub-query, all running in parallel via `asyncio.gather`.

Tools available:
- `search_arxiv(query, max_results)` → list of paper metadata
- `search_semantic_scholar(query, max_results)` → list of paper metadata
- `fetch_paper_metadata(arxiv_id_or_doi)` → full metadata + abstract

Output: ranked list of up to 5 `PaperCandidate` objects per sub-query, deduplicated by DOI/arXiv ID at the orchestrator level.

### 4. Reader Agent

**Model:** `STRONG_MODEL` (config)
**Role:** Deeply read and understand a single paper.

One instance spawned per paper candidate, all running in parallel.

Tools available:
- `fetch_pdf(url)` → raw PDF bytes
- `extract_text_from_pdf(pdf_bytes)` → extracted text with section labels
- `fetch_html_abstract(url)` → HTML abstract page (fallback when PDF unavailable)

Output: `PaperSummary` object with:
- `title`, `authors`, `year`, `doi`
- `core_claim`: one sentence
- `methodology`: 2–3 sentences
- `key_findings`: list of strings
- `limitations`: list of strings
- `relevant_quotes`: list of verbatim excerpts with page references
- `relevance_score`: 1–5 (how relevant to the original query)

Reader Agents are the most context-heavy step. Each runs in an isolated context window.

### 5. Fact Checker Agent

**Model:** `STRONG_MODEL` (config, fresh context, no prior session state)
**Role:** Adversarial reviewer. Given the full set of `PaperSummary` objects, identify factual inconsistencies, unsupported claims, and misattributions.

This is the adversarial verification pattern: the Fact Checker has never seen the Reader's work before and approaches it skeptically.

Output: `FactCheckReport` with:
- `issues`: list of `FactIssue` objects (severity: critical/minor, description, affected paper)
- `overall_confidence`: high/medium/low
- `proceed_recommendation`: bool

If `proceed_recommendation` is False, the Orchestrator routes specific papers back to the Reader for re-analysis.

### 6. Report Writer Agent

**Model:** `STRONG_MODEL` (config)
**Role:** Synthesize all `PaperSummary` objects and the `FactCheckReport` into the final Markdown report.

The writer groups findings thematically (not paper-by-paper), resolves conflicts between papers, and flags open questions. It does not add information beyond what is in the summaries — its job is synthesis and clear communication.

---

## LLM Client: LiteLLM

All agents call LLMs through LiteLLM, which provides a unified interface across providers. The model name is the only thing that changes when switching backends.

```python
# config.py
import litellm

# Default model config — change these to switch providers
STRONG_MODEL = "claude-sonnet-4-6"       # Decomposer, Reader, Fact Checker, Report Writer
FAST_MODEL   = "claude-haiku-4-5"        # Discovery Agent

# Alternative configs (uncomment to switch):
# STRONG_MODEL = "gpt-4o"
# FAST_MODEL   = "gpt-4o-mini"

# STRONG_MODEL = "gemini/gemini-1.5-pro"
# FAST_MODEL   = "gemini/gemini-1.5-flash"

# STRONG_MODEL = "groq/llama-3.3-70b-versatile"
# FAST_MODEL   = "groq/llama-3.1-8b-instant"

# STRONG_MODEL = "ollama/qwen3:4b"       # fully local, no API key needed
# FAST_MODEL   = "ollama/qwen3:4b"
```

All agents call LiteLLM the same way regardless of provider:

```python
import litellm
from config import STRONG_MODEL

response = await litellm.acompletion(
    model=STRONG_MODEL,
    messages=[{"role": "user", "content": prompt}],
    tools=tools,        # tool schemas passed here
    tool_choice="auto"
)
```

LiteLLM reads API keys from environment variables automatically (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.). For local Ollama, no key is needed.

**Provider setup (pick one):**

| Provider | Env var | Free tier |
|----------|---------|-----------|
| Anthropic | `ANTHROPIC_API_KEY` | No (pay per token) |
| OpenAI | `OPENAI_API_KEY` | No |
| Gemini | `GEMINI_API_KEY` | Yes (aistudio.google.com) |
| Groq | `GROQ_API_KEY` | Yes (console.groq.com) |
| Ollama (local) | None | Yes (fully local) |
| LiteLLM proxy | `LITELLM_PROXY_URL` | Reuse existing container |

To use an existing LiteLLM proxy container:

```python
# Point litellm at the proxy instead of calling providers directly
import os
os.environ["LITELLM_PROXY_URL"] = "http://localhost:4000"
# Then use model names as configured in the proxy
STRONG_MODEL = "claude-sonnet-4-6"
```

---

## Tool Layer

All tools are implemented as Python async functions with typed signatures and registered with LiteLLM's tool-use API via JSON schema (OpenAI-compatible format, works across all providers).

```python
tools = [
    {
        "name": "search_arxiv",
        "description": "Search arXiv for papers matching a query. Returns metadata including title, authors, abstract, and PDF URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_semantic_scholar",
        "description": "Search Semantic Scholar for papers. Returns citation counts and influence scores in addition to metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_paper_metadata",
        "description": "Fetch full metadata for a specific paper by arXiv ID or DOI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "arXiv ID (e.g. 2301.07094) or DOI"}
            },
            "required": ["identifier"]
        }
    },
    {
        "name": "fetch_pdf",
        "description": "Download a PDF from a URL and return the raw bytes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "extract_text_from_pdf",
        "description": "Extract text from PDF bytes, preserving section structure where possible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pdf_bytes": {"type": "string", "description": "base64-encoded PDF bytes"},
                "max_pages": {"type": "integer", "default": 20}
            },
            "required": ["pdf_bytes"]
        }
    }
]
```

## Web API

FastAPI exposes three endpoints. The agent pipeline runs as a background asyncio task; progress and thinking are streamed to the browser via SSE.

```
POST /api/runs
  Body: { "query": "..." }
  Response: { "run_id": "abc123" }
  → Creates RunContext, starts orchestrator as background task

GET /api/runs/{run_id}/stream
  Response: text/event-stream (SSE)
  → Streams all events (progress, thinking, tool calls) until pipeline completes

GET /api/runs/{run_id}
  Response: { "status": "done", "report": "# Research Report...", "run_context": {...} }
  → Returns completed report and full run metadata

GET /api/runs
  Response: list of past runs (run_id, query, status, created_at)
  → Powers the past runs page
```

### SSE Event Schema

All events share a `type` field. The frontend renders each type differently.

```json
// Pipeline stage progress
{ "type": "progress", "stage": "reader", "message": "Reading paper 4/8", "pct": 55 }

// Model thinking/reasoning (streaming, may arrive in chunks)
{ "type": "thinking", "stage": "decomposer", "agent": "Decomposer", "text": "The query touches three distinct areas..." }

// Tool invocation
{ "type": "tool_call", "stage": "discovery", "agent": "Discovery-1", "tool": "search_arxiv", "input": {"query": "RAG hallucination reduction"} }

// Tool result summary (not full result — too large)
{ "type": "tool_result", "stage": "discovery", "agent": "Discovery-1", "tool": "search_arxiv", "summary": "12 papers found" }

// Pipeline complete
{ "type": "done", "stage": "done" }

// Error
{ "type": "error", "stage": "reader", "message": "Failed to fetch PDF for arxiv:2301.07094" }
```

### Streaming Thinking from LiteLLM

Agents stream LLM responses and emit thinking chunks to the queue as they arrive:

```python
async def call_llm_streaming(model, messages, tools, queue, stage, agent_name):
    stream = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        thinking={"type": "enabled", "budget_tokens": 2000}  # Claude extended thinking
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        # Emit thinking tokens as they arrive
        if hasattr(delta, "thinking") and delta.thinking:
            await queue.put({
                "type": "thinking",
                "stage": stage,
                "agent": agent_name,
                "text": delta.thinking
            })
        # Emit text content tokens
        if delta.content:
            # accumulate for tool use parsing; not streamed to UX directly
            pass
```

Events are put into a per-run `asyncio.Queue` by the Orchestrator and all agents, consumed by the SSE endpoint:

```python
@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str):
    queue = run_queues[run_id]
    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "done":
                break
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Frontend

Single-page React app. Minimal dependencies — no heavy framework needed.

**Pages:**
- `/` — query input form
- `/runs/{run_id}` — live progress + thinking + report view
- `/history` — past runs list

**Key behaviors:**
- On submit, POST to `/api/runs`, then redirect to `/runs/{run_id}`
- On the run page, open an SSE connection to `/api/runs/{run_id}/stream` and render events in real time
- When SSE emits `type: done`, fetch `/api/runs/{run_id}` and render the report using a Markdown renderer (`react-markdown`)
- Past runs page fetches `/api/runs` and lists them with links

**SSE event rendering in `ProgressFeed.tsx`:**

| Event type | UI treatment |
|------------|-------------|
| `progress` | Stage label + animated progress bar |
| `thinking` | Collapsible "Thinking..." block, dimmed text, collapsed by default. Expands on click to show full reasoning. New text appends as chunks arrive. |
| `tool_call` | Inline card: tool name + input summary (e.g. "search_arxiv: RAG hallucination") |
| `tool_result` | Inline card continuation: result summary (e.g. "12 papers found") |
| `error` | Red inline warning, non-blocking |
| `done` | Progress feed freezes, report slides in below |

**Thinking block design (follows Claude.ai pattern):**
```
▶ Decomposer is thinking...          ← collapsed, click to expand
  ───────────────────────────────
  The query touches three distinct
  areas: RAG approaches, RLHF-based
  methods, and inference-time...     ← dimmed monospace text, streams in
```

**Tech:** React + Vite, `react-markdown` for report rendering, `tailwindcss` for styling. No Redux — local state is sufficient.

---



```python
@dataclass
class RunContext:
    run_id: str
    query: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    paper_candidates: list[PaperCandidate] = field(default_factory=list)
    paper_summaries: list[PaperSummary] = field(default_factory=list)
    fact_check_report: FactCheckReport | None = None
    final_report: str | None = None
    stage: str = "init"  # init | decomposed | discovered | read | checked | done
    errors: list[str] = field(default_factory=list)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "RunContext":
        with open(path) as f:
            return RunContext(**json.load(f))
```

The Orchestrator saves `RunContext` to `runs/<run_id>/state.json` after each stage completes. If the process is interrupted, re-running with the same `run_id` resumes from the last completed stage.

---

## Data Flow

```
1. User submits query via browser
        │  POST /api/runs
2. FastAPI creates RunContext, starts orchestrator as background task
        │  returns run_id immediately
3. Browser opens SSE stream GET /api/runs/{run_id}/stream
        │
4. Orchestrator runs pipeline, emits progress events to asyncio.Queue
   Decomposer → SubQuery list (3–5 queries)
        │
5. Discovery (parallel fan-out, one agent per SubQuery)
        │ → PaperCandidate list (deduplicated)
        │
6. Reader (parallel fan-out, one agent per PaperCandidate)
        │ → PaperSummary list
        │
7. Fact Checker (single agent, adversarial review)
        │ → FactCheckReport
        │ ↓ (if issues found, loop back to Reader for affected papers)
8. Report Writer → final Markdown report
        │
9. RunContext saved to disk, SSE emits { stage: "done" }
        │
10. Browser fetches GET /api/runs/{run_id}, renders report
```

---

## Agentic Patterns Showcased

**1. Hierarchical orchestration:** The Orchestrator never does research work; it only coordinates. All research is delegated to specialist agents.

**2. Parallel fan-out with asyncio.gather:** Discovery and Reader agents for independent sub-queries and papers run concurrently, dramatically reducing wall-clock time.

**3. Structured tool use:** All agents use the Claude tool-use API with strict JSON schemas. Outputs are machine-parseable typed objects, not free text.

**4. Adversarial verification (fresh-context reviewer):** The Fact Checker receives no prior session state. It is explicitly designed to find problems the Reader missed — loaded context creates blind spots, fresh context catches them.

**5. Resumable runs via checkpointing:** `RunContext` is persisted to disk after every stage. Interrupted runs resume from the last checkpoint, not from scratch.

**6. Progressive model tiering:** `FAST_MODEL` (e.g. Haiku, Gemini Flash) for cheap, high-volume search and retrieval tasks; `STRONG_MODEL` (e.g. Sonnet, GPT-4o) for deep reading, synthesis, and writing. Provider and model are configured in one place (`config.py`) and swappable without touching agent code.

**7. Self-correcting pipeline:** If the Fact Checker flags issues, the Orchestrator routes specific papers back to the Reader for re-analysis before proceeding to report writing.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM abstraction | LiteLLM (provider-agnostic) |
| LLM backends | Anthropic / OpenAI / Gemini / Groq / Ollama (configurable) |
| Default models | claude-sonnet-4-6 (strong), claude-haiku-4-5 (fast) |
| Backend framework | FastAPI |
| Streaming | Server-Sent Events (SSE) via asyncio.Queue |
| Async concurrency | Python asyncio |
| PDF extraction | pypdf |
| HTTP client | httpx (async) |
| Academic search | arXiv API, Semantic Scholar API, CrossRef API |
| Frontend | React + Vite |
| Markdown rendering | react-markdown |
| Styling | Tailwind CSS |
| Config | python-dotenv (.env file for API keys) |
| State persistence | JSON files in runs/<run_id>/ |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
research-agent/
├── backend/
│   ├── main.py                    # FastAPI app, API endpoints
│   ├── config.py                  # Model config (STRONG_MODEL, FAST_MODEL)
│   ├── orchestrator.py            # Orchestrator agent
│   ├── agents/
│   │   ├── decomposer.py
│   │   ├── discovery.py
│   │   ├── reader.py
│   │   ├── fact_checker.py
│   │   └── report_writer.py
│   ├── tools/
│   │   ├── arxiv.py
│   │   ├── semantic_scholar.py
│   │   ├── pdf.py
│   │   └── registry.py
│   ├── models/
│   │   ├── context.py             # RunContext dataclass
│   │   ├── paper.py               # PaperCandidate, PaperSummary
│   │   └── report.py              # FactCheckReport
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx           # Query input
│   │   │   ├── Run.tsx            # Live progress + report
│   │   │   └── History.tsx        # Past runs
│   │   └── components/
│   │       ├── ProgressFeed.tsx   # SSE event renderer (progress, thinking, tool calls)
│   │       ├── ThinkingBlock.tsx  # Collapsible thinking display
│   │       ├── ToolCallCard.tsx   # Tool invocation + result display
│   │       └── ReportView.tsx     # Markdown report renderer
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env                           # API keys (gitignored)
├── .env.example
└── runs/                          # Checkpoint state (gitignored)
```

---

## Example Report Structure

```
# Research Report: Reducing Hallucinations in Large Language Models

## Executive Summary
...

## Theme 1: Retrieval-Augmented Generation (RAG)
...

## Theme 2: Self-Consistency and Chain-of-Thought
...

## Theme 3: Reinforcement Learning from Human Feedback (RLHF)
...

## Open Questions and Conflicting Evidence
...

## Bibliography
[1] Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401
...
```

---

## Deployment

### Local Development

```bash
# Start backend
cd backend && uvicorn main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev   # runs on localhost:5173, proxies /api to :8000
```

### Docker Compose (single command for demo)

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./runs:/app/runs      # persist run state

  frontend:
    build: ./frontend
    ports: ["3000:80"]        # nginx serving built React app
    depends_on: [backend]
```

```bash
docker-compose up   # app available at http://localhost:3000
```

### Cloud Hosting Options

**Recommended for demo: Railway.app**
- Connect GitHub repo, Railway auto-detects docker-compose
- Free tier: 500 hours/month (enough for demo use)
- Shareable URL in ~5 minutes
- Set API keys as environment variables in Railway dashboard

**Alternative: Render.com**
- Similar to Railway, also free tier
- Deploy backend as a Web Service, frontend as a Static Site
- Frontend: `npm run build` → serve `dist/` as static files

**Alternative: Azure Container Apps**
- If you have Azure credits through work
- More control, scales to zero when idle (no idle cost)
- Deploy with: `az containerapp up`

### Environment Variables

```
# .env.example
ANTHROPIC_API_KEY=        # or GEMINI_API_KEY, GROQ_API_KEY, etc.
STRONG_MODEL=claude-sonnet-4-6
FAST_MODEL=claude-haiku-4-5
CORS_ORIGINS=http://localhost:3000,https://your-app.railway.app
MAX_PAPERS_PER_RUN=12     # cap to control cost
```

### Production Considerations (post-demo)

- **Rate limiting:** Add per-IP rate limiting to prevent abuse (slowapi library)
- **Auth:** Simple API key or GitHub OAuth if opening to others
- **Persistent storage:** Replace JSON files with SQLite or Postgres for run history
- **Queue:** For multiple concurrent users, move pipeline execution to a task queue (Celery or arq) instead of inline asyncio tasks

---



1. **PDF access:** Many papers behind paywalls cannot be fetched. The system falls back to abstract-only analysis, which reduces summary depth.

2. **Context window limits:** Very long papers (100+ pages) must be chunked, which risks losing cross-section connections.

3. **Search coverage:** arXiv and Semantic Scholar have strong coverage of CS/ML but weaker coverage of medicine, law, and social sciences.

4. **Novelty detection:** The system cannot reliably identify which claims are novel vs. well-established — this requires citation graph traversal not yet implemented.

5. **Multi-turn refinement:** Currently single-shot. A future version could let users ask follow-up questions that refine the existing report without re-running the full pipeline.

6. **Cost management:** For broad queries, parallel Reader agents can make 20+ Sonnet API calls. A budget cap and dynamic paper-count adjustment would improve cost predictability.
