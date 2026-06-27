import asyncio
import json
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    query: str


class _QueueWriter:
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def write(self, text: str) -> int:
        stripped = text.rstrip("\n")
        if stripped:
            self._queue.put_nowait(stripped)
        return len(text)

    def flush(self):
        pass


async def _stream_pipeline(query: str):
    queue: asyncio.Queue = asyncio.Queue()
    writer = _QueueWriter(queue)

    async def _run():
        try:
            with redirect_stdout(writer):
                _, report = await run_pipeline(query)
            await queue.put({"type": "report", "content": report})
        except Exception as e:
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(_run())

    while True:
        item = await queue.get()
        if item is None:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break
        if isinstance(item, dict):
            yield f"data: {json.dumps(item)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'log', 'message': item})}\n\n"


@app.post("/api/run")
async def run(request: RunRequest):
    return StreamingResponse(
        _stream_pipeline(request.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
