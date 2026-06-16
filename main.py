import asyncio
import sys

from orchestrator import run_pipeline


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<research query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n{'=' * 60}\n", flush=True)

    _, report = await run_pipeline(query)

    print("=" * 60)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
