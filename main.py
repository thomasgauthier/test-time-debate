import asyncio

from conversation_orchestrator import run_swarm_intelligence_congress


async def main():
    while True:
        query = input("Enter a question about your knowledge base (or 'quit' to exit): ").strip()

        if query.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid question.\n")
            continue

        await run_swarm_intelligence_congress(query)


if __name__ == "__main__":
    asyncio.run(main())
