import argparse
import asyncio

from .wulf_integration import generate_response


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wulf CLI")
    parser.add_argument("prompt", help="prompt text")
    parser.add_argument("--mode", default="grok3", choices=["grok3", "wulf"])
    args = parser.parse_args()

    async for token in generate_response(args.prompt, mode=args.mode):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
