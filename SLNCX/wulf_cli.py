import argparse

from .wulf_integration import generate_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Wulf CLI")
    parser.add_argument("prompt", help="prompt text")
    parser.add_argument("--mode", default="grok3", choices=["grok3", "wulf"])
    args = parser.parse_args()
    if args.mode == "wulf":
        from .model import generate as run_wulf

        print(run_wulf(args.prompt))
    else:
        print(generate_response(args.prompt, mode="grok3"))


if __name__ == "__main__":
    main()
