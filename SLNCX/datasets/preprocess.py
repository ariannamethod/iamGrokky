import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dialogue CSV to plain text")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output text file")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as csv_file, open(
        args.output, "w", encoding="utf-8"
    ) as out_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            line = row.get("Line", "").strip()
            if line:
                out_file.write(line + "\n")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
