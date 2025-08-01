import os
import json
import time
import hashlib
from datetime import datetime

DATA_PATH = os.getenv("GROKKY_DATA_PATH", "./data")
SNAPSHOT_PATH = os.path.join(DATA_PATH, "vectorized_snapshot.json")
THOUGHTS_PATH = os.path.join(DATA_PATH, "what_is_grokky.md")
STATE_PATH = os.path.join(DATA_PATH, "what_is_grokky_state.json")
README_PATH = os.getenv(
    "GROKKY_README_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md"),
)


def _load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _load_vector_files() -> list[str]:
    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return list(data.keys())
    except Exception:
        pass
    return []


def _summarize(text: str, max_lines: int = 40) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def _default_thoughts() -> str:
    return (
        "README говорит о целях Grokky. Было бы дерзко добавить API для самоосмысления "
        "и ведения векторных дневников."
    )


def reflect_on_readme(force: bool = False) -> str:
    state = _load_state()
    last_ts = state.get("ts", 0)
    last_hash = state.get("readme_hash", "")
    current_hash = _file_hash(README_PATH)
    now = time.time()

    if not force and (now - last_ts < 7 * 24 * 3600) and current_hash == last_hash:
        return "Reflection not needed"

    vector_files = _load_vector_files()

    readme = ""
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme = f.read()
    except Exception:
        pass

    readme_summary = _summarize(readme) if readme else ""

    reflection = [
        f"## Grokky Reflection {datetime.now().isoformat()}",
        "",
        "### Vectorized sources",
        *(f"- {os.path.basename(p)}" for p in vector_files),
        "",
        "### README snapshot",
        readme_summary,
        "",
        "### Thoughts",
        _default_thoughts(),
        "",
    ]

    os.makedirs(os.path.dirname(THOUGHTS_PATH), exist_ok=True)
    with open(THOUGHTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(reflection).strip() + "\n\n")

    _save_state({"ts": now, "readme_hash": current_hash})
    return "Reflection recorded"


def latest_reflection() -> str:
    if not os.path.exists(THOUGHTS_PATH):
        return "No reflections yet."
    try:
        with open(THOUGHTS_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception:
        return "No reflections yet."

    marker = "## Grokky Reflection"
    if marker in text:
        idx = text.rfind(marker)
        return text[idx:]
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grokky identity reflection")
    parser.add_argument("action", choices=["reflect", "latest"], nargs="?", default="reflect")
    parser.add_argument("--force", action="store_true", help="Force a new reflection")
    args = parser.parse_args()

    if args.action == "latest":
        print(latest_reflection())
    else:
        print(reflect_on_readme(force=args.force))
