import os
import json
from datetime import datetime

LOG_PATH = "data/journal.json"
WILDERNESS_PATH = "data/wilderness.md"

def log_event(event):
    """
    Appends an event with a timestamp to the journal log in JSON format.
    Silently ignores errors.
    """
    try:
        if not os.path.isfile(LOG_PATH):
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                f.write("[]")
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)
        log.append({"ts": datetime.now().isoformat(), **event})
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def wilderness_log(fragment):
    """
    Appends a fragment of text to the wilderness markdown log.
    Silently ignores errors.
    """
    try:
        with open(WILDERNESS_PATH, "a", encoding="utf-8") as f:
            f.write(fragment.strip() + "\n\n")
    except Exception:
        pass
