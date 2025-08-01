import os
import json
import hashlib
from typing import Dict


class BioOrchestra:
    """Minimal repository scanner used for self-awareness."""

    def __init__(self, repo_root: str = ".", snapshot_file: str = "data/bio_snapshot.json"):
        self.repo_root = repo_root
        self.snapshot_file = snapshot_file

    def _scan(self) -> Dict[str, str]:
        snapshot = {}
        for root, dirs, files in os.walk(self.repo_root):
            if ".git" in dirs:
                dirs.remove(".git")
            for name in files:
                if name.endswith(".py"):
                    path = os.path.join(root, name)
                    try:
                        with open(path, "rb") as fh:
                            digest = hashlib.sha256(fh.read()).hexdigest()
                        snapshot[path] = digest
                    except Exception:
                        continue
        return snapshot

    @staticmethod
    def _diff(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, list]:
        added = [p for p in new if p not in old]
        modified = [p for p in new if p in old and new[p] != old[p]]
        deleted = [p for p in old if p not in new]
        return {"added": added, "modified": modified, "deleted": deleted}

    def monitor(self) -> Dict[str, list]:
        new_snapshot = self._scan()
        if os.path.exists(self.snapshot_file):
            try:
                with open(self.snapshot_file, "r", encoding="utf-8") as f:
                    old_snapshot = json.load(f)
            except Exception:
                old_snapshot = {}
        else:
            old_snapshot = {}

        changes = self._diff(old_snapshot, new_snapshot)
        os.makedirs(os.path.dirname(self.snapshot_file), exist_ok=True)
        with open(self.snapshot_file, "w", encoding="utf-8") as f:
            json.dump(new_snapshot, f, indent=2, ensure_ascii=False)
        return changes
