import os
from utils.bioorchestra import BioOrchestra


def test_monitor_tracks_changes(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "a.py"
    file_path.write_text("print('a')", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    bo = BioOrchestra(repo_root=str(repo), snapshot_file="snap.json")
    changes = bo.monitor()
    assert str(file_path) in changes["added"]

    file_path.write_text("print('b')", encoding="utf-8")
    changes = bo.monitor()
    assert str(file_path) in changes["modified"]

    file_path.unlink()
    changes = bo.monitor()
    assert str(file_path) in changes["deleted"]
