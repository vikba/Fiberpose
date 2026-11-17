"""Usage log of the web interface: one JSON line per analysis, kept in a private Hugging Face dataset.

Off unless ``FIBERPOSE_USAGE_REPO`` names the dataset; uploading needs a write token in ``HF_TOKEN``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_scheduler = None  # uploads the folder of `_file` every few minutes
_file: Path | None = None
_failed = False


def log(**fields) -> None:
    """Append one event; never raises, and stops trying after the first failure."""
    global _scheduler, _file, _failed
    repo = os.environ.get("FIBERPOSE_USAGE_REPO")
    if not repo or _failed:
        return
    try:
        with _lock:
            if _scheduler is None:
                from huggingface_hub import CommitScheduler

                folder = Path(tempfile.mkdtemp(prefix="fiberpose_usage_"))
                # A Space loses its disk on restart, so every process logs to a file of its own:
                # a fixed name would replace the history already in the dataset.
                _file = folder / f"events-{uuid.uuid4().hex}.jsonl"
                _scheduler = CommitScheduler(repo_id=repo, repo_type="dataset", folder_path=folder, private=True, every=5)
            event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
            with _scheduler.lock, _file.open("a") as lines:
                lines.write(json.dumps(event) + "\n")
    except Exception as exc:  # logging must never break an analysis
        _failed = True
        print(f"Usage logging disabled: {exc}")
