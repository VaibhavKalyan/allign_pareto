"""Structured JSON + Console logging utilities for ALIGN-PARETO."""
import sys
from pathlib import Path

# Prevent shadowing Python standard library `logging` and `io` when executed directly as `python src/utils/logging.py`
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredLogger:
    """Dual console and JSON-lines logger for reproducible ML experiments."""

    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.name = name
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        payload = {**(extra or {}), **kwargs} or None
        self.logger.info(message)
        self._write_jsonl("INFO", message, payload)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        payload = {**(extra or {}), **kwargs} or None
        self.logger.warning(message)
        self._write_jsonl("WARNING", message, payload)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        payload = {**(extra or {}), **kwargs} or None
        self.logger.error(message)
        self._write_jsonl("ERROR", message, payload)

    def _write_jsonl(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.log_file:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level,
            "message": message,
        }
        if extra:
            record["data"] = extra
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    from pathlib import Path
    log = StructuredLogger("demo_logger")
    log.info("StructuredLogger initialized and verified", step="0b", status="ok")
