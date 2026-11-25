from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReviewRecord:
    applicant_id: int | None
    model_version: str
    model_outcome: str
    probability_of_default: float
    final_outcome: str
    reviewer: str
    note: str
    timestamp: str = ""

    @property
    def is_override(self) -> bool:
        return self.model_outcome != self.final_outcome


class AuditLog:
    COLUMNS = [f.name for f in fields(ReviewRecord)]

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, entry: ReviewRecord) -> ReviewRecord:
        if not entry.reviewer.strip():
            raise ValueError("Reviewer is required")
        if entry.final_outcome not in {"approve", "decline"}:
            raise ValueError("Final outcome must be approve or decline")
        stamped = ReviewRecord(
            **{**asdict(entry), "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(asdict(stamped))
        return stamped

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.path)
