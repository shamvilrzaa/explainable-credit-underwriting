from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from credit_underwriting.decision import DecisionPolicy


@dataclass(frozen=True)
class PolicyChange:
    approve_below: float
    decline_above: float
    changed_by: str
    reason: str
    timestamp: str


class PolicyStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def history_path(self) -> Path:
        return self.path.with_name(self.path.stem + "_history.jsonl")

    def load(self, default: DecisionPolicy | None = None) -> DecisionPolicy:
        if not self.path.exists():
            return default or DecisionPolicy()
        data = json.loads(self.path.read_text())
        return DecisionPolicy(data["approve_below"], data["decline_above"])

    def save(self, policy: DecisionPolicy, changed_by: str, reason: str) -> PolicyChange:
        if not changed_by.strip():
            raise ValueError("Name of the person changing the policy is required")
        if not reason.strip():
            raise ValueError("A reason for the policy change is required")
        change = PolicyChange(
            approve_below=policy.approve_below,
            decline_above=policy.decline_above,
            changed_by=changed_by.strip(),
            reason=reason.strip(),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(policy)))
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(asdict(change)) + "\n")
        return change

    def history(self) -> list[PolicyChange]:
        if not self.history_path.exists():
            return []
        lines = self.history_path.read_text().splitlines()
        return [PolicyChange(**json.loads(line)) for line in lines if line.strip()]
