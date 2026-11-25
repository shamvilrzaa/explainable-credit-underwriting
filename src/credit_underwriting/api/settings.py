from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from credit_underwriting.decision import DecisionPolicy
from credit_underwriting.policy_store import PolicyStore


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("artifacts/gbm.joblib")
    policy_path: Path = Path("artifacts/policy.json")
    approve_below: float = 0.05
    decline_above: float = 0.15

    @classmethod
    def from_env(cls) -> Settings:
        policy_path = Path(os.getenv("POLICY_PATH", str(cls.policy_path)))
        stored = PolicyStore(policy_path).load(DecisionPolicy(cls.approve_below, cls.decline_above))
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            policy_path=policy_path,
            approve_below=float(os.getenv("APPROVE_BELOW", stored.approve_below)),
            decline_above=float(os.getenv("DECLINE_ABOVE", stored.decline_above)),
        )

    @property
    def policy(self) -> DecisionPolicy:
        return DecisionPolicy(self.approve_below, self.decline_above)
