from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("artifacts/gbm.joblib")
    approve_below: float = 0.05
    decline_above: float = 0.15

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            approve_below=float(os.getenv("APPROVE_BELOW", cls.approve_below)),
            decline_above=float(os.getenv("DECLINE_ABOVE", cls.decline_above)),
        )
