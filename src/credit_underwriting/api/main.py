from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request

from credit_underwriting import __version__
from credit_underwriting.api.schemas import (
    Application,
    BatchRequest,
    BatchResponse,
    DecisionOut,
    ModelInfo,
    to_frame,
)
from credit_underwriting.api.settings import Settings
from credit_underwriting.data.schema import FEATURES
from credit_underwriting.decision import Decision, Underwriter

log = logging.getLogger(__name__)


def load_underwriter(settings: Settings) -> tuple[Underwriter, str]:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Model not found at {settings.model_path}; run `make train-gbm`")
    version = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()[:12]
    return Underwriter(joblib.load(settings.model_path), policy=settings.policy), version


def create_app(underwriter: Underwriter | None = None, model_version: str = "test") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if underwriter is None:
            app.state.underwriter, app.state.model_version = load_underwriter(Settings.from_env())
        else:
            app.state.underwriter, app.state.model_version = underwriter, model_version
        log.info("Loaded model %s", app.state.model_version)
        yield

    app = FastAPI(
        title="Credit Underwriting API",
        version=__version__,
        description="Probability of default, score, decision and reasons for loan applications.",
        lifespan=lifespan,
    )

    def get_underwriter(request: Request) -> Underwriter:
        return request.app.state.underwriter

    def to_response(decision: Decision, version: str) -> DecisionOut:
        return DecisionOut(**decision.to_dict(), model_version=version)

    def decide(applications: list[Application], request: Request) -> list[DecisionOut]:
        try:
            decisions = get_underwriter(request).decide(to_frame(applications))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return [to_response(d, request.app.state.model_version) for d in decisions]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        underwriter = get_underwriter(request)
        return ModelInfo(
            model_version=request.app.state.model_version,
            approve_below=underwriter.policy.approve_below,
            decline_above=underwriter.policy.decline_above,
            features=list(FEATURES),
        )

    @app.post("/v1/decisions", response_model=DecisionOut)
    def decide_one(application: Application, request: Request) -> DecisionOut:
        return decide([application], request)[0]

    @app.post("/v1/decisions/batch", response_model=BatchResponse)
    def decide_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        return BatchResponse(decisions=decide(batch.applications, request))

    return app


app = create_app()
