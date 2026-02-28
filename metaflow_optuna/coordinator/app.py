"""
FastAPI application for the Optuna coordinator service.

Single-worker uvicorn: the asyncio event loop serializes all requests, so
study.ask() / study.tell() are never called concurrently — no explicit
locking is needed.

Global state is populated by run_coordinator_service() before uvicorn starts.
"""
from __future__ import annotations

import threading
from typing import Any

import optuna
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="metaflow-optuna coordinator")

# Populated by run_coordinator_service()
_study: optuna.Study | None = None
_pending: dict[int, optuna.trial.Trial] = {}   # trial_number → live Trial
_n_total: int = 0
_completed: int = 0
_done = threading.Event()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TellRequest(BaseModel):
    trial_number: int
    value: float | None = None
    state: str = "complete"   # "complete" | "failed"


class SuggestFloatRequest(BaseModel):
    trial_number: int
    name: str
    low: float
    high: float
    log: bool = False


class SuggestIntRequest(BaseModel):
    trial_number: int
    name: str
    low: int
    high: int
    step: int = 1


class SuggestCategoricalRequest(BaseModel):
    trial_number: int
    name: str
    choices: list[Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask():
    """
    Sample the next hyperparameter configuration.
    TPE is fitted on all completed trials at this point.
    """
    trial = _study.ask()
    _pending[trial.number] = trial
    return {"trial_number": trial.number}


@app.post("/suggest/float")
async def suggest_float(req: SuggestFloatRequest):
    trial = _pending.get(req.trial_number)
    if trial is None:
        raise HTTPException(404, f"No pending trial #{req.trial_number}")
    value = trial.suggest_float(req.name, req.low, req.high, log=req.log)
    return {"value": value}


@app.post("/suggest/int")
async def suggest_int(req: SuggestIntRequest):
    trial = _pending.get(req.trial_number)
    if trial is None:
        raise HTTPException(404, f"No pending trial #{req.trial_number}")
    value = trial.suggest_int(req.name, req.low, req.high, step=req.step)
    return {"value": value}


@app.post("/suggest/categorical")
async def suggest_categorical(req: SuggestCategoricalRequest):
    trial = _pending.get(req.trial_number)
    if trial is None:
        raise HTTPException(404, f"No pending trial #{req.trial_number}")
    value = trial.suggest_categorical(req.name, req.choices)
    return {"value": value}


@app.post("/tell")
async def tell(req: TellRequest):
    global _completed

    trial = _pending.pop(req.trial_number, None)
    if trial is None:
        # Idempotent — already told or unknown (e.g. retry after crash)
        return {"ok": True, "completed": _completed, "total": _n_total}

    if req.state == "complete" and req.value is not None:
        _study.tell(trial, req.value)
    else:
        _study.tell(trial, state=optuna.trial.TrialState.FAIL)

    _completed += 1
    if _completed >= _n_total:
        _done.set()

    return {"ok": True, "completed": _completed, "total": _n_total}


@app.get("/health")
async def health():
    return {"ready": True, "completed": _completed, "total": _n_total}
