"""
Coordinator service lifecycle management.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    pass


def run_coordinator_service(
    coordinator_id: str,
    n_trials: int,
    direction: str = "minimize",
    sampler_name: str = "tpe",
    port: int | None = None,
    timeout: int = 7200,
    journal_interval: int = 5,
) -> optuna.Study:
    """
    Start the FastAPI coordinator service, register its endpoint, block
    until all n_trials complete, then return the populated Study.

    Called from within the @optuna_coordinator-decorated step.
    """
    from metaflow_coordinator import FastAPIService

    from .app import app, _done
    import metaflow_optuna.coordinator.app as _app_mod

    # --- Build sampler ---
    sampler_map = {
        "tpe":    optuna.samplers.TPESampler(seed=42),
        "random": optuna.samplers.RandomSampler(seed=42),
        "cmaes":  optuna.samplers.CmaEsSampler(seed=42),
        "qmc":    optuna.samplers.QMCSampler(seed=42),
    }
    sampler = sampler_map.get(sampler_name.lower(), optuna.samplers.TPESampler(seed=42))

    # Silence optuna's own logs; we surface results via the card
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Init global state ---
    from metaflow_optuna.rendezvous import load_checkpoint
    prev_completed = load_checkpoint(coordinator_id)

    _app_mod._study = optuna.create_study(direction=direction, sampler=sampler)
    _app_mod._pending = {}
    _app_mod._n_total = n_trials
    _app_mod._coordinator_id = coordinator_id
    _app_mod._completed = prev_completed
    _app_mod._done.clear()
    if prev_completed >= n_trials:
        _app_mod._done.set()

    print(
        f"[metaflow-optuna] starting coordinator for {n_trials} trials ({direction})"
    )

    # --- Delegate lifecycle to FastAPIService ---
    svc = FastAPIService(app=app, done=_done, port=port, timeout=timeout)
    svc.run(service_id=coordinator_id, namespace="metaflow-optuna")

    return _app_mod._study
