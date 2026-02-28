"""
@hyperparam and @optuna_coordinator step decorators.

Both use functools.wraps to preserve Metaflow's step attributes
(is_step, decorators, wrappers, name) so the graph parser correctly
inspects the original step source code via inspect's __wrapped__ chain.
"""
from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Any, Callable, Literal

from .exceptions import HyperparamError
from .trial import LiveTrial, ReplayTrial, TrialConfig, TrialResult


# ---------------------------------------------------------------------------
# @hyperparam
# ---------------------------------------------------------------------------

def hyperparam(
    objective: str,
    direction: Literal["minimize", "maximize"] = "minimize",
    suppress_logs: bool = True,
    mode: Literal["adaptive", "batch"] = "adaptive",
) -> Callable:
    """
    Step decorator for trial steps.

    adaptive mode (default):
        Reads self.coordinator_url from the parent step's artifacts.
        Injects self.trial as a LiveTrial — suggest_* calls go to the coordinator.

    batch mode:
        Reads self.input as a TrialConfig (from create_study_inputs).
        Injects self.trial as a ReplayTrial — no coordinator needed.

    On soft exceptions: writes TrialResult(state="failed"), calls /tell(FAIL),
    does NOT re-raise — task exits 0 so join_trials/join can proceed with
    partial results.  Hard crashes (SIGKILL) still kill the process; use
    @retry on the train step for those.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self) -> None:
            # ---- Suppress Optuna INFO logs ----
            if suppress_logs:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            # ---- Build trial object ----
            coordinator_url = getattr(self, "coordinator_url", None)

            if mode == "adaptive" and coordinator_url:
                trial = LiveTrial(coordinator_url)
            elif mode == "batch" or (mode == "adaptive" and not coordinator_url):
                raw = getattr(self, "input", None)
                if raw is None:
                    raise HyperparamError(
                        "@hyperparam batch mode: self.input is None. "
                        "Did you set up a foreach with create_study_inputs()?"
                    )
                if isinstance(raw, TrialConfig):
                    config = raw
                elif isinstance(raw, dict) and "params" in raw:
                    config = TrialConfig.from_dict(raw)
                else:
                    raise HyperparamError(
                        f"@hyperparam batch mode: self.input has unexpected type "
                        f"{type(raw).__name__}. Expected TrialConfig or dict."
                    )
                trial = ReplayTrial(config)
            else:
                raise HyperparamError(
                    f"@hyperparam: unrecognised mode={mode!r}. Use 'adaptive' or 'batch'."
                )

            self.trial = trial
            start = time.perf_counter()

            # ---- Execute step body ----
            try:
                func(self)

                # Read objective value
                obj_val = getattr(self, objective, None)
                if obj_val is None:
                    raise HyperparamError(
                        f"@hyperparam: objective attribute '{objective}' not found on self "
                        f"after step body. Did you set self.{objective} = <float>?"
                    )

                self.trial_result = TrialResult(
                    trial_number=trial.number,
                    params=dict(trial.params),
                    value=float(obj_val),
                    state="complete",
                    duration_seconds=time.perf_counter() - start,
                    start_datetime=datetime.utcnow(),
                )

                if hasattr(trial, "_tell"):
                    trial._tell(float(obj_val), "complete")

            except Exception as exc:
                # Soft crash: write failed result, do NOT re-raise
                self.trial_result = TrialResult(
                    trial_number=getattr(trial, "number", -1),
                    params=dict(getattr(trial, "params", {})),
                    value=None,
                    state="failed",
                    duration_seconds=time.perf_counter() - start,
                    start_datetime=datetime.utcnow(),
                )
                if hasattr(trial, "_tell"):
                    try:
                        trial._tell(None, "failed")
                    except Exception:
                        pass

                print(
                    f"[metaflow-optuna] trial #{getattr(trial, 'number', '?')} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Do not re-raise — task exits 0, join proceeds with partial results

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# @optuna_coordinator
# ---------------------------------------------------------------------------

def optuna_coordinator(
    direction: Literal["minimize", "maximize"] = "minimize",
    sampler: str = "tpe",
    port: int | None = None,
    timeout: int = 7200,
    journal_interval: int = 5,
) -> Callable:
    """
    Step decorator for the coordinator branch step.

    Reads self.coordinator_id and self.n_trials_int from the flow namespace.

    What it does:
      1. Starts the FastAPI+uvicorn Optuna service in a background daemon thread.
      2. Discovers its IP and registers the endpoint URL via rendezvous.
      3. Executes the original step body (which should only call self.next(...)).
      4. Blocks until all n_trials_int tell() calls arrive (or timeout).
      5. Sets self.study to the fully populated optuna.Study.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self) -> None:
            from .coordinator import run_coordinator_service

            coordinator_id = getattr(self, "coordinator_id", None)
            n_trials = getattr(self, "n_trials_int", None)

            if coordinator_id is None:
                raise HyperparamError(
                    "@optuna_coordinator: self.coordinator_id not set. "
                    "Did you set self.coordinator_id = current.run_id in start()?"
                )
            if n_trials is None:
                raise HyperparamError(
                    "@optuna_coordinator: self.n_trials_int not set. "
                    "Did you set self.n_trials_int = int(self.n_trials) in start()?"
                )

            # Execute original step body first (records self.next(...))
            func(self)

            # Now start service and block — step process stays alive until done
            study = run_coordinator_service(
                coordinator_id=coordinator_id,
                n_trials=n_trials,
                direction=direction,
                sampler_name=sampler,
                port=port,
                timeout=timeout,
                journal_interval=journal_interval,
            )
            self.study = study

        return wrapper

    return decorator
