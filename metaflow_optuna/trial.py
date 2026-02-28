"""
Trial objects: TrialConfig (batch mode), TrialResult (output artifact),
ReplayTrial (batch mode injection), LiveTrial (adaptive mode injection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from .exceptions import CoordinatorUnreachableError, HyperparamError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    """Pre-sampled trial for batch (non-adaptive) mode. Used as foreach input."""
    trial_number: int
    params: dict[str, Any]  # {name: value} — all primitives (float|int|str|bool)
    direction: str = "minimize"

    def to_dict(self) -> dict:
        return {"trial_number": self.trial_number, "params": self.params, "direction": self.direction}

    @classmethod
    def from_dict(cls, d: dict) -> "TrialConfig":
        return cls(trial_number=d["trial_number"], params=d["params"], direction=d.get("direction", "minimize"))


@dataclass
class TrialResult:
    """Written as self.trial_result by @hyperparam. Read by rebuild_study / join_study."""
    trial_number: int
    params: dict[str, Any]
    value: float | None       # None if failed
    state: str                # "complete" | "failed"
    duration_seconds: float
    start_datetime: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# ReplayTrial — batch mode
# ---------------------------------------------------------------------------

class ReplayTrial:
    """
    Implements Optuna's trial.suggest_* interface by replaying pre-sampled values
    from a TrialConfig.  Range arguments are accepted but ignored — values are
    already determined.  Raises a descriptive KeyError on name mismatches.
    """

    def __init__(self, config: TrialConfig) -> None:
        self.number = config.trial_number
        self.params: dict[str, Any] = dict(config.params)

    # --- suggest_* interface ---

    def suggest_float(self, name: str, low: float = 0.0, high: float = 1.0, **kwargs) -> float:
        return float(self._get(name))

    def suggest_int(self, name: str, low: int = 0, high: int = 1, **kwargs) -> int:
        return int(self._get(name))

    def suggest_categorical(self, name: str, choices: Sequence = (), **kwargs) -> Any:
        return self._get(name)

    def suggest_loguniform(self, name: str, low: float = 1e-5, high: float = 1.0, **kwargs) -> float:
        return float(self._get(name))

    def suggest_uniform(self, name: str, low: float = 0.0, high: float = 1.0, **kwargs) -> float:
        return float(self._get(name))

    def suggest_discrete_uniform(self, name: str, low: float = 0.0, high: float = 1.0, q: float = 1.0, **kwargs) -> float:
        return float(self._get(name))

    def suggest_int_loguniform(self, name: str, low: int = 1, high: int = 100, **kwargs) -> int:
        return int(self._get(name))

    def set_user_attr(self, key: str, value: Any) -> None:
        pass  # no-op in replay mode

    def __getitem__(self, name: str) -> Any:
        return self._get(name)

    def _get(self, name: str) -> Any:
        if name not in self.params:
            raise KeyError(
                f"'{name}' not found in trial params. "
                f"Available: {list(self.params.keys())}. "
                "Check that search_space in create_study_inputs() and "
                "suggest_* calls in the trial step use identical parameter names."
            )
        return self.params[name]


# ---------------------------------------------------------------------------
# LiveTrial — adaptive mode (coordinator)
# ---------------------------------------------------------------------------

class LiveTrial:
    """
    Implements Optuna's trial.suggest_* interface via HTTP calls to the
    coordinator service.  On construction, calls POST /ask to get a trial
    number from the live study (TPE-guided).  Each suggest_* call is one
    HTTP round-trip that registers the parameter and returns its value.
    """

    def __init__(self, coordinator_url: str, timeout: float = 30.0) -> None:
        self._url = coordinator_url.rstrip("/")
        self._timeout = timeout
        self.number: int | None = None
        self.params: dict[str, Any] = {}
        self._ask()

    def _ask(self) -> None:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(10):
            try:
                r = httpx.post(f"{self._url}/ask", timeout=self._timeout)
                r.raise_for_status()
                self.number = r.json()["trial_number"]
                return
            except Exception as exc:
                last_exc = exc
                wait = min(2 ** attempt * 0.5, 30)
                time.sleep(wait)
        raise CoordinatorUnreachableError(
            f"Could not reach coordinator at {self._url} after 10 attempts. "
            f"Last error: {last_exc}"
        )

    def _suggest(self, endpoint: str, payload: dict) -> Any:
        import httpx

        payload["trial_number"] = self.number
        r = httpx.post(f"{self._url}/{endpoint}", json=payload, timeout=self._timeout)
        r.raise_for_status()
        value = r.json()["value"]
        self.params[payload["name"]] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False, **kwargs) -> float:
        return float(self._suggest("suggest/float", {"name": name, "low": low, "high": high, "log": log}))

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, **kwargs) -> int:
        return int(self._suggest("suggest/int", {"name": name, "low": low, "high": high, "step": step}))

    def suggest_categorical(self, name: str, choices: Sequence, **kwargs) -> Any:
        return self._suggest("suggest/categorical", {"name": name, "choices": list(choices)})

    def suggest_loguniform(self, name: str, low: float, high: float, **kwargs) -> float:
        return self.suggest_float(name, low, high, log=True)

    def suggest_uniform(self, name: str, low: float, high: float, **kwargs) -> float:
        return self.suggest_float(name, low, high, log=False)

    def set_user_attr(self, key: str, value: Any) -> None:
        pass

    def __getitem__(self, name: str) -> Any:
        return self.params[name]

    def _tell(self, value: float | None, state: str = "complete") -> None:
        import httpx

        try:
            httpx.post(
                f"{self._url}/tell",
                json={"trial_number": self.number, "value": value, "state": state},
                timeout=self._timeout,
            )
        except Exception:
            pass  # best-effort; coordinator already counting
