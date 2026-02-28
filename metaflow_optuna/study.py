"""
Study helpers: create_study_inputs() for batch mode, rebuild_study() for join steps.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

import optuna

from .exceptions import EmptyStudyError, TrialConfigSerializationError
from .trial import TrialConfig, TrialResult


def create_study_inputs(
    search_space: Callable[[optuna.trial.Trial], None],
    n_trials: int,
    sampler: optuna.samplers.BaseSampler | None = None,
    direction: Literal["minimize", "maximize"] = "minimize",
) -> list[TrialConfig]:
    """
    Pre-sample n_trials parameter sets using the given sampler.
    Returns a list[TrialConfig] ready to use as a Metaflow foreach input.

    Note: All trials are sampled before any executes, so TPE cannot adapt
    between trials.  For adaptive sampling use @optuna_coordinator instead.
    Use QMCSampler or RandomSampler for best results in batch mode.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if sampler is None:
        sampler = optuna.samplers.QMCSampler(seed=42)

    study = optuna.create_study(direction=direction, sampler=sampler)
    configs: list[TrialConfig] = []

    for i in range(n_trials):
        trial = study.ask()
        search_space(trial)  # registers suggest_* calls → populates trial.params
        study.tell(trial, state=optuna.trial.TrialState.WAITING)

        # Validate all values are JSON-serializable primitives
        for name, val in trial.params.items():
            if not isinstance(val, (int, float, str, bool)):
                raise TrialConfigSerializationError(
                    f"Parameter '{name}' has non-serializable type {type(val).__name__}. "
                    "Only float, int, str, bool are supported."
                )

        configs.append(TrialConfig(trial_number=i, params=dict(trial.params), direction=direction))

    return configs


def rebuild_study(
    inputs: Iterable[Any],
    objective: str,
    direction: Literal["minimize", "maximize"] = "minimize",
    study_name: str | None = None,
) -> optuna.Study:
    """
    Reconstruct a complete optuna.Study from foreach join inputs.

    Each input must have a `trial_result` attribute (TrialResult).
    Failed tasks (missing trial_result or state="failed") are added as
    FAIL-state trials so the study is complete and accurate.

    Returns a populated optuna.Study with best_params, best_value, trials.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction, storage=None)
    n_complete = 0
    n_failed = 0

    results: list[TrialResult] = []
    for inp in inputs:
        try:
            result: TrialResult = inp.trial_result
        except AttributeError:
            try:
                tc = inp.input
                trial_number = tc.trial_number if isinstance(tc, TrialConfig) else int(tc)
                params: dict[str, Any] = tc.params if isinstance(tc, TrialConfig) else {}
            except Exception:
                trial_number, params = -1, {}
            result = TrialResult(
                trial_number=trial_number,
                params=params,
                value=None,
                state="failed",
                duration_seconds=0.0,
            )
        results.append(result)

    # Build consistent distributions across all trials so fANOVA works
    complete_results = [r for r in results if r.state == "complete" and r.value is not None]
    distributions = _build_distributions(complete_results) if complete_results else {}

    for result in results:
        if result.state == "complete" and result.value is not None:
            _add_complete_trial(study, result, distributions)
            n_complete += 1
        else:
            _add_failed_trial(study, result, distributions)
            n_failed += 1

    if n_complete == 0:
        raise EmptyStudyError(
            f"Zero trials completed successfully ({n_failed} failed). "
            "Check trial step logs for errors."
        )

    if n_failed > 0:
        print(f"[metaflow-optuna] rebuild_study: {n_complete} complete, {n_failed} failed")

    return study


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_complete_trial(
    study: optuna.Study,
    result: TrialResult,
    distributions: dict[str, optuna.distributions.BaseDistribution],
) -> None:
    import optuna.trial as ot

    frozen = ot.FrozenTrial(
        number=result.trial_number,
        trial_id=result.trial_number,
        state=ot.TrialState.COMPLETE,
        value=result.value,
        values=None,
        datetime_start=result.start_datetime,
        datetime_complete=result.start_datetime,
        params=result.params,
        distributions=distributions,
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    study.add_trial(frozen)


def _add_failed_trial(
    study: optuna.Study,
    result: TrialResult,
    distributions: dict[str, optuna.distributions.BaseDistribution],
) -> None:
    import optuna.trial as ot

    frozen = ot.FrozenTrial(
        number=result.trial_number,
        trial_id=result.trial_number,
        state=ot.TrialState.FAIL,
        value=None,
        values=None,
        datetime_start=result.start_datetime,
        datetime_complete=result.start_datetime,
        params=result.params,
        distributions=distributions,
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    study.add_trial(frozen)


def _build_distributions(
    results: list[TrialResult],
) -> dict[str, optuna.distributions.BaseDistribution]:
    """
    Infer consistent per-parameter distributions from all trial results.

    Scans every result to collect the full range/set of values for each
    parameter, then returns one distribution per parameter that covers all
    observed values.  All trials share the same distribution objects, which
    is required for fANOVA importance computation.
    """
    from collections import defaultdict

    all_vals: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        for name, val in r.params.items():
            all_vals[name].append(val)

    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for name, vals in all_vals.items():
        non_none = [v for v in vals if v is not None]

        # Categorical if any value is str, bool, None, or mixed types
        types = {type(v) for v in non_none}
        if not non_none or len(types) > 1 or any(isinstance(v, (str, bool)) for v in non_none):
            unique = sorted({str(v) for v in vals})  # None → "None"
            # Convert back: keep original type where unambiguous
            typed_unique: list[Any] = []
            for v in vals:
                sv = str(v)
                if sv not in [str(u) for u in typed_unique]:
                    typed_unique.append(v)
            distributions[name] = optuna.distributions.CategoricalDistribution(typed_unique)
        elif all(isinstance(v, bool) for v in non_none):
            distributions[name] = optuna.distributions.CategoricalDistribution([False, True])
        elif all(isinstance(v, int) for v in non_none):
            lo, hi = min(non_none), max(non_none)
            distributions[name] = optuna.distributions.IntDistribution(
                low=lo, high=max(hi, lo + 1)
            )
        else:
            lo, hi = float(min(non_none)), float(max(non_none))
            distributions[name] = optuna.distributions.FloatDistribution(
                low=lo, high=max(hi, lo + 1e-9)
            )

    return distributions
