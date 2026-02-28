"""Tests for rebuild_study and distribution inference."""
import datetime
import pytest
import optuna

from metaflow_optuna.study import rebuild_study, _build_distributions
from metaflow_optuna.trial import TrialConfig, TrialResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(trial_number, params, value, state="complete"):
    return TrialResult(
        trial_number=trial_number,
        params=params,
        value=value,
        state=state,
        duration_seconds=1.0,
        start_datetime=datetime.datetime(2024, 1, 1),
    )


class _FakeInput:
    def __init__(self, result):
        self.trial_result = result


# ---------------------------------------------------------------------------
# _build_distributions
# ---------------------------------------------------------------------------

def test_build_distributions_float():
    results = [
        _make_result(0, {"lr": 0.01}, 0.9),
        _make_result(1, {"lr": 0.001}, 0.8),
        _make_result(2, {"lr": 0.1}, 0.85),
    ]
    dists = _build_distributions(results)
    assert "lr" in dists
    assert isinstance(dists["lr"], optuna.distributions.FloatDistribution)


def test_build_distributions_int():
    results = [
        _make_result(0, {"layers": 2}, 0.9),
        _make_result(1, {"layers": 4}, 0.8),
    ]
    dists = _build_distributions(results)
    assert isinstance(dists["layers"], optuna.distributions.IntDistribution)


def test_build_distributions_categorical_str():
    results = [
        _make_result(0, {"feat": "log2"}, 0.9),
        _make_result(1, {"feat": "sqrt"}, 0.8),
        _make_result(2, {"feat": "log2"}, 0.85),
    ]
    dists = _build_distributions(results)
    assert isinstance(dists["feat"], optuna.distributions.CategoricalDistribution)


def test_build_distributions_categorical_with_none():
    results = [
        _make_result(0, {"feat": "log2"}, 0.9),
        _make_result(1, {"feat": None}, 0.8),
        _make_result(2, {"feat": "sqrt"}, 0.85),
    ]
    dists = _build_distributions(results)
    assert isinstance(dists["feat"], optuna.distributions.CategoricalDistribution)


# ---------------------------------------------------------------------------
# rebuild_study
# ---------------------------------------------------------------------------

def test_rebuild_study_basic():
    inputs = [
        _FakeInput(_make_result(0, {"lr": 0.01, "layers": 2}, 0.9)),
        _FakeInput(_make_result(1, {"lr": 0.001, "layers": 4}, 0.85)),
        _FakeInput(_make_result(2, {"lr": 0.05, "layers": 3}, 0.92)),
    ]
    study = rebuild_study(inputs, objective="val_acc", direction="maximize")
    assert len(study.trials) == 3
    assert study.best_value == pytest.approx(0.92)
    assert study.best_trial.params["lr"] == pytest.approx(0.05)


def test_rebuild_study_with_failed_trials():
    inputs = [
        _FakeInput(_make_result(0, {"lr": 0.01}, 0.9)),
        _FakeInput(_make_result(1, {"lr": 0.001}, None, state="failed")),
        _FakeInput(_make_result(2, {"lr": 0.05}, 0.92)),
    ]
    study = rebuild_study(inputs, objective="val_acc", direction="maximize")
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]
    assert len(complete) == 2
    assert len(failed) == 1


def test_rebuild_study_importance_works():
    """fANOVA must work after rebuild_study — distributions must be consistent."""
    inputs = [
        _FakeInput(_make_result(i, {"lr": lr, "layers": l, "feat": f}, v))
        for i, (lr, l, f, v) in enumerate([
            (0.01,  2, "log2", 0.90),
            (0.001, 4, "sqrt", 0.85),
            (0.05,  3, "log2", 0.92),
            (0.1,   2, "sqrt", 0.88),
            (0.005, 4, "log2", 0.91),
        ])
    ]
    study = rebuild_study(inputs, objective="val_acc", direction="maximize")
    importances = optuna.importance.get_param_importances(study)
    assert len(importances) > 0, "fANOVA returned empty importances"


def test_rebuild_study_raises_on_all_failed():
    from metaflow_optuna.exceptions import EmptyStudyError
    inputs = [
        _FakeInput(_make_result(0, {"lr": 0.01}, None, state="failed")),
    ]
    with pytest.raises(EmptyStudyError):
        rebuild_study(inputs, objective="val_acc")


# ---------------------------------------------------------------------------
# TrialConfig
# ---------------------------------------------------------------------------

def test_trial_config_round_trip():
    cfg = TrialConfig(trial_number=5, params={"lr": 0.01, "layers": 3}, direction="minimize")
    d = cfg.to_dict()
    cfg2 = TrialConfig.from_dict(d)
    assert cfg2.trial_number == 5
    assert cfg2.params["lr"] == pytest.approx(0.01)
    assert cfg2.direction == "minimize"
