"""
metaflow-optuna: adaptive hyperparameter tuning with Optuna inside Metaflow.

Quick start (adaptive / coordinator mode):

    from metaflow_optuna import hyperparam, optuna_coordinator
    from metaflow_optuna import render_study_card, render_study_html

Quick start (batch / pre-sampled mode):

    from metaflow_optuna import hyperparam
    from metaflow_optuna import create_study_inputs, rebuild_study
"""
from .decorators import hyperparam, optuna_coordinator
from .study import create_study_inputs, rebuild_study
from .cards import render_study_card, render_study_html
from .trial import TrialConfig, TrialResult

__all__ = [
    "hyperparam",
    "optuna_coordinator",
    "create_study_inputs",
    "rebuild_study",
    "render_study_card",
    "render_study_html",
    "TrialConfig",
    "TrialResult",
]
