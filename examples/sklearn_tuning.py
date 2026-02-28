"""
sklearn_tuning.py — metaflow-optuna demo

Tunes a RandomForestClassifier on the Wine dataset using the
adaptive coordinator pattern (true TPE, no pre-sampling).

Run locally:
    python sklearn_tuning.py run --n_trials 20

The coordinator step and trial steps run in parallel branches.
After the join, self.study holds the full optuna.Study and
self.study_html is a standalone interactive HTML report.
"""
from metaflow import FlowSpec, Parameter, card, current, step

from metaflow_optuna import (
    hyperparam,
    optuna_coordinator,
    render_study_card,
    render_study_html,
)


class SklearnTuningFlow(FlowSpec):
    """Tune a RandomForest on the Wine dataset with Optuna + Metaflow."""

    n_trials = Parameter("n_trials", default=20, type=int,
                         help="Number of Optuna trials to run")

    @step
    def start(self):
        """Set up coordinator identity and launch parallel branches."""
        self.coordinator_id = current.run_id
        self.n_trials_int   = int(self.n_trials)
        # Kick off coordinator branch AND trial fan-out branch in parallel
        self.next(self.run_coordinator, self.launch_trials)

    # ------------------------------------------------------------------
    # Coordinator branch — hosts the Optuna study service
    # ------------------------------------------------------------------

    @optuna_coordinator(direction="maximize", sampler="tpe")
    @step
    def run_coordinator(self):
        """
        Starts the FastAPI coordinator service that drives the Optuna study.
        Blocks until all n_trials tell() calls arrive, then sets self.study.
        """
        self.next(self.join_study)

    # ------------------------------------------------------------------
    # Trial branch — foreach fan-out, one task per trial
    # ------------------------------------------------------------------

    @step
    def launch_trials(self):
        """Wait for coordinator, then fan out — one task per trial index."""
        from metaflow_optuna.rendezvous import await_coordinator
        # Block until the coordinator has registered its endpoint URL.
        # The URL is written to /tmp (local) or S3 (remote) by run_coordinator.
        self.coordinator_url = await_coordinator(self.coordinator_id, timeout=120)
        self.trial_indices = list(range(self.n_trials_int))
        self.next(self.run_trial, foreach="trial_indices")

    @hyperparam(objective="val_accuracy", direction="maximize")
    @step
    def run_trial(self):
        """
        Single Optuna trial — suggests params, trains RandomForest,
        records self.val_accuracy.  @hyperparam handles the rest.
        """
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        trial = self.trial  # injected by @hyperparam

        n_estimators = trial.suggest_int("n_estimators", 20, 300)
        max_depth    = trial.suggest_int("max_depth", 2, 20)
        min_samples  = trial.suggest_int("min_samples_split", 2, 20)
        max_features = trial.suggest_categorical("max_features",
                                                 ["sqrt", "log2", None])

        X, y = load_wine(return_X_y=True)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples,
            max_features=max_features,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        self.val_accuracy = float(scores.mean())

        self.next(self.join_trials)

    @step
    def join_trials(self, inputs):
        """Collect per-trial results into a list for the join_study step."""
        self.trial_results = [
            inp.trial_result for inp in inputs if hasattr(inp, "trial_result")
        ]
        self.merge_artifacts(inputs, exclude=["trial", "val_accuracy", "trial_result"])
        self.next(self.join_study)

    # ------------------------------------------------------------------
    # Merge coordinator + trial branches
    # ------------------------------------------------------------------

    @card(type="blank")
    @step
    def join_study(self, inputs):
        """
        Merge coordinator branch (self.study) and trial branch.
        Render the Metaflow card and save standalone HTML.
        """
        # Pull study from the coordinator branch input
        study = None
        for inp in inputs:
            if hasattr(inp, "study"):
                study = inp.study
                break

        if study is None:
            # Fallback: rebuild from trial results (batch-mode style)
            from metaflow_optuna import rebuild_study
            trial_inputs = [inp for inp in inputs if not hasattr(inp, "study")]
            study = rebuild_study(
                inputs=trial_inputs,
                objective="val_accuracy",
                direction="maximize",
            )

        self.study = study
        self.study_html = render_study_html(study)

        render_study_card(study)

        self.next(self.end)

    @step
    def end(self):
        """Print best result summary."""
        best = self.study.best_trial
        print(f"\n{'='*50}")
        print(f"Best trial:  #{best.number}")
        print(f"Best value:  {best.value:.4f} (val_accuracy)")
        print(f"Best params: {best.params}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    SklearnTuningFlow()
