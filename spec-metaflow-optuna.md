# Spec: metaflow-optuna
**Status:** Draft
**Date:** 2026-02-27
**Author:** Claude

---

## Problem Statement

Hyperparameter tuning with Optuna is the de-facto standard workflow for ML practitioners using Metaflow, and Optuna is officially documented in Metaflow's own docs. Despite this, there is no first-class integration: users wire up `study.optimize()` manually inside a single Metaflow task, which serializes all trials onto one machine, produces 50–200 lines of `[I YYYY-MM-DD HH:MM:SS] Trial N finished with value...` log spam that drowns out real training output, and produces no structured result view. When practitioners do attempt to parallelize using Metaflow's foreach, they pre-sample all hyperparameter configs upfront — which defeats the purpose of a smart sampler like TPE, since the sampler has no outcome data at sample time and degrades to random search. This library, `metaflow-optuna`, closes that gap with an ephemeral coordinator service that gives each trial genuine TPE guidance from all previously-completed results, requires no external database, and produces a rich results card.

---

## Goals

- Trial N is sampled by a live TPE model fitted on the outcomes of trials 0..N-1: no pre-sampling, no degraded-to-random behavior.
- A 50-trial tuning run completes with zero trial-level log spam in any task's stdout/stderr.
- Parallel foreach fan-out requires no user-written coordination plumbing beyond adding `@optuna_coordinator` to one step and `@hyperparam` to the trial step.
- The Optuna study card renders in < 3 seconds for studies up to 500 trials and contains: best-trial hero, optimization history, parallel coordinates, parameter importance, and a sortable trial table.
- The `@hyperparam` trial step accepts native `trial.suggest_*` calls — existing single-machine Optuna objective functions port to the parallel coordinator model with ≤ 5 line changes.
- Works transparently with `@batch` and `@kubernetes` on trial steps via VPC-internal networking; no external database or Redis instance required.

---

## Non-Goals

- **Pruning / intermediate reporting**: `trial.should_prune()` and `trial.report()` require round-trips to the coordinator mid-step. Adds latency and complexity disproportionate to v1 scope. Deferred to v2.
- **Multi-objective optimization**: single-objective studies only in v1.
- **Coordinator high availability / replication**: the coordinator is a single-process service. It is crash-tolerant via study journaling to S3 (see Failure Modes), but not replicated.
- **Optuna dashboard or real-time study monitoring UI**: the card is rendered post-hoc at join time. No live dashboard.
- **Hyperparameter tuning result persistence outside Metaflow artifacts**: the reconstructed `optuna.Study` is a Metaflow artifact; no external database writes.
- **Batch / pre-sampled mode as a first-class feature**: pre-sampling is available as `mode="batch"` on `@hyperparam` for grid/random search that needs no coordinator, but it is not the primary design.

---

## Background and Context

Optuna's `MLflowCallback` (in `optuna_integration`) is the direct prior art for the callback pattern. Optuna's own parallel execution model — `study.optimize()` with `n_jobs > 1` — uses a shared storage backend (SQLite, PostgreSQL, or Redis) as a coordination bus: each worker calls `study.ask()`, runs a trial, and calls `study.tell()`. The ask/tell split is the key: `ask()` samples the next params from TPE's model, and `tell()` reports the result and updates the model.

This library ports the ask/tell protocol to Metaflow by running an ephemeral coordinator service that owns the `optuna.Study` in-memory. The coordinator exposes a small HTTP API that mirrors Optuna's `trial.suggest_*` interface; trial tasks are Metaflow `@batch` or `@kubernetes` tasks that call this API over the VPC-internal network. No external database is needed because the coordinator process itself is the storage — it is part of the flow, started before the foreach, and terminated when all trials complete.

Metaflow's parallel branch pattern (calling `self.next(step_a, step_b)` from a single step) allows two independent chains to run simultaneously and join at a later step. This is the structural primitive that enables the coordinator to run alongside the foreach: one branch hosts the coordinator, the other fans out the trial tasks.

S3 is used as a lightweight rendezvous bus: the coordinator writes its private IP and port to a well-known S3 key under Metaflow's own bucket immediately on startup. Trial tasks read this key (with exponential-backoff polling) before beginning their work. No new infrastructure is needed beyond what Metaflow already requires.

---

## Design

### API / Interface

#### Complete flow example

```python
from metaflow import FlowSpec, step, Parameter, card, batch, current
from metaflow_optuna import optuna_coordinator, hyperparam, await_coordinator, rebuild_study

class XGBoostTuning(FlowSpec):

    n_trials = Parameter("n_trials", default=50)

    @step
    def start(self):
        self.coordinator_id = current.run_id   # unique per run; used as S3 rendezvous key
        self.n_trials_int = int(self.n_trials)
        self.next(self.run_coordinator, self.launch_trials)

    @optuna_coordinator(direction="minimize", sampler="tpe")
    @batch(cpu=1, memory=512)
    @step
    def run_coordinator(self):
        # Decorator handles everything:
        #   - starts FastAPI+uvicorn in background thread
        #   - discovers own VPC private IP
        #   - registers endpoint in S3 at coordinator_id key
        #   - blocks until n_trials_int tell() calls received
        #   - journals study state to S3 after each tell() for crash recovery
        #   - populates self.study when done
        self.next(self.join_study)

    @step
    def launch_trials(self):
        self.coordinator_url = await_coordinator(
            self.coordinator_id,    # polls s3://.../metaflow-optuna/{id}/endpoint
            timeout=120,
        )
        self.trial_ids = list(range(self.n_trials_int))
        self.next(self.train, foreach="trial_ids")

    @hyperparam(objective="val_loss")
    @batch(cpu=4, memory=16000)
    @step
    def train(self):
        # self.trial is a LiveTrial — suggest_* calls hit the coordinator via HTTP.
        # TPE sees all previously-completed trials' outcomes before sampling this one.
        lr    = self.trial.suggest_float("lr", 1e-4, 0.3, log=True)
        depth = self.trial.suggest_int("depth", 3, 10)
        sub   = self.trial.suggest_float("subsample", 0.6, 1.0)

        self.val_loss = run_training(lr=lr, depth=depth, subsample=sub)
        self.next(self.join_trials)

    @step
    def join_trials(self, inputs):
        self.trial_results = [inp.trial_result for inp in inputs
                              if hasattr(inp, "trial_result")]
        self.next(self.join_study)

    @card(type="optuna_study")
    @step
    def join_study(self, inputs):
        # One input is from run_coordinator (has self.study).
        # The other is from join_trials (has self.trial_results).
        coord = next(i for i in inputs if hasattr(i, "study"))
        self.study      = coord.study
        self.best_params = self.study.best_params
        self.best_value  = self.study.best_value
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    XGBoostTuning()
```

#### `@optuna_coordinator` decorator

```python
def optuna_coordinator(
    direction: Literal["minimize", "maximize"] = "minimize",
    sampler: Literal["tpe", "random", "cmaes", "qmc"] | optuna.samplers.BaseSampler = "tpe",
    port: int = 8765,
    journal_interval: int = 5,    # write study snapshot to S3 every N tell() calls
    timeout: int = 7200,          # seconds before coordinator gives up waiting
) -> Callable:
    """
    Step decorator for the coordinator branch.

    Reads self.coordinator_id and self.n_trials_int from the flow namespace.
    Starts the Optuna coordinator service, blocks until all trials report,
    then sets self.study to the populated optuna.Study.

    Must be placed above @step. The decorated step body should contain
    only self.next(...).
    """
```

#### `await_coordinator` function

```python
def await_coordinator(
    coordinator_id: str,       # current.run_id from start step
    timeout: int = 120,        # seconds to wait before raising CoordinatorNotReadyError
    poll_interval: float = 2.0,
) -> str:
    """
    Polls s3://{metaflow_bucket}/metaflow-optuna/{coordinator_id}/endpoint
    with exponential backoff until the coordinator registers its URL.
    Returns the coordinator base URL, e.g. "http://10.0.3.47:8765".
    """
```

#### `@hyperparam` decorator

```python
def hyperparam(
    objective: str,                                       # self attribute holding the metric
    direction: Literal["minimize", "maximize"] = "minimize",
    suppress_logs: bool = True,
    mode: Literal["adaptive", "batch"] = "adaptive",      # "batch" = pre-sampled ReplayTrial
) -> Callable:
    """
    Step decorator for trial steps.

    adaptive mode (default):
        Reads self.coordinator_url from parent step namespace.
        Injects self.trial as a LiveTrial — suggest_* calls hit the coordinator.

    batch mode:
        Reads self.input as a TrialConfig (pre-sampled dict).
        Injects self.trial as a ReplayTrial — no coordinator needed.
        Use for grid/random search when adaptivity is not required.
    """
```

#### Batch (pre-sampled) mode — no coordinator needed

For grid or random search where true adaptivity is not needed:

```python
from metaflow_optuna import create_study_inputs, hyperparam

@step
def start(self):
    def search_space(trial):
        trial.suggest_float("lr", 1e-4, 0.3, log=True)
        trial.suggest_int("depth", 3, 10)

    self.trial_configs = create_study_inputs(
        search_space=search_space,
        n_trials=int(self.n_trials),
        sampler=optuna.samplers.QMCSampler(seed=42),  # quasi-random, good coverage
        direction="minimize",
    )
    self.next(self.train, foreach="trial_configs")

@hyperparam(objective="val_loss", mode="batch")
@step
def train(self):
    lr    = self.trial.suggest_float("lr", 1e-4, 0.3, log=True)
    depth = self.trial.suggest_int("depth", 3, 10)
    self.val_loss = run_training(lr=lr, depth=depth)
    self.next(self.join)

@card(type="optuna_study")
@step
def join(self, inputs):
    self.study = rebuild_study(inputs, objective="val_loss", direction="minimize")
    self.best_params = self.study.best_params
    self.best_value  = self.study.best_value
    self.next(self.end)
```

---

### Data Model

#### TrialResult — written by `@hyperparam`, read by `join_study`

```python
@dataclass
class TrialResult:
    """Stored as self.trial_result on each trial task."""
    trial_number: int
    params: dict[str, Any]        # populated from LiveTrial after ask() completes
    value: float | None           # None if step failed
    state: str                    # "complete" | "failed"
    duration_seconds: float
    start_datetime: datetime
```

#### TrialConfig — used in batch mode only

```python
@dataclass
class TrialConfig:
    """Pre-sampled trial for batch (non-adaptive) mode. foreach input."""
    trial_number: int
    params: dict[str, Any]        # all values are float | int | str | bool
    direction: str
```

#### LiveTrial — injected as `self.trial` in adaptive mode

```python
class LiveTrial:
    """
    Implements the optuna.trial.Trial suggest_* interface via HTTP calls to the
    coordinator service. Each suggest_* call is a POST to the coordinator;
    the coordinator's event loop serializes all concurrent calls so TPE sees
    a consistent study state.
    """
    number: int                   # trial_number assigned by coordinator on ask()
    params: dict[str, Any]        # populated lazily as suggest_* is called

    def suggest_float(self, name: str, low: float, high: float,
                      *, log: bool = False, **kwargs) -> float: ...
    def suggest_int(self, name: str, low: int, high: int, **kwargs) -> int: ...
    def suggest_categorical(self, name: str, choices: Sequence, **kwargs) -> Any: ...
    def set_user_attr(self, key: str, value: Any) -> None: ...

    # Internal — called by @hyperparam task_post_step:
    def _tell(self, value: float, state: str = "complete") -> None: ...
```

`LiveTrial.__init__` immediately calls `POST /ask` on construction, which:
1. Calls `study.ask()` on the coordinator — TPE samples using all completed results.
2. Returns `{"trial_number": N}` — params are not returned yet.

Subsequent `suggest_*` calls call `POST /suggest/{type}` which triggers
`trial.suggest_*` on the coordinator side (Optuna's define-by-run protocol),
registers the param in the study's internal distribution map, and returns the value.

#### ReplayTrial — injected as `self.trial` in batch mode

```python
class ReplayTrial:
    """
    Implements the optuna.trial.Trial suggest_* interface by replaying values
    from a pre-sampled TrialConfig. Range arguments are accepted but ignored.
    Raises KeyError with a descriptive message if a name is not in the config.
    """
    number: int
    params: dict[str, Any]

    def suggest_float(self, name: str, *args, **kwargs) -> float: ...
    def suggest_int(self, name: str, *args, **kwargs) -> int: ...
    def suggest_categorical(self, name: str, *args, **kwargs) -> Any: ...
```

#### Coordinator HTTP API

```
POST /ask
  → calls study.ask(), registers pending trial
  ← {"trial_number": int}

POST /suggest/float
  body: {"trial_number": int, "name": str, "low": float, "high": float, "log": bool}
  → calls pending_trial.suggest_float(name, low, high, log=log)
  ← {"value": float}

POST /suggest/int
  body: {"trial_number": int, "name": str, "low": int, "high": int}
  ← {"value": int}

POST /suggest/categorical
  body: {"trial_number": int, "name": str, "choices": list}
  ← {"value": any}

POST /tell
  body: {"trial_number": int, "value": float, "state": "complete"|"failed"}
  → calls study.tell(trial_number, value)
  → increments completed count; sets done_event if count == n_trials_int
  → journals study snapshot to S3 every journal_interval calls
  ← {"ok": true}

GET /health
  ← {"ready": true, "completed": int, "total": int}
```

The coordinator runs a single-worker uvicorn instance. FastAPI's async event loop serializes all request handling, so concurrent `ask()` and `suggest_*()` calls from parallel trial tasks are processed one at a time without explicit locks. Optuna's `InMemoryStorage` is never accessed from multiple threads simultaneously.

#### OptunaStudyCard layout

Unchanged from initial design. The card renders from `self.study` (a fully populated `optuna.Study`) at `join_study` time:

```
╔══════════════════════════════════════════════════════════════════╗
║  BEST TRIAL — #23 of 50                        ↓ minimize       ║
║  val_loss  0.1823                                                ║
║  lr: 0.00412   depth: 7   subsample: 0.82                       ║
╠═══════════════════════════════╦══════════════════════════════════╣
║  Optimization History         ║  Parameter Importance            ║
║  [Plotly — trial# vs value,   ║  [Plotly — fANOVA importances,  ║
║   best-so-far overlay]        ║   horizontal bar, sorted]        ║
╠═══════════════════════════════╩══════════════════════════════════╣
║  Parallel Coordinates  (colored: green=best → red=worst)         ║
║  [Plotly parcoords — one axis per param + objective axis]        ║
╠══════════════════════════════════════════════════════════════════╣
║  All Trials  (sortable, color-coded)                             ║
║  #  │ lr     │ depth │ subsample │ val_loss  │  dur              ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### Workflow / Sequence

```
start                             single task
│
│  1. Set self.coordinator_id = current.run_id
│  2. Set self.n_trials_int = int(self.n_trials)
│  3. self.next(self.run_coordinator, self.launch_trials)  ← parallel split
│
├──────────────────────────────────────────────────────────────────────
│ BRANCH A: run_coordinator                @batch(cpu=1, memory=512)
│
│  4.  [decorator] Create optuna.Study(direction, sampler) in-memory
│  5.  [decorator] Bind port 8765, get VPC private IP via socket.gethostbyname()
│  6.  [decorator] Write {"url": "http://10.x.x.x:8765"} to
│                  s3://{bucket}/metaflow-optuna/{coordinator_id}/endpoint
│  7.  [decorator] Start uvicorn in background daemon thread
│  8.  [decorator] Block on done_event (set when tell() count == n_trials_int)
│  9.  [decorator] On each tell(): journal study snapshot to
│                  s3://{bucket}/metaflow-optuna/{coordinator_id}/journal/{n}
│  10. [decorator] done_event fires; populate self.study
│  11. [step body] self.next(self.join_study)
│
├──────────────────────────────────────────────────────────────────────
│ BRANCH B: launch_trials → train (foreach) → join_trials
│
│  12. Poll s3://.../endpoint with 2s exponential backoff until present
│  13. Set self.coordinator_url = URL from S3
│  14. Set self.trial_ids = list(range(n_trials_int))
│  15. self.next(self.train, foreach="trial_ids")
│
│  For each trial task (runs in parallel):
│
│  16. [decorator pre_step] Read self.coordinator_url from parent namespace
│  17. [decorator pre_step] LiveTrial(coordinator_url) → POST /ask
│                           → coordinator: study.ask() → trial_number=N
│  18. [decorator pre_step] Inject self.trial = LiveTrial(number=N)
│  19. [decorator pre_step] Set optuna.logging.WARNING
│
│  [user step body executes]
│
│  20. self.trial.suggest_float("lr", ...) → POST /suggest/float
│                → coordinator: pending[N].suggest_float("lr", ...)
│                → TPE model (fitted on completed trials) returns value
│                ← value
│  21. ... (each suggest_* call is one HTTP round-trip to coordinator)
│  22. self.val_loss = run_training(...)
│
│  [decorator post_step]
│
│  23. Read self.val_loss (via objective="val_loss")
│  24. Build TrialResult(trial_number=N, params=self.trial.params,
│                        value=self.val_loss, state="complete", ...)
│  25. Store as self.trial_result
│  26. self.trial._tell(value)  → POST /tell
│                → coordinator: study.tell(N, value)
│                → coordinator: journal snapshot if due
│                → coordinator: done_event.set() if count == total
│
│  [if step body raised a Python exception:]
│  23b. Write TrialResult(state="failed"), DO NOT re-raise
│       task exits 0; join_trials proceeds with partial results
│  24b. self.trial._tell(None, state="failed") → POST /tell with FAIL
│
│  27. self.next(self.join_trials)
│
│  join_trials:
│  28. Collect self.trial_results from all inputs
│      (AttributeError on missing → synthetic failed record)
│  29. self.next(self.join_study)
│
├──────────────────────────────────────────────────────────────────────
│ JOIN: join_study  (waits for both branches)
│
│  30. Find the branch input that has self.study (from run_coordinator)
│  31. self.study = that input's study   ← fully populated optuna.Study
│  32. self.best_params, self.best_value populated
│  33. OptunaStudyCard renders from self.study (5 panels via Plotly)
│  34. self.next(self.end)
```

---

### Networking

#### VPC-internal (@batch / @kubernetes — recommended)

`run_coordinator` and `train` tasks run in the same AWS VPC or Kubernetes cluster. The coordinator discovers its private IP on startup:

```python
import socket
private_ip = socket.gethostbyname(socket.gethostname())
# e.g. "10.0.3.47" — reachable from all tasks in the same VPC subnet
```

**Security group / NetworkPolicy requirement**: the coordinator's task needs an inbound TCP rule on port 8765 from the trial tasks' security group. This is a VPC-internal rule — no public internet exposure.

For AWS Batch: both tasks should use the same VPC and subnet. The coordinator's ECS task security group needs to allow inbound on port 8765 from the trial tasks' security group.

For Kubernetes: a ClusterIP Service in front of the coordinator pod is the standard approach, but since the coordinator pod's IP is dynamic and registered in S3, a headless direct-IP connection is simpler for v1.

#### Hybrid: local coordinator + remote @batch trials (ngrok)

When the user runs the flow locally (`python flow.py run`) with `@batch` decorators on trial steps only, `run_coordinator` runs locally. The local machine is not reachable from @batch containers without a tunnel:

```python
# Automatically invoked by @optuna_coordinator when not running inside @batch:
from pyngrok import ngrok
tunnel = ngrok.connect(port, "http")
coordinator_url = tunnel.public_url   # "https://a3f2b1cd.ngrok.io"
# Written to S3 in place of the VPC private IP
```

ngrok is an optional dependency (`pip install metaflow-optuna[ngrok]`). The `@optuna_coordinator` decorator detects whether it is running inside an ECS/Batch container (via `AWS_BATCH_JOB_ID` env var) and selects VPC IP vs ngrok accordingly.

#### Fully local (no @batch)

Coordinator runs on `localhost`; trial tasks are local subprocesses. `await_coordinator` reads from a local temp file instead of S3 in this mode (detected via `METAFLOW_PROFILE` or absence of S3 datastore config).

---

### Artifact Storage and Crash Recovery

**Where state lives.** All durable state is in Metaflow's artifact store and a small S3 prefix. No external database.

```
s3://bucket/metaflow-optuna/{coordinator_id}/
  endpoint                    ← coordinator URL, written on startup
  journal/0.pkl               ← study snapshot after first journal_interval tells
  journal/1.pkl               ← snapshot after second journal_interval tells
  ...

s3://bucket/metaflow/XGBoostTuning/
  {run_id}/
    start/1/coordinator_id, n_trials_int
    run_coordinator/1/study               ← final populated study (Metaflow artifact)
    launch_trials/1/coordinator_url, trial_ids
    train/1/trial_result                  ← TrialResult per task
    train/2/trial_result
    ...
    join_trials/1/trial_results
    join_study/1/study, best_params, best_value
```

**Coordinator crash and recovery.** After every `journal_interval` `tell()` calls (default: 5), the coordinator pickles the full `optuna.Study` to `s3://.../journal/{n}.pkl`. If the coordinator crashes and is restarted (via `@retry(times=1)` on `run_coordinator`), the decorator:

1. Checks for journal files under the coordinator_id prefix.
2. If found, loads the most recent snapshot and restores `_study` and `_completed_count`.
3. Re-registers the (new) endpoint URL in S3.
4. Trial tasks that lost their connection retry `await_coordinator` and reconnect.
5. Trials that already called `tell()` before the crash are already in the restored study.
6. Trials that called `ask()` but not yet `tell()` have an unknown state — the coordinator re-issues them as new `ask()` requests when those tasks reconnect.

**Resume behavior.** `python flow.py resume {run_id}` re-runs only failed tasks. If `run_coordinator` crashed, resume re-runs it with journal recovery. If some `train` tasks succeeded and some failed, resume re-runs only the failed trial tasks. Successful trial tasks are not re-run; their `trial_result` artifacts are reused by `join_trials`.

**Trial task crash (soft).** Python exception in `train` step body: `@hyperparam` catches, writes `TrialResult(state="failed")`, calls `/tell` with `state="failed"`, task exits 0. `join_trials` proceeds with partial results. Failed count is shown in card warning banner.

**Trial task crash (hard).** SIGKILL / OOM kill: `task_exception` never runs, `/tell` is never called. The coordinator does not receive a tell for this trial — `done_event` waits. Metaflow marks the task failed; run stops. Recovery: `resume` re-runs the killed task, which calls `/ask` (getting a fresh trial assignment) and eventually calls `/tell`. The coordinator's `_completed_count` increments and eventually `done_event` fires.

---

### Key Design Decisions

| Decision | Options Considered | Chosen | Rationale |
|---|---|---|---|
| Coordinator placement | Separate infra (RDS, Redis); a long-running @batch task in a parallel branch | Parallel branch @batch task | No external infra; coordinator is part of the flow; lifecycle tied to the run |
| Coordinator storage | In-memory (lost on crash); SQLite file on EFS/NFS; S3 journaling | In-memory + S3 journal | No NFS dependency; journal gives crash recovery; in-memory is fastest for ask/tell |
| Rendezvous mechanism | Hardcoded port + service discovery; Metaflow metadata API; S3 key | S3 key under coordinator_id | S3 is already a Metaflow dependency; coordinator_id = run_id guarantees uniqueness |
| HTTP server | Flask (sync, simple); FastAPI+uvicorn (async); gRPC | FastAPI+uvicorn single-worker | Async event loop serializes study access without explicit locks; fast; modern |
| suggest_* protocol | Return all params on /ask; define-by-run via individual /suggest/* calls | Define-by-run /suggest/* | Mirrors Optuna's native protocol; no need to pre-declare search space on coordinator; existing objective functions port unchanged |
| Soft crash behavior | Re-raise (run fails); swallow (task exits 0, partial results survive) | Swallow + write failed TrialResult | HPO should not fail on one bad trial; partial results with 48/50 complete is useful |
| Batch mode | Remove entirely; keep as mode="batch" on @hyperparam | Keep as mode="batch" | Grid/random search users don't need coordinator complexity; QMCSampler + batch mode is a valid and simpler path |
| ngrok for hybrid | Always required; optional dep with auto-detection | Optional dep, auto-detected | Most production users run all steps on @batch (VPC native); ngrok only needed for local dev |

---

## Failure Modes

| Failure | Probability | Impact | Mitigation |
|---|---|---|---|
| Coordinator task crashes (soft, Python exception) | Low | High | `@retry(times=1)` on `run_coordinator`; journal recovery restores study state up to last snapshot |
| Coordinator task hard-crashes (OOM kill) | Low | High | Same as above; journal recovery; trial tasks retry `await_coordinator` for new endpoint URL |
| Coordinator crash with no journal yet (crash within first 5 tells) | Very Low | High | `@retry` restarts with empty study; re-run trial tasks via Metaflow `resume`; cost = redo up to journal_interval trials |
| Trial task soft crash (Python exception in train) | Medium | Low | `@hyperparam` catches, writes failed TrialResult, calls /tell(state=failed), exits 0. join_trials proceeds. |
| Trial task hard crash (SIGKILL/OOM) | Low | Medium | `/tell` never called; coordinator waits; run stops. Recover via `resume`. Add `@retry` on train step for transient OOMs. |
| Network unreachable: trial task cannot reach coordinator | Low | High | `LiveTrial.__init__` retries `/ask` with exponential backoff (10 attempts, max 60s). If coordinator never responds, raises `CoordinatorUnreachableError` and task fails cleanly. |
| `await_coordinator` timeout (coordinator takes >120s to start) | Low | Medium | Raises `CoordinatorNotReadyError` in `launch_trials`. Increase timeout via parameter; investigate coordinator startup logs. |
| All trials fail (0 complete results) | Very Low | High | `rebuild_study` / `join_study` raises `EmptyStudyError` with failed count; flow fails with clear message |
| Port conflict on coordinator node | Low | Low | `@optuna_coordinator` checks if port is in use and increments until free; registers actual port in S3 |
| Plotly not installed | Low | Medium | `OptunaStudyCard` renders best-trial summary + plain trial table only; warns user |
| Parameter importance fails (< 2 complete trials) | Medium | Low | Per-panel try/except; renders placeholder text in that section |
| ngrok tunnel expires (> 2h session for free tier) | Medium | High | Log warning at 90min; document ngrok pro for long runs; production use should be all-@batch (VPC native, no ngrok) |

---

## Success Metrics

- A 50-trial run produces **zero Optuna INFO log lines** in any task stdout/stderr.
- Trial N's hyperparameters are sampled by a TPE model fitted on N completed results: verified by asserting `len(study.trials) == N` at the coordinator when trial N's `/ask` is processed.
- Porting an existing single-machine Optuna objective function requires **≤ 5 line changes**: (1) add `@optuna_coordinator` step, (2) add `await_coordinator` in `launch_trials`, (3) add `@hyperparam`, (4) change `return value` to `self.metric = value`, (5) add `@card(type="optuna_study")` on `join_study`.
- The study card renders completely for a 50-trial study in **< 3 seconds** from card open.
- A coordinator crash with `@retry(times=1)` and `journal_interval=5` loses **≤ 5 trials' worth of study history**, verified by comparing recovered study trial count.
- A 200-trial study with 5 hyperparameters produces a card with all five sections rendering within the 3-second budget.
- `pip install metaflow-optuna` (without `[ngrok]`) on a machine with no ngrok token runs a fully all-@batch 50-trial flow end-to-end with no errors.

---

## Open Questions

1. ~~**Foreach inside a parallel branch**~~ **RESOLVED — supported.** Verified against Metaflow source (`metaflow/lint.py`, `metaflow/graph.py`). The only nested-foreach restriction is `check_nested_foreach` (lint.py:410-418), which checks `if any(graph[p].type == "foreach" for p in node.split_parents)` — it only fires when a foreach is nested inside another foreach, not inside a static split. The `parents()` function used by the cross-join lint check returns `("start",)` for both `run_coordinator` (linear, `split_parents=["start"]`) and `join_trials` (join, `split_parents=["start", "launch_trials"]`, `[:-1]` → `["start"]`), so the all_equal check passes. The analogous pattern — foreach inside a conditional split — is an existing tested flow: `test/core/graphs/foreach_in_switch.json`. No implementation blocker.

2. **Security group automation**: should `@optuna_coordinator` automatically add the inbound port rule to the ECS task's security group via boto3, or require users to configure it manually? Auto-config requires IAM `ec2:AuthorizeSecurityGroupIngress` permission which many users won't have. — Owner: TBD, Deadline: before v1 release.

3. **`@retry` interaction with LiveTrial `/ask`**: if a trial task fails and Metaflow retries it, the retry calls `/ask` again and gets a NEW trial number (different params from the original attempt). The original trial's ask is never told. The coordinator should detect stale pending asks (timeout-based) and reissue them. Timeout threshold TBD. — Owner: TBD, Deadline: before v1 release.

4. **Journal granularity**: `journal_interval=5` means up to 5 trials are re-run after a coordinator crash. Is this acceptable? Alternative: journal after every tell (higher S3 write cost, zero re-run cost). — Owner: TBD, Deadline: before v1 release.

5. **Fully local mode S3 substitute**: when running with local Metaflow datastore (no S3), the coordinator rendezvous can't use S3. Use a temp file or a local HTTP port for rendezvous instead. Detection via `METAFLOW_DATASTORE_SYSROOT_LOCAL` env var. — Owner: TBD, Deadline: before v1 release.

6. **`suggest_*` HTTP latency impact**: each `suggest_*` call is one HTTP round-trip to the coordinator (~1ms VPC-internal). For a search space with 10 parameters, that's ~10ms overhead per trial. Acceptable for trials > 1 minute; potentially significant for sub-second trials. Optimization: batch all `suggest_*` calls into a single `/ask_with_space` endpoint. Deferred to v2. — Owner: TBD, Deadline: can defer.

---

## Appendix

### Package structure

```
metaflow_optuna/
├── __init__.py                  # exports: optuna_coordinator, hyperparam,
│                                #          await_coordinator, rebuild_study,
│                                #          create_study_inputs (batch mode)
├── decorators.py                # HyperparamDecorator, CoordinatorDecorator
├── coordinator/
│   ├── __init__.py              # run_coordinator_service(), CoordinatorClient
│   ├── app.py                   # FastAPI app: /ask, /suggest/*, /tell, /health
│   └── journal.py               # S3 snapshot write/read helpers
├── trial.py                     # LiveTrial, ReplayTrial, TrialResult, TrialConfig
├── rendezvous.py                # await_coordinator(), register_endpoint()
├── study.py                     # rebuild_study(), create_study_inputs() (batch)
├── exceptions.py                # CoordinatorUnreachableError, CoordinatorNotReadyError,
│                                #   EmptyStudyError, HyperparamError
└── cards/
    ├── __init__.py
    ├── study_card.py            # OptunaStudyCard(MetaflowCard)
    └── study_card.html          # best-trial + chart containers + table template
```

### pyproject.toml

```toml
[project.dependencies]
metaflow  = ">=2.9"
optuna    = ">=3.0"
fastapi   = ">=0.100"
uvicorn   = ">=0.23"
httpx     = ">=0.24"           # async HTTP client for LiveTrial
plotly    = ">=5.0"
boto3     = ">=1.26"           # S3 rendezvous + journal

[project.optional-dependencies]
ngrok = ["pyngrok>=6.0"]

[project.entry-points."metaflow.plugins"]
optuna_coordinator = "metaflow_optuna.decorators:CoordinatorDecorator"
optuna_hyperparam  = "metaflow_optuna.decorators:HyperparamDecorator"

[project.entry-points."metaflow.cards"]
optuna_study = "metaflow_optuna.cards.study_card:OptunaStudyCard"
```

### Coordinator concurrency model

The coordinator runs a single-worker uvicorn instance. All HTTP requests are handled by FastAPI's asyncio event loop. Because uvicorn's default mode is single-process, single-thread async (ASGI), all request handlers run in the same thread and the same event loop. Optuna's `InMemoryStorage` operations (called from within async handlers) are therefore never executed concurrently — no explicit locking is needed. This is the key reason single-worker uvicorn was chosen over multi-worker or threaded alternatives.

The tradeoff: high request volume (many trial tasks calling `/suggest` simultaneously) could create a queue in the event loop. In practice, suggest calls are fast (< 1ms in Optuna's InMemoryStorage), so the queue drains quickly. For studies with > 100 concurrent trial tasks and > 10 parameters each, this may become a bottleneck. The v2 mitigation is a `/ask_with_space` batch endpoint that handles all suggest calls for one trial in a single request.

### Pre-sampling vs. coordinator quality comparison

| Scenario | Pre-sample (batch mode) | Coordinator (adaptive mode) |
|---|---|---|
| Random sampler, 50 trials | Identical | Identical |
| TPE, 50 trials | ~random (0 results at sample time) | Full TPE (N results at sample time for trial N) |
| TPE, 10 trials (small budget) | ~random | Full TPE — most important case |
| Grid search | Correct (use QMCSampler) | Unnecessary overhead |
| Trials < 30 seconds each | Fine | 10ms HTTP overhead is < 0.05% of trial time |
| Trials < 1 second each | Fine | Consider `/ask_with_space` batch endpoint (v2) |
