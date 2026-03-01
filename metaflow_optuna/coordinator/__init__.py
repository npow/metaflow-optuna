"""
Coordinator service lifecycle management.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:
    pass


def _get_local_ip() -> str:
    """
    Returns 127.0.0.1 for local runs (all tasks on the same host),
    or the VPC private IP for remote runs (AWS Batch / Kubernetes).
    """
    from metaflow_optuna.rendezvous import _is_remote
    if not _is_remote():
        return "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _find_free_port(start: int = 8765) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting at {start}")


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
    import uvicorn

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

    # --- Discover address ---
    actual_port = _find_free_port(port or 8765)
    ip = _get_local_ip()
    endpoint_url = f"http://{ip}:{actual_port}"

    # --- Register rendezvous endpoint ---
    from metaflow_optuna.rendezvous import register_endpoint
    register_endpoint(coordinator_id, endpoint_url)

    # --- Start uvicorn in a daemon thread ---
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=actual_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Give uvicorn a moment to bind
    time.sleep(1.5)

    print(
        f"[metaflow-optuna] coordinator listening on {endpoint_url} "
        f"({n_trials} trials, {direction})"
    )

    # --- Block until all trials tell() in ---
    fired = _app_mod._done.wait(timeout=timeout)
    if not fired:
        print(
            f"[metaflow-optuna] coordinator timeout after {timeout}s "
            f"({_app_mod._completed}/{n_trials} completed)"
        )

    server.should_exit = True

    return _app_mod._study
