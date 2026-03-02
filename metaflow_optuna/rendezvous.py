"""
Thin wrapper around metaflow_session_service.rendezvous with a fixed
namespace of "metaflow-optuna".

S3 key format (preserved from v0.1):
    {prefix}/metaflow-optuna/{coordinator_id}/endpoint
    {prefix}/metaflow-optuna/{coordinator_id}/checkpoint
"""
from __future__ import annotations

import metaflow_session_service.rendezvous as _r

_NAMESPACE = "metaflow-optuna"


# Re-export for code that imports _is_remote directly
def _is_remote() -> bool:
    return _r._is_remote()


def register_endpoint(coordinator_id: str, url: str) -> None:
    _r.register_service(namespace=_NAMESPACE, service_id=coordinator_id, url=url)


def await_coordinator(
    coordinator_id: str,
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> str:
    from .exceptions import CoordinatorNotReadyError
    from metaflow_session_service.exceptions import ServiceNotReadyError

    try:
        return _r.await_service(
            service_id=coordinator_id,
            namespace=_NAMESPACE,
            timeout=timeout,
            poll_interval=poll_interval,
        )
    except ServiceNotReadyError as exc:
        raise CoordinatorNotReadyError(str(exc)) from exc


def save_checkpoint(coordinator_id: str, completed: int) -> None:
    _r.save_checkpoint(
        namespace=_NAMESPACE,
        service_id=coordinator_id,
        data={"completed": completed},
    )


def load_checkpoint(coordinator_id: str) -> int:
    data = _r.load_checkpoint(namespace=_NAMESPACE, service_id=coordinator_id)
    if data is None:
        return 0
    return data.get("completed", 0)
