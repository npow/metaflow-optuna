"""
Lightweight rendezvous: the coordinator writes its URL to a well-known
location; trial tasks poll until it appears.

Local mode  → /tmp/mf-optuna-{coordinator_id}.json
Remote mode → s3://{bucket}/metaflow-optuna/{coordinator_id}/endpoint
              (detected via AWS_BATCH_JOB_ID or METAFLOW_DATASTORE_SYSROOT_S3)
"""
from __future__ import annotations

import json
import os
import time

_LOCAL_TMP = "/tmp/mf-optuna-{}.json"


def _is_remote() -> bool:
    return bool(
        os.environ.get("AWS_BATCH_JOB_ID")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
    )


def _s3_key(coordinator_id: str) -> tuple[str, str]:
    """Returns (bucket, key) for the rendezvous object."""
    root = os.environ.get("METAFLOW_DATASTORE_SYSROOT_S3", "")
    if root.startswith("s3://"):
        root = root[5:]
    bucket, _, prefix = root.partition("/")
    key = f"{prefix}/metaflow-optuna/{coordinator_id}/endpoint".lstrip("/")
    return bucket, key


def register_endpoint(coordinator_id: str, url: str) -> None:
    payload = json.dumps({"url": url})

    if _is_remote():
        import boto3
        bucket, key = _s3_key(coordinator_id)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=payload)
        print(f"[metaflow-optuna] registered endpoint in s3://{bucket}/{key}")
    else:
        path = _LOCAL_TMP.format(coordinator_id)
        with open(path, "w") as f:
            f.write(payload)
        print(f"[metaflow-optuna] registered endpoint at {path}")


def await_coordinator(
    coordinator_id: str,
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> str:
    """
    Poll until the coordinator registers its URL, then return it.
    Raises CoordinatorNotReadyError if timeout is exceeded.
    """
    from .exceptions import CoordinatorNotReadyError

    deadline = time.monotonic() + timeout
    attempt = 0

    while time.monotonic() < deadline:
        try:
            url = _read_endpoint(coordinator_id)
            if url:
                return url
        except Exception:
            pass
        wait = min(poll_interval * (1.5 ** attempt), 15)
        time.sleep(wait)
        attempt += 1

    raise CoordinatorNotReadyError(
        f"Coordinator for id={coordinator_id!r} did not register within {timeout}s. "
        "Check run_coordinator step logs."
    )


def save_checkpoint(coordinator_id: str, completed: int) -> None:
    """Persist completed trial count so a restarted coordinator can resume."""
    payload = json.dumps({"completed": completed})
    if _is_remote():
        import boto3
        bucket, key = _s3_key(coordinator_id)
        key = key.replace("/endpoint", "/checkpoint")
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=payload)
    else:
        path = _LOCAL_TMP.format(coordinator_id + "-checkpoint")
        with open(path, "w") as f:
            f.write(payload)


def load_checkpoint(coordinator_id: str) -> int:
    """Return previously persisted completed count, or 0 if none exists."""
    try:
        if _is_remote():
            import boto3
            bucket, key = _s3_key(coordinator_id)
            key = key.replace("/endpoint", "/checkpoint")
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())["completed"]
        else:
            path = _LOCAL_TMP.format(coordinator_id + "-checkpoint")
            if not os.path.exists(path):
                return 0
            with open(path) as f:
                return json.loads(f.read())["completed"]
    except Exception:
        return 0


def _read_endpoint(coordinator_id: str) -> str | None:
    if _is_remote():
        import boto3
        bucket, key = _s3_key(coordinator_id)
        try:
            obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())["url"]
        except Exception:
            return None
    else:
        path = _LOCAL_TMP.format(coordinator_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.loads(f.read())["url"]
