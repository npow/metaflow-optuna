class CoordinatorNotReadyError(Exception):
    """Raised when await_coordinator times out before the endpoint is registered."""


class CoordinatorUnreachableError(Exception):
    """Raised when a LiveTrial cannot connect to the coordinator after retries."""


class EmptyStudyError(Exception):
    """Raised by rebuild_study when zero trials completed successfully."""


class HyperparamError(Exception):
    """Raised when @hyperparam misconfiguration is detected at runtime."""


class TrialConfigSerializationError(Exception):
    """Raised when a TrialConfig contains non-serializable param values."""
