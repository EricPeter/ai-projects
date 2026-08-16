import os


def is_enabled(flag: str, default: bool = False) -> bool:
    """Read a feature flag from environment (ConfigMap-injected in k8s)."""
    val = os.getenv(flag, str(default)).lower()
    return val in ("true", "1", "yes")
