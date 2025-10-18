from __future__ import annotations

import os

import httpx
import litellm


def _configure_litellm_session(pool_size: int, retries: int, backoff: float) -> None:
    limits = httpx.Limits(max_connections=pool_size, max_keepalive_connections=pool_size)
    transport = httpx.HTTPTransport(retries=retries)
    client = httpx.Client(transport=transport, limits=limits, timeout=None)
    async_client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=retries), limits=limits, timeout=None)

    litellm.client_session = client
    litellm.aclient_session = async_client


def _configure_mlflow_session(pool_size: int) -> None:
    os.environ.setdefault("MLFLOW_HTTP_POOL_CONNECTIONS", str(pool_size))
    os.environ.setdefault("MLFLOW_HTTP_POOL_MAXSIZE", str(pool_size))
    try:
        from mlflow.utils import request_utils

        request_utils._cached_get_request_session.cache_clear()
    except Exception:
        # If MLflow isn't imported yet (or not installed), there's nothing to clear.
        pass


def set_pool_size_for_litellm_session(pool_size: int = 64, retries: int = 2, backoff: float = 0.1) -> None:
    """Expand LiteLLM and MLflow HTTP connection pools for heavily parallel runs."""
    _configure_litellm_session(pool_size=pool_size, retries=retries, backoff=backoff)
    _configure_mlflow_session(pool_size=pool_size)
