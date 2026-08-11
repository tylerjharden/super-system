"""Async Adapt-1 client with state-aware retry behavior."""

from __future__ import annotations

import asyncio
import time
import uuid
from types import TracebackType
from typing import Any, Self

import httpx

from adapt1_fle.models import ApiExchange, JsonObject

PERMANENT_STATUS_CODES = {400, 401, 403, 404, 409, 413, 422}


class AdaptClientError(RuntimeError):
    """Base error carrying the redacted exchange."""

    def __init__(self, message: str, exchange: ApiExchange):
        super().__init__(message)
        self.exchange = exchange


class PermanentAdaptError(AdaptClientError):
    """Request cannot succeed without caller correction."""


class TransientAdaptError(AdaptClientError):
    """A safe read exhausted bounded retries."""


class AmbiguousWriteError(AdaptClientError):
    """A mutating request may have reached persistent state."""


class AdaptClient:
    """Thin typed gateway over the documented Adapt-1 HTTP surface."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float = 30.0,
        read_retry_attempts: int = 4,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.read_retry_attempts = read_retry_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> tuple[JsonObject, ApiExchange]:
        return await self._request("GET", "/healthz", authenticated=False, safe_to_retry=True)

    async def version(self) -> tuple[JsonObject, ApiExchange]:
        return await self._request("GET", "/version", authenticated=False, safe_to_retry=True)

    async def list_domains(self) -> tuple[JsonObject, ApiExchange]:
        return await self._request("GET", "/api/v1/domains", safe_to_retry=True)

    async def get_domain(self, domain_id: str) -> tuple[JsonObject, ApiExchange]:
        return await self._request("GET", f"/api/v1/domains/{domain_id}", safe_to_retry=True)

    async def create_domain(self, payload: JsonObject) -> tuple[JsonObject, ApiExchange]:
        return await self._request("POST", "/api/v1/domains", payload=payload, safe_to_retry=False)

    async def delete_domain(self, domain_id: str) -> tuple[JsonObject, ApiExchange]:
        return await self._request("DELETE", f"/api/v1/domains/{domain_id}", safe_to_retry=False)

    async def reset_policy(self, domain_id: str) -> tuple[JsonObject, ApiExchange]:
        return await self._request(
            "POST",
            f"/api/v1/domains/{domain_id}/policy/reset",
            payload={"session_id": "ignored"},
            safe_to_retry=False,
        )

    async def query_domain(
        self,
        domain_id: str,
        *,
        question: str,
        top_k: int,
        metadata_filter: JsonObject | None = None,
        return_fields: list[str] | None = None,
        frozen: bool = False,
    ) -> tuple[JsonObject, ApiExchange]:
        payload: JsonObject = {
            "session_id": "ignored",
            "question": question,
            "top_k": top_k,
            "metadata_filter": metadata_filter,
            "return_fields": return_fields,
            "allow_exploration": not frozen,
            "update_memory_state": not frozen,
        }
        return await self._request(
            "POST",
            f"/api/v1/domains/{domain_id}/query",
            payload=payload,
            safe_to_retry=frozen,
        )

    async def explain_domain(
        self,
        domain_id: str,
        *,
        question: str | None = None,
        decision_id: str | None = None,
        top_k: int = 10,
    ) -> tuple[JsonObject, ApiExchange]:
        payload: JsonObject = {
            "session_id": "ignored",
            "top_k": top_k,
            "allow_exploration": False,
            "update_memory_state": False,
        }
        if question is not None:
            payload["question"] = question
        if decision_id is not None:
            payload["decision_id"] = decision_id
        return await self._request(
            "POST",
            f"/api/v1/domains/{domain_id}/explain",
            payload=payload,
            safe_to_retry=True,
        )

    async def submit_feedback(
        self, domain_id: str, payload: JsonObject
    ) -> tuple[JsonObject, ApiExchange]:
        return await self._request(
            "POST",
            f"/api/v1/domains/{domain_id}/feedback",
            payload=payload,
            safe_to_retry=False,
        )

    async def store_domain_event(
        self, domain_id: str, payload: JsonObject
    ) -> tuple[JsonObject, ApiExchange]:
        return await self._request(
            "POST",
            f"/api/v1/domains/{domain_id}/events",
            payload=payload,
            safe_to_retry=False,
        )

    async def query_memory(
        self,
        *,
        message: str,
        top_k: int,
        metadata_filter: JsonObject | None = None,
        frozen: bool = False,
    ) -> tuple[JsonObject, ApiExchange]:
        payload: JsonObject = {
            "session_id": "ignored",
            "user_message": message,
            "top_k": top_k,
            "include_reasoning": True,
            "metadata_filter": metadata_filter,
            "update_memory_state": not frozen,
        }
        return await self._request(
            "POST",
            "/api/v1/memory/query",
            payload=payload,
            safe_to_retry=frozen,
        )

    async def store_memory(
        self, *, message: str, response: str, context: JsonObject
    ) -> tuple[JsonObject, ApiExchange]:
        payload: JsonObject = {
            "session_id": "ignored",
            "user_message": message,
            "ai_message": response,
            "context": context,
        }
        return await self._request(
            "POST", "/api/v1/memory/store", payload=payload, safe_to_retry=False
        )

    async def memory_state(self) -> tuple[JsonObject, ApiExchange]:
        return await self._request(
            "POST",
            "/api/v1/memory/state",
            payload={"session_id": "ignored"},
            safe_to_retry=True,
        )

    async def _request(
        self,
        method: str,
        route: str,
        *,
        payload: JsonObject | None = None,
        authenticated: bool = True,
        safe_to_retry: bool,
    ) -> tuple[JsonObject, ApiExchange]:
        request_id = str(uuid.uuid4())
        attempts = self.read_retry_attempts if safe_to_retry else 1
        last_exchange: ApiExchange | None = None
        request_started = time.monotonic()
        attempt_records: list[JsonObject] = []

        for attempt in range(attempts):
            started = time.monotonic()
            headers = {"X-Request-ID": request_id}
            if authenticated:
                if not self.api_key:
                    exchange = self._exchange(
                        request_id=request_id,
                        method=method,
                        route=route,
                        payload=payload,
                        started=request_started,
                        attempt_count=attempt + 1,
                        attempts=attempt_records,
                        error="Adapt-1 authentication is not configured",
                    )
                    raise PermanentAdaptError(exchange.error or "missing API key", exchange)
                headers["Authorization"] = f"Bearer {self.api_key}"

            try:
                response = await self._client.request(
                    method,
                    route,
                    json=payload,
                    headers=headers,
                )
            except httpx.RequestError as error:
                attempt_records.append(
                    {
                        "attempt": attempt + 1,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                exchange = self._exchange(
                    request_id=request_id,
                    method=method,
                    route=route,
                    payload=payload,
                    started=request_started,
                    attempt_count=attempt + 1,
                    attempts=attempt_records,
                    ambiguous=not safe_to_retry,
                    error=f"{type(error).__name__}: {error}",
                )
                last_exchange = exchange
                if not safe_to_retry:
                    raise AmbiguousWriteError(
                        "Adapt-1 write outcome is ambiguous; inspect state before replay",
                        exchange,
                    ) from error
                if attempt + 1 < attempts:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                raise TransientAdaptError("Adapt-1 read retries exhausted", exchange) from error

            body = _response_object(response)
            attempt_records.append(
                {
                    "attempt": attempt + 1,
                    "elapsed_seconds": time.monotonic() - started,
                    "status_code": response.status_code,
                    "response": body,
                }
            )
            exchange = self._exchange(
                request_id=request_id,
                method=method,
                route=route,
                payload=payload,
                started=request_started,
                attempt_count=attempt + 1,
                attempts=attempt_records,
                status_code=response.status_code,
                response=body,
            )
            last_exchange = exchange

            if 200 <= response.status_code < 300:
                return body, exchange

            detail = _error_detail(body, response.text)
            if _is_transient_status(response.status_code):
                if not safe_to_retry:
                    ambiguous = exchange.model_copy(update={"ambiguous": True, "error": detail})
                    raise AmbiguousWriteError(
                        "Adapt-1 write received a transient response; reconcile before replay",
                        ambiguous,
                    )
                if attempt + 1 < attempts:
                    await asyncio.sleep(_retry_delay(response, attempt))
                    continue
                failed = exchange.model_copy(update={"error": detail})
                raise TransientAdaptError("Adapt-1 read retries exhausted", failed)

            failed = exchange.model_copy(update={"error": detail})
            if response.status_code in PERMANENT_STATUS_CODES:
                raise PermanentAdaptError(
                    f"Adapt-1 rejected the request ({response.status_code}): {detail}",
                    failed,
                )
            raise AdaptClientError(
                f"Unexpected Adapt-1 response ({response.status_code}): {detail}",
                failed,
            )

        if last_exchange is None:
            raise RuntimeError("Adapt-1 request loop did not execute")
        raise TransientAdaptError("Adapt-1 read retries exhausted", last_exchange)

    @staticmethod
    def _exchange(
        *,
        request_id: str,
        method: str,
        route: str,
        payload: JsonObject | None,
        started: float,
        attempt_count: int,
        attempts: list[JsonObject],
        status_code: int | None = None,
        response: JsonObject | None = None,
        ambiguous: bool = False,
        error: str | None = None,
    ) -> ApiExchange:
        return ApiExchange(
            request_id=request_id,
            method=method,
            route=route,
            request=payload,
            status_code=status_code,
            response=response,
            elapsed_seconds=time.monotonic() - started,
            attempt_count=attempt_count,
            attempts=attempts,
            ambiguous=ambiguous,
            error=error,
        )


def _response_object(response: httpx.Response) -> JsonObject:
    try:
        value: Any = response.json()
    except ValueError:
        return {"raw_text": response.text}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _error_detail(body: JsonObject, raw_text: str) -> str:
    detail = body.get("detail") or body.get("message") or body.get("error")
    if detail is not None:
        return str(detail)
    return raw_text[:500] or "empty response"


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return _backoff_seconds(attempt)


def _backoff_seconds(attempt: int) -> float:
    delay = 0.5 * float(2**attempt)
    return min(delay, 8.0)


def _is_transient_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599
