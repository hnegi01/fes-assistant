"""
backend/agent/mcp_client.py

Thin async HTTP client for the PySisense MCP server.

Responsibilities
----------------
- Talk to MCP server via Streamable HTTP JSON-RPC:
    - POST /mcp/ for JSON-RPC (initialize, tools/list, tools/call)
    - GET  /health for health checks
- Track and reuse Mcp-Session-Id returned by the server.
- Inject credentials into each tool call:
    - Non-migration tools: tenant_config -> (domain, token, ssl)
    - Migration tools: migration_config -> (source_* and target_*)
- Support streaming tool calls:
    - Server may respond to POST /mcp/ with SSE (text/event-stream).
    - Notifications are forwarded to backend.runtime.publish_progress.
    - Final JSON-RPC response is selected by request id.

Cancellation
------------
- A dedicated cancel endpoint is used (POST /mcp/cancel) with Mcp-Session-Id.
- On CancelledError during a streaming tools/call, we best-effort call /mcp/cancel
  before re-raising, so the server can stop the running migration.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("backend.agent.mcp_client")


def _setup_logger() -> None:
    """
    Configure file logging for this module.

    Notes
    -----
    Idempotent: avoids duplicate handlers.
    """
    log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    if any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        logger.info(
            "mcp_client logger already configured at level %s (env %s)",
            log_level_name,
            LOG_LEVEL_ENV_VAR,
        )
        return

    fh = TimedRotatingFileHandler(
        LOG_DIR / "mcp_client.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(log_level)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)

    logger.addHandler(fh)

    logger.info(
        "mcp_client logger initialized at level %s (env %s)",
        log_level_name,
        LOG_LEVEL_ENV_VAR,
    )


_setup_logger()

# -----------------------------------------------------------------------------
# Retry config for MCP calls
# -----------------------------------------------------------------------------
MAX_MCP_HTTP_RETRIES = int(os.getenv("MCP_HTTP_MAX_RETRIES", "1"))
MCP_HTTP_RETRY_BASE_DELAY = float(os.getenv("MCP_HTTP_RETRY_BASE_DELAY", "0.5"))

# -----------------------------------------------------------------------------
# Streaming config
# -----------------------------------------------------------------------------
_DEFAULT_STREAMING_TOOL_IDS = [
    "migration.migrate_all_groups",
    "migration.migrate_all_users",
    "migration.migrate_dashboards",
    "migration.migrate_all_dashboards",
    "migration.migrate_datamodels",
    "migration.migrate_all_datamodels",
]

_raw_stream_ids = os.getenv("MCP_STREAMING_TOOL_IDS", "").strip()
if _raw_stream_ids:
    STREAMING_TOOL_IDS = {t.strip() for t in _raw_stream_ids.split(",") if t.strip()}
else:
    STREAMING_TOOL_IDS = set(_DEFAULT_STREAMING_TOOL_IDS)


# -----------------------------------------------------------------------------
# Helpers: scrubbing + truncated logging
# -----------------------------------------------------------------------------
def _scrub_secrets(obj: Any) -> Any:
    """
    Recursively scrub likely secrets from dicts/lists (tokens, passwords, auth).
    """
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if (
                "token" in key_l
                or key_l in ("api-key", "apikey")
                or "authorization" in key_l
                or key_l in ("auth", "password", "passwd", "secret")
            ):
                cleaned[k] = "***REDACTED***"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]
    return obj


def _log_json(label: str, obj: Any) -> None:
    """Debug-log the FULL JSON (scrubbed). Truncation removed 2026-08-10 —
    disk is bounded by the 7-day rotation, not by cutting payloads."""
    safe_obj = _scrub_secrets(obj)
    try:
        text = json.dumps(safe_obj, indent=2, default=str, ensure_ascii=False)
    except Exception:
        text = str(safe_obj)
    logger.debug("%s:\n%s", label, text)


# -----------------------------------------------------------------------------
# SSE parsing helpers
# -----------------------------------------------------------------------------
SseMessageCallback = Callable[[Dict[str, Any]], Awaitable[None]]


async def _parse_sse_jsonrpc_messages(
    resp: httpx.Response,
    *,
    on_message: Optional[SseMessageCallback] = None,
) -> List[Dict[str, Any]]:
    """
    Parse an SSE (text/event-stream) response into JSON-RPC dict messages.
    """
    messages: List[Dict[str, Any]] = []
    data_lines: List[str] = []

    async for line in resp.aiter_lines():
        if line is None:
            continue

        line = line.strip("\r")

        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines = []
                try:
                    obj = json.loads(data_str)
                    if isinstance(obj, dict):
                        messages.append(obj)
                        if on_message is not None:
                            try:
                                await on_message(obj)
                            except Exception:
                                logger.debug("on_message callback failed; ignored.", exc_info=True)
                except Exception:
                    pass
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    if data_lines:
        try:
            obj = json.loads("\n".join(data_lines))
            if isinstance(obj, dict):
                messages.append(obj)
                if on_message is not None:
                    try:
                        await on_message(obj)
                    except Exception:
                        logger.debug("on_message callback failed; ignored.", exc_info=True)
        except Exception:
            pass

    return messages


def _pick_jsonrpc_response_for_id(messages: List[Dict[str, Any]], rid: int) -> Optional[Dict[str, Any]]:
    """
    Find the JSON-RPC response message that matches a request id.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("jsonrpc") != "2.0":
            continue
        if msg.get("id") == rid:
            return msg
    return None


# -----------------------------------------------------------------------------
# MCP HTTP client
# -----------------------------------------------------------------------------
class McpClient:
    """
    Minimal MCP client for the PySisense MCP HTTP server.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
        migration_config: Optional[Dict[str, Any]] = None,
    ):
        self._base_url = (base_url or os.getenv("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")).rstrip("/")
        self._timeout_seconds = float(os.getenv("PYSISENSE_MCP_HTTP_TIMEOUT", "60"))

        self._tenant_config: Dict[str, Any] = tenant_config or {}
        self._migration_config: Dict[str, Any] = migration_config or {}

        self._mcp_protocol_version = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25").strip()
        self._mcp_session_id: Optional[str] = None

        self._initialized = False
        self._init_lock = asyncio.Lock()

        self._id_lock = asyncio.Lock()
        self._next_id = 1

        # Shared HTTP clients (connection pooling)
        self._http: Optional[httpx.AsyncClient] = None
        self._http_sse: Optional[httpx.AsyncClient] = None

        logger.info("McpClient initialized base_url=%s timeout=%ss", self._base_url, self._timeout_seconds)
        logger.info("Streaming tool ids=%d", len(STREAMING_TOOL_IDS))

    # ------------------------------------------------------------------
    # Lifecycle: connect/close
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """
        Connect and initialize MCP lifecycle.
        """
        self._ensure_http_clients()
        await self._ensure_initialized()

    async def close(self) -> None:
        """
        Close the client.
        """
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None

        if self._http_sse is not None:
            with contextlib.suppress(Exception):
                await self._http_sse.aclose()
            self._http_sse = None

        logger.debug("McpClient.close() completed.")

    def _ensure_http_clients(self) -> None:
        """
        Ensure shared httpx clients exist for normal and long-lived SSE traffic.
        """
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds),
            )

        # Dedicated SSE client: timeout=None to avoid read timeouts for long-lived streams.
        if self._http_sse is None:
            self._http_sse = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=None,
            )

    # ------------------------------------------------------------------
    # JSON-RPC id helper
    # ------------------------------------------------------------------
    async def _new_id(self) -> int:
        async with self._id_lock:
            rid = self._next_id
            self._next_id += 1
            return rid

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    def _mcp_headers(self) -> Dict[str, str]:
        """
        Headers for JSON-RPC calls (POST /mcp/).
        """
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._mcp_protocol_version,
        }

    def _cancel_headers(self) -> Dict[str, str]:
        """
        Headers for cancel endpoint (POST /mcp/cancel).
        """
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._mcp_protocol_version,
        }

    def _attach_session_header(self, hdrs: Dict[str, str]) -> Dict[str, str]:
        """
        Ensure Mcp-Session-Id is attached if known.
        """
        out = dict(hdrs)
        if self._mcp_session_id and "Mcp-Session-Id" not in out:
            out["Mcp-Session-Id"] = self._mcp_session_id
        return out

    def _capture_session_id(self, headers: httpx.Headers) -> None:
        """
        Capture Mcp-Session-Id from server response, if present.
        """
        sid = headers.get("Mcp-Session-Id")
        if sid and sid != self._mcp_session_id:
            self._mcp_session_id = sid
            logger.info("Captured Mcp-Session-Id from server response.")

    # ------------------------------------------------------------------
    # Streaming detection
    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_canonicalize_tool_name(name: str) -> str:
        if "." in name:
            return name
        if name.startswith("migration_"):
            return name.replace("_", ".", 1)
        return name

    def _jsonrpc_is_streaming_call(self, json_body: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(json_body, dict):
            return False
        if json_body.get("jsonrpc") != "2.0":
            return False
        if json_body.get("method") != "tools/call":
            return False

        params = json_body.get("params") or {}
        if not isinstance(params, dict):
            return False

        name = params.get("name")
        if not isinstance(name, str) or not name:
            return False

        canonical = self._maybe_canonicalize_tool_name(name)
        return (name in STREAMING_TOOL_IDS) or (canonical in STREAMING_TOOL_IDS)

    # ------------------------------------------------------------------
    # Progress forwarding
    # ------------------------------------------------------------------
    async def _handle_sse_message(self, msg: Dict[str, Any]) -> None:
        """
        Forward JSON-RPC notifications (no id) to backend.runtime.publish_progress.
        """
        if not isinstance(msg, dict):
            return
        if msg.get("jsonrpc") != "2.0":
            return
        if "method" not in msg:
            return
        if msg.get("id") is not None:
            return

        method = msg.get("method")
        params = msg.get("params")

        try:
            from backend import runtime as runtime_mod
        except Exception as exc:
            logger.error("Failed to import backend.runtime; cannot publish progress: %s", exc, exc_info=True)
            return

        event = {"source": "mcp", "type": "notification", "method": method, "params": params}

        try:
            await runtime_mod.publish_progress(event)
        except Exception as exc:
            logger.error("publish_progress failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Cancellation: explicit endpoint
    # ------------------------------------------------------------------
    async def cancel_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Request cancellation of a running tool for a given MCP session.

        Parameters
        ----------
        session_id
            If omitted, uses the currently captured Mcp-Session-Id.

        Returns
        -------
        dict
            JSON response from the server (best-effort), or an error payload.
        """
        self._ensure_http_clients()
        assert self._http is not None

        sid = session_id or self._mcp_session_id
        if not sid:
            return {"ok": False, "error": "Missing Mcp-Session-Id"}

        headers = dict(self._cancel_headers())
        headers["Mcp-Session-Id"] = sid

        payload = {"session_id": sid}
        _log_json("HTTP POST /mcp/cancel headers", headers)
        _log_json("HTTP POST /mcp/cancel json", payload)

        try:
            resp = await self._http.post("/mcp/cancel", headers=headers, json=payload)
            self._capture_session_id(resp.headers)

            if not (200 <= resp.status_code < 300):
                body_preview = (resp.text or "")[:1000]
                logger.error("MCP cancel failed status=%s body=%s", resp.status_code, body_preview)
                return {"ok": False, "session_id": sid, "status_code": resp.status_code, "error": body_preview}

            try:
                data = resp.json()
            except ValueError:
                data = {"ok": True, "session_id": sid}

            _log_json("HTTP /mcp/cancel response", data)
            return data if isinstance(data, dict) else {"ok": True, "session_id": sid, "result": data}

        except Exception as exc:
            logger.exception("MCP cancel request failed session_id=%s err=%s", sid, exc)
            return {"ok": False, "session_id": sid, "error": str(exc)}

    # ------------------------------------------------------------------
    # Core HTTP request wrapper
    # ------------------------------------------------------------------
    async def _request_jsonrpc_post(self, json_body: Dict[str, Any]) -> Any:
        """
        POST /mcp/ using httpx streaming so we can handle both JSON and SSE responses.
        """
        assert self._http is not None
        assert self._http_sse is not None

        headers = self._attach_session_header(self._mcp_headers())

        _log_json("HTTP POST /mcp/ json", json_body)
        _log_json("HTTP POST /mcp/ headers", headers)

        is_streaming_call = self._jsonrpc_is_streaming_call(json_body)
        client = self._http_sse if is_streaming_call else self._http

        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_MCP_HTTP_RETRIES + 1):
            resp_ref: Optional[httpx.Response] = None

            try:
                async with client.stream("POST", "/mcp/", headers=headers, json=json_body) as resp:
                    resp_ref = resp
                    self._capture_session_id(resp.headers)

                    if not (200 <= resp.status_code < 300):
                        body_preview = (await resp.aread()).decode("utf-8", errors="ignore")[:1000]

                        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_MCP_HTTP_RETRIES:
                            logger.warning(
                                "MCP POST /mcp/ failed status=%s attempt=%d/%d; retryable. Body=%s",
                                resp.status_code,
                                attempt,
                                MAX_MCP_HTTP_RETRIES,
                                body_preview,
                            )
                            delay = MCP_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                            await asyncio.sleep(delay)
                            continue

                        logger.error("MCP POST /mcp/ failed status=%s body=%s", resp.status_code, body_preview)
                        resp.raise_for_status()

                    content_type = (resp.headers.get("Content-Type") or "").lower()

                    if "text/event-stream" in content_type:
                        msgs = await _parse_sse_jsonrpc_messages(resp, on_message=self._handle_sse_message)
                        _log_json("HTTP SSE messages", msgs)
                        return msgs

                    raw = await resp.aread()
                    if not raw:
                        _log_json("HTTP response JSON", None)
                        return None

                    try:
                        data = json.loads(raw.decode("utf-8"))
                    except Exception:
                        data = None

                    _log_json("HTTP response JSON", data)
                    return data

            except asyncio.CancelledError:
                # Cancellation path:
                # - Close the streaming response first (signals disconnect)
                # - Dispatch /mcp/cancel in a separate task so it actually reaches the server
                #   even though this task is being cancelled.
                logger.info("MCP POST /mcp/ cancelled.")

                if resp_ref is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(resp_ref.aclose())

                if is_streaming_call:
                    logger.info("Streaming tools/call cancelled; dispatching server-side cancel (/mcp/cancel).")

                    def _log_cancel_done(fut: "asyncio.Future[Any]") -> None:
                        try:
                            _ = fut.result()
                            logger.debug("cancel_session task completed successfully.")
                        except Exception as exc:
                            logger.debug("cancel_session task failed (ignored): %s", exc, exc_info=True)

                    with contextlib.suppress(Exception):
                        cancel_task = asyncio.create_task(self.cancel_session())
                        cancel_task.add_done_callback(_log_cancel_done)

                raise

            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "MCP POST /mcp/ request error attempt=%d/%d: %s",
                    attempt,
                    MAX_MCP_HTTP_RETRIES,
                    exc,
                )
                if attempt >= MAX_MCP_HTTP_RETRIES:
                    raise
                delay = MCP_HTTP_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

            finally:
                if resp_ref is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(resp_ref.aclose())

        raise RuntimeError(f"MCP POST /mcp/ failed without response; last_exc={last_exc}")

    async def _request_normal(self, method: str, path: str) -> Any:
        """
        Normal (non-JSON-RPC) request, used for endpoints like GET /health.
        """
        assert self._http is not None
        resp = await self._http.request(method=method, url=path)
        self._capture_session_id(resp.headers)

        if not (200 <= resp.status_code < 300):
            body_preview = (resp.text or "")[:1000]
            logger.error("%s %s failed status=%s body=%s", method, path, resp.status_code, body_preview)
            resp.raise_for_status()

        try:
            return resp.json()
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------
    async def _rpc_call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Send a JSON-RPC request (with id) and return its 'result' or raise on 'error'.
        """
        rid = await self._new_id()

        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params

        data = await self._request_jsonrpc_post(msg)

        if isinstance(data, list):
            resp_msg = _pick_jsonrpc_response_for_id(data, rid)
            if not isinstance(resp_msg, dict):
                raise RuntimeError("Missing JSON-RPC response message in SSE stream.")
            if resp_msg.get("error") is not None:
                raise RuntimeError(f"JSON-RPC error: {resp_msg['error']}")
            return resp_msg.get("result")

        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid JSON-RPC response type: {type(data).__name__}")

        if data.get("error") is not None:
            raise RuntimeError(f"JSON-RPC error: {data['error']}")

        return data.get("result")

    async def _rpc_notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Send a JSON-RPC notification (no id).
        """
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._request_jsonrpc_post(msg)

    async def _ensure_initialized(self) -> None:
        """
        Ensure MCP lifecycle is complete: initialize -> notifications/initialized.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing MCP session via JSON-RPC initialize...")

            init_params = {
                "protocolVersion": self._mcp_protocol_version,
                "capabilities": {
                    "tools": {"listChanged": True},
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {"name": "FES Assistant MCP Client", "version": "1.0.0"},
            }

            await self._rpc_call("initialize", init_params)
            await self._rpc_notify("notifications/initialized")

            self._initialized = True
            logger.info("MCP session initialized.")

    # ------------------------------------------------------------------
    # Credential injection
    # ------------------------------------------------------------------
    def _with_tenant(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._tenant_config:
            return dict(arguments)

        merged = dict(arguments)
        for key in ("domain", "token", "ssl"):
            if key in self._tenant_config and key not in merged:
                merged[key] = self._tenant_config[key]
        return merged

    def _with_migration(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._migration_config:
            return dict(arguments)

        merged = dict(arguments)
        src = self._migration_config.get("source", {}) or {}
        tgt = self._migration_config.get("target", {}) or {}

        for out_key, in_key in (("source_domain", "domain"), ("source_token", "token"), ("source_ssl", "ssl")):
            if in_key in src and out_key not in merged:
                merged[out_key] = src[in_key]

        for out_key, in_key in (("target_domain", "domain"), ("target_token", "token"), ("target_ssl", "ssl")):
            if in_key in tgt and out_key not in merged:
                merged[out_key] = tgt[in_key]

        return merged

    def _inject_credentials(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not tool_id:
            return dict(arguments)

        canonical = self._maybe_canonicalize_tool_name(tool_id)
        module = canonical.split(".", 1)[0] if "." in canonical else canonical.split("_", 1)[0]

        if module == "migration":
            return self._with_migration(arguments)
        return self._with_tenant(arguments)

    # ------------------------------------------------------------------
    # Public API (used by LLM agent)
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        """
        Call GET /health.
        """
        self._ensure_http_clients()
        data = await self._request_normal("GET", "/health")
        return data if isinstance(data, dict) else {}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Call JSON-RPC tools/list and return the list of tools as provided by server.
        """
        self._ensure_http_clients()
        await self._ensure_initialized()

        result = await self._rpc_call("tools/list", {})

        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            tools = result["tools"]
            return tools if isinstance(tools, list) else []

        if isinstance(result, list):
            return result

        logger.warning("tools/list returned unexpected payload; returning [].")
        return []

    async def invoke_tool(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call JSON-RPC tools/call and parse the server's CallToolResult payload.
        """
        self._ensure_http_clients()
        await self._ensure_initialized()

        args_with_creds = self._inject_credentials(tool_id, arguments or {})

        logger.info("Calling MCP JSON-RPC tools/call name=%s", tool_id)
        _log_json("tools/call arguments", args_with_creds)

        result = await self._rpc_call("tools/call", {"name": tool_id, "arguments": args_with_creds})

        if isinstance(result, dict) and isinstance(result.get("content"), list):
            content = result.get("content") or []
            if content and isinstance(content[0], dict) and content[0].get("type") == "text":
                text = content[0].get("text")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed
                        return {"tool_id": tool_id, "result": parsed}
                    except Exception:
                        return {"tool_id": tool_id, "result": text}

        if isinstance(result, dict):
            return {"tool_id": tool_id, "result": result}

        return {"tool_id": tool_id, "result": result}
