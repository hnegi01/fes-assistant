# mcp_server/server.py
#
# Run:
#   uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
#
# Manual JSON-RPC + Manual SSE (no StreamableHTTPSessionManager):
# - GET  /mcp  => SSE "subscribe" endpoint (needed for Claude Desktop / mcp-remote probing)
# - POST /mcp  => JSON-RPC:
#     - initialize, tools/list, non-stream tools/call => normal JSON responses
#     - streaming tools/call (only tools_core.STREAMING_TOOL_IDS) => SSE response on the POST connection:
#         emits notifications/message frames for progress and ends with final JSON-RPC result frame
#
# Cancellation:
# - POST /mcp/cancel => explicitly request cancellation for a session via Mcp-Session-Id header
#
# Health:
# - GET /health

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# -----------------------------------------------------------------------------
# Optional .env loading
# -----------------------------------------------------------------------------
if load_dotenv is not None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

import mcp.types as types

from mcp_server import tools_core

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("mcp_http")


def _setup_logger() -> None:
    """
    Configure a rotating file logger.

    Notes
    -----
    - This function is idempotent and avoids duplicating handlers.
    """
    log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.info(
            "mcp_http logger already configured at level %s (env %s)",
            log_level_name,
            LOG_LEVEL_ENV_VAR,
        )
        return

    fh = RotatingFileHandler(
        LOG_DIR / "server.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(
        "mcp_http logger initialized at level %s (env %s)",
        log_level_name,
        LOG_LEVEL_ENV_VAR,
    )


_setup_logger()


# -----------------------------------------------------------------------------
# Context: current session id (for correlation)
# -----------------------------------------------------------------------------
_CURRENT_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mcp_current_session_id", default=None
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _scrub_secrets(obj: Any) -> Any:
    """
    Redact likely secret fields from nested dict/list structures before logging.
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


async def _read_body(receive: Receive) -> bytes:
    """
    Read the full HTTP request body from ASGI receive().
    """
    body = b""
    more_body = True

    while more_body:
        message = await receive()
        if message.get("type") != "http.request":
            continue

        body += message.get("body", b"") or b""
        more_body = bool(message.get("more_body", False))

    return body


def _get_header(scope: Scope, name: str) -> Optional[str]:
    """
    Get a header value from ASGI scope by case-insensitive name.
    """
    target = name.lower().encode("utf-8")
    for k, v in scope.get("headers") or []:
        if k.lower() == target:
            try:
                return v.decode("utf-8")
            except Exception:
                return None
    return None


def _accept_has_both(scope: Scope) -> bool:
    """
    MCP JSON-RPC POST requires client Accept header to include both:
      - application/json
      - text/event-stream
    """
    accept = (_get_header(scope, "accept") or "").lower()
    return ("application/json" in accept) and ("text/event-stream" in accept)


def _jsonrpc_error(rid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _tool_to_wire(t: types.Tool) -> Dict[str, Any]:
    """
    Convert MCP Tool object into a dict with only the fields clients expect.
    Ensure enums/optionals serialize cleanly for strict clients (Inspector).
    """
    if hasattr(t, "model_dump"):
        d = t.model_dump(mode="json", exclude_none=True)
    else:
        d = t.dict(exclude_none=True)

    return {
        "name": d.get("name"),
        "description": d.get("description") or "",
        "inputSchema": d.get("inputSchema") or {"type": "object", "properties": {}},
    }


def _result_to_dict(r: Any) -> Dict[str, Any]:
    """
    Serialize pydantic models without null optionals (e.g., annotations=null),
    which breaks strict MCP clients like the Inspector.
    """
    if hasattr(r, "model_dump"):
        return r.model_dump(mode="json", exclude_none=True)
    if hasattr(r, "dict"):
        return r.dict(exclude_none=True)
    return dict(r)


def _make_initialize_result(protocol_version: str) -> Dict[str, Any]:
    """
    Build initialize response payload.

    Notes
    -----
    Some clients expect specific capability keys to exist.
    """
    return {
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": {"listChanged": True},
            "roots": {"listChanged": True},
            "sampling": {},
        },
        "serverInfo": {"name": "pysisense-tools", "version": "1.0.0"},
    }


def _get_or_create_session_id(scope: Scope) -> str:
    """
    Return client session id if provided, otherwise generate a new one.
    """
    session_id = _get_header(scope, "Mcp-Session-Id")
    return session_id or uuid.uuid4().hex


# -----------------------------------------------------------------------------
# Tool naming (underscore aliases for clients like Claude)
# -----------------------------------------------------------------------------
MCP_TOOL_NAME_MODE = os.getenv("MCP_TOOL_NAME_MODE", "claude").strip().lower()


def _to_claude_tool_name(tool_id: str) -> str:
    return tool_id.replace(".", "_")


def _build_tool_name_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    canonical_to_alias: Dict[str, str] = {}
    alias_to_canonical: Dict[str, str] = {}

    tool_ids = list(getattr(tools_core, "TOOLS_BY_ID", {}).keys())
    for tid in tool_ids:
        alias = _to_claude_tool_name(tid)
        canonical_to_alias[tid] = alias

        if alias not in alias_to_canonical:
            alias_to_canonical[alias] = tid
        elif alias_to_canonical[alias] != tid:
            logger.warning(
                "Claude tool alias collision: alias=%s maps to both %s and %s. Keeping first mapping=%s",
                alias,
                alias_to_canonical[alias],
                tid,
                alias_to_canonical[alias],
            )

    return canonical_to_alias, alias_to_canonical


_CANONICAL_TO_ALIAS, _ALIAS_TO_CANONICAL = _build_tool_name_maps()


def _canonicalize_tool_name(name: str) -> str:
    if not name:
        return name
    if name in getattr(tools_core, "TOOLS_BY_ID", {}):
        return name
    mapped = _ALIAS_TO_CANONICAL.get(name)
    return mapped or name


def _exposed_tool_name(canonical_tool_id: str) -> str:
    if MCP_TOOL_NAME_MODE == "claude":
        return _CANONICAL_TO_ALIAS.get(canonical_tool_id, canonical_tool_id)
    return canonical_tool_id


def _is_streaming_tools_call(obj: Any) -> bool:
    """
    Return True if the JSON-RPC request is a streaming tools/call.
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("method") != "tools/call":
        return False

    params = obj.get("params") or {}
    name = str(params.get("name") or "")
    canonical = _canonicalize_tool_name(name)
    return canonical in getattr(tools_core, "STREAMING_TOOL_IDS", set())


# -----------------------------------------------------------------------------
# Tool schema augmentation (tenant props)
# -----------------------------------------------------------------------------
TENANT_PROPS = {
    "domain": {"type": "string", "description": "Sisense base URL (e.g. https://acme.sisense.com)"},
    "token": {"type": "string", "description": "Sisense API token"},
    "ssl": {"type": "boolean", "description": "Verify SSL certificates (default true)"},
}

SOURCE_TENANT_PROPS = {
    "source_domain": {"type": "string", "description": "Source Sisense base URL"},
    "source_token": {"type": "string", "description": "Source Sisense API token"},
    "source_ssl": {"type": "boolean", "description": "Verify SSL certs for source (default true)"},
}

TARGET_TENANT_PROPS = {
    "target_domain": {"type": "string", "description": "Target Sisense base URL"},
    "target_token": {"type": "string", "description": "Target Sisense API token"},
    "target_ssl": {"type": "boolean", "description": "Verify SSL certs for target (default true)"},
}


def _augment_input_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject tenant configuration properties into each tool's input schema.

    Notes
    -----
    - Migration tools get source_* and target_* properties.
    - Other tools get domain/token/ssl properties.
    - "emit" is removed because MCP clients cannot pass callbacks.
    """
    schema = copy.deepcopy(row.get("parameters") or {})
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "required": []}

    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])

    module = row.get("module")

    if module == "migration":
        for k, v in SOURCE_TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for k, v in TARGET_TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for req in ("source_domain", "source_token", "target_domain", "target_token"):
            if req not in schema["required"]:
                schema["required"].append(req)
    else:
        for k, v in TENANT_PROPS.items():
            schema["properties"].setdefault(k, v)
        for req in ("domain", "token"):
            if req not in schema["required"]:
                schema["required"].append(req)

    schema["properties"].pop("emit", None)
    if "emit" in schema.get("required", []):
        schema["required"].remove("emit")

    return schema


# -----------------------------------------------------------------------------
# MCP-ish: tools/list (manual, but uses your registry)
# -----------------------------------------------------------------------------
async def _list_tools_payload() -> Dict[str, Any]:
    rows = tools_core.list_tools()
    out: List[Dict[str, Any]] = []

    for row in rows:
        tool_id = row.get("tool_id")
        if not tool_id:
            continue

        t = types.Tool(
            name=_exposed_tool_name(tool_id),
            description=row.get("description") or "",
            inputSchema=_augment_input_schema(row),
        )
        out.append(_tool_to_wire(t))

    logger.info("MCP tools/list returning %d tools", len(out))
    return {"tools": out}


# -----------------------------------------------------------------------------
# MCP-ish: non-streaming tool call (manual)
# -----------------------------------------------------------------------------
async def _call_tool_non_streaming(tool_name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
    canonical_name = _canonicalize_tool_name(tool_name)
    logger.info("MCP tools/call (JSON) tool_id=%s", canonical_name)

    payload = await tools_core.invoke_tool_async(tool_id=canonical_name, arguments=arguments or {})
    ok = bool(payload.get("ok"))
    text = json.dumps(payload, indent=2, default=str)

    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        isError=not ok,
    )


# -----------------------------------------------------------------------------
# SSE helpers
# -----------------------------------------------------------------------------
def _sse_frame(obj: Dict[str, Any], event: str = "message") -> str:
    data = json.dumps(obj, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _notification_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "info", "data": payload},
    }


async def _sse_subscribe(session_id: str) -> AsyncIterator[str]:
    """
    GET /mcp SSE endpoint used by clients to validate stream transport and keep a stream open.
    """
    yield f": mcp subscribe ok (session {session_id})\n\n"
    while True:
        await asyncio.sleep(15)
        yield ": keepalive\n\n"


async def _send_sse_stream(
    send: Send,
    body_iter: AsyncIterator[str],
    *,
    status_code: int,
    headers: Dict[str, str],
) -> None:
    """
    Manual ASGI SSE sender.
    """
    raw_headers: List[Tuple[bytes, bytes]] = []
    for k, v in headers.items():
        raw_headers.append((k.encode("utf-8"), v.encode("utf-8")))

    # Ensure Content-Type is present.
    if not any(k.lower() == b"content-type" for k, _ in raw_headers):
        raw_headers.append((b"content-type", b"text/event-stream"))

    await send({"type": "http.response.start", "status": status_code, "headers": raw_headers})

    try:
        async for chunk in body_iter:
            data = chunk.encode("utf-8")
            await send({"type": "http.response.body", "body": data, "more_body": True})

        await send({"type": "http.response.body", "body": b"", "more_body": False})

    except Exception:
        # Client disconnected while writing, or generator errored.
        with contextlib.suppress(Exception):
            await body_iter.aclose()
        with contextlib.suppress(Exception):
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        raise


# -----------------------------------------------------------------------------
# Streaming tool call (manual SSE) - streams on the POST connection
# -----------------------------------------------------------------------------
async def _stream_tool_call_sse(
    rid: Any,
    tool_name: str,
    arguments: Dict[str, Any],
    session_id: str,
) -> AsyncIterator[str]:
    """
    Stream a single tools/call execution over SSE on the POST connection.

    Notes
    -----
    - Emits notifications/message frames for progress.
    - Terminates with a final JSON-RPC response frame with the CallToolResult.
    - Uses a queue to send keepalives even when the underlying tool stream is quiet.
    """
    canonical_name = _canonicalize_tool_name(tool_name)
    logger.info("Streaming tools/call start tool_id=%s session_id=%s", canonical_name, session_id)

    keepalive_seconds = 10.0
    final_payload: Optional[Dict[str, Any]] = None

    queue: asyncio.Queue[Any] = asyncio.Queue()
    done_sentinel = object()

    async def _producer() -> None:
        try:
            async for item in tools_core.invoke_tool_stream_async(
                tool_id=canonical_name,
                arguments=arguments or {},
                session_id=session_id,  # REQUIRED: propagate session_id for cancellation
            ):
                await queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(done_sentinel)

    producer_task = asyncio.create_task(_producer())

    try:
        yield _sse_frame(_notification_message({"type": "started", "tool_id": canonical_name}), event="message")

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if item is done_sentinel:
                break

            if isinstance(item, Exception):
                raise item

            if isinstance(item, dict):
                yield _sse_frame(_notification_message(item), event="message")
                if item.get("type") == "final":
                    final_payload = item.get("payload")
            else:
                yield _sse_frame(
                    _notification_message({"type": "progress", "data": str(item)}),
                    event="message",
                )

        if final_payload is None:
            final_payload = {"tool_id": canonical_name, "ok": False, "error": "Missing final payload"}

        ok = bool(final_payload.get("ok"))
        text = json.dumps(final_payload, indent=2, default=str)

        call_result = types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            isError=not ok,
        )

        yield _sse_frame(
            {"jsonrpc": "2.0", "id": rid, "result": _result_to_dict(call_result)},
            event="message",
        )

        logger.info("Streaming tools/call completed tool_id=%s ok=%s", canonical_name, ok)

    except Exception as exc:
        logger.exception("Error in streaming tools/call tool_id=%s: %s", canonical_name, exc)
        yield _sse_frame(
            {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(exc)}},
            event="message",
        )

    finally:
        if not producer_task.done():
            producer_task.cancel()
            with contextlib.suppress(Exception):
                await producer_task


# -----------------------------------------------------------------------------
# JSON-RPC handler (manual)
# -----------------------------------------------------------------------------
async def _handle_one_rpc(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rid = req.get("id")
    method = req.get("method")
    is_notification = rid is None

    if req.get("jsonrpc") != "2.0" or not method:
        if is_notification:
            return None
        return _jsonrpc_error(rid, -32600, "Invalid Request")

    if method == "initialize":
        params = req.get("params") or {}
        protocol_version = str(params.get("protocolVersion") or os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")).strip()
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": rid, "result": _make_initialize_result(protocol_version)}

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        payload = await _list_tools_payload()
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": rid, "result": payload}

    if method == "tools/call":
        params = req.get("params") or {}
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        canonical = _canonicalize_tool_name(tool_name)

        # Non-streaming tools handled here.
        if canonical not in getattr(tools_core, "STREAMING_TOOL_IDS", set()):
            res = await _call_tool_non_streaming(tool_name, dict(arguments))
            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": rid, "result": _result_to_dict(res)}

        # Streaming tools must be handled by the SSE POST path.
        if is_notification:
            return None
        return _jsonrpc_error(rid, -32600, "Streaming tool must be handled via SSE path")

    if is_notification:
        return None
    return _jsonrpc_error(rid, -32601, f"Method not found: {method}")


# -----------------------------------------------------------------------------
# Cancel endpoint: explicit cancellation for a session
# -----------------------------------------------------------------------------
async def cancel(request: Request):
    """
    POST /mcp/cancel

    Cancellation is keyed by Mcp-Session-Id.
    Clients should send the Mcp-Session-Id header (preferred).
    As a fallback, JSON body may include {"session_id": "..."}.
    """
    session_id = request.headers.get("Mcp-Session-Id")

    if not session_id:
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            session_id = body.get("session_id") or body.get("mcp_session_id")

    if not session_id:
        return JSONResponse({"ok": False, "error": "Missing Mcp-Session-Id"}, status_code=400)

    logger.info("Cancel requested via /mcp/cancel session_id=%s", session_id)

    try:
        logger.info("Calling tools_core.request_cancel session_id=%s", session_id)
        tools_core.request_cancel(session_id)
        logger.info("tools_core.request_cancel returned session_id=%s", session_id)
    except Exception as exc:
        logger.exception("tools_core.request_cancel failed session_id=%s err=%s", session_id, exc)
        return JSONResponse({"ok": False, "session_id": session_id, "error": str(exc)}, status_code=500)

    return JSONResponse({"ok": True, "session_id": session_id})


# -----------------------------------------------------------------------------
# HTTP handler for /mcp (single implementation)
# -----------------------------------------------------------------------------
async def _handle_mcp_get(scope: Scope, receive: Receive, send: Send) -> None:
    session_id = _get_or_create_session_id(scope)

    resp = StreamingResponse(
        _sse_subscribe(session_id),
        media_type="text/event-stream",
        headers={
            "Mcp-Session-Id": session_id,
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await resp(scope, receive, send)


async def _handle_mcp_post(scope: Scope, receive: Receive, send: Send) -> None:
    if not _accept_has_both(scope):
        await JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": "server-error",
                "error": {
                    "code": -32600,
                    "message": "Not Acceptable: Client must accept both application/json and text/event-stream",
                },
            },
            status_code=406,
        )(scope, receive, send)
        return

    body_bytes = await _read_body(receive)
    if not body_bytes:
        await JSONResponse(
            _jsonrpc_error("server-error", -32600, "Invalid Request: empty body"),
            status_code=400,
        )(scope, receive, send)
        return

    try:
        payload_obj: Any = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        await JSONResponse(
            _jsonrpc_error("server-error", -32700, "Parse error: invalid JSON"),
            status_code=400,
        )(scope, receive, send)
        return

    logger.debug("Incoming /mcp POST payload (scrubbed): %s", _scrub_secrets(payload_obj))

    session_id = _get_or_create_session_id(scope)
    token = _CURRENT_SESSION_ID.set(session_id)

    try:
        wants_streaming = False
        if isinstance(payload_obj, dict):
            wants_streaming = _is_streaming_tools_call(payload_obj)
        elif isinstance(payload_obj, list):
            wants_streaming = any(_is_streaming_tools_call(x) for x in payload_obj if isinstance(x, dict))

        # Streaming path: only allow a single tools/call object
        if wants_streaming:
            if not isinstance(payload_obj, dict) or payload_obj.get("method") != "tools/call":
                await JSONResponse(
                    _jsonrpc_error("server-error", -32600, "Batch/unsupported streaming request shape"),
                    status_code=400,
                    headers={"Mcp-Session-Id": session_id},
                )(scope, receive, send)
                return

            rid = payload_obj.get("id")
            params = payload_obj.get("params") or {}
            tool_name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}

            headers = {
                "Mcp-Session-Id": session_id,
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
            }

            body_iter = _stream_tool_call_sse(rid, tool_name, dict(arguments), session_id)
            await _send_sse_stream(
                send,
                body_iter,
                status_code=200,
                headers=headers,
            )
            return

        # Manual JSON path (initialize, tools/list, non-stream tools/call, etc.)
        responses: List[Dict[str, Any]] = []

        if isinstance(payload_obj, dict):
            r = await _handle_one_rpc(payload_obj)
            if r is not None:
                responses.append(r)
        elif isinstance(payload_obj, list):
            for item in payload_obj:
                if not isinstance(item, dict):
                    continue
                r = await _handle_one_rpc(item)
                if r is not None:
                    responses.append(r)
        else:
            await JSONResponse(
                _jsonrpc_error("server-error", -32600, "Invalid Request: expected object or array"),
                status_code=400,
                headers={"Mcp-Session-Id": session_id},
            )(scope, receive, send)
            return

        # Only notifications
        if not responses:
            await PlainTextResponse("", status_code=202, headers={"Mcp-Session-Id": session_id})(scope, receive, send)
            return

        out_payload: Any = responses[0] if (isinstance(payload_obj, dict) and len(responses) == 1) else responses
        await JSONResponse(out_payload, status_code=200, headers={"Mcp-Session-Id": session_id})(scope, receive, send)

    finally:
        _CURRENT_SESSION_ID.reset(token)


async def _handle_mcp(scope: Scope, receive: Receive, send: Send) -> None:
    if scope.get("type") != "http":
        await PlainTextResponse("Unsupported", status_code=400)(scope, receive, send)
        return

    http_method = (scope.get("method") or "").upper()

    if http_method == "GET":
        await _handle_mcp_get(scope, receive, send)
        return

    if http_method == "POST":
        await _handle_mcp_post(scope, receive, send)
        return

    await JSONResponse({"ok": False, "error": "Unsupported method"}, status_code=405)(scope, receive, send)


# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------
async def health(_request):
    try:
        return JSONResponse(tools_core.health_summary())
    except Exception as exc:
        logger.exception("Error in /health: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# -----------------------------------------------------------------------------
# App wiring
# -----------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncIterator[None]:
    logger.info("Starting MCP HTTP server (manual JSON + manual SSE)")
    yield
    logger.info("Stopped MCP HTTP server")


app = Starlette(
    debug=False,
    lifespan=lifespan,
    routes=[
        Route("/health", endpoint=health, methods=["GET"]),
        # IMPORTANT: route must be before the /mcp mount so it doesn't get swallowed by the mount.
        Route("/mcp/cancel", endpoint=cancel, methods=["POST"]),
        Mount("/mcp", app=_handle_mcp),
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
