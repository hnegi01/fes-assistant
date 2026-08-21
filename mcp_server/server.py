"""FES MCP server — Streamable HTTP on the OFFICIAL SDK machinery.

Rebuilt 2026-08-16 on the official SDK — a probe spike confirmed the
old session-manager incompatibility no longer reproduces on mcp>=1.24. The
hand-rolled transport (866 lines of manual JSON-RPC + manual SSE, written when
the SDK's session manager had compatibility issues that no longer reproduce on
mcp==1.24) is replaced by `StreamableHTTPSessionManager` + the lowlevel
`Server`. Everything protocol-shaped — sessions, SSE-on-POST, the GET stream,
notifications — is the SDK's. What remains here is ONLY ours:

  - tools/list from the allowlisted registry (tools_core.TOOLS_BY_ID), schemas
    augmented with the per-call credential properties, name mode applied
    (MCP_TOOL_NAME_MODE=claude exposes underscores; both forms accepted on call)
  - tools/call → tools_core dispatch. Streaming tools publish on BOTH channels:
    notifications/message carries the full emit() payload (the run-log / UI
    contract, unchanged), and spec notifications/progress fires when the caller
    sent a progressToken (processed_so_far → progress, total_count → total)
  - spec cancellation bridged to the tools_core cancel flag: anyio cancellation
    cannot interrupt the SDK thread, so on cancel we set the flag and the
    thread stops at its next emit() checkpoint
  - POST /mcp/cancel — out-of-band operator kill switch (deliberate extension:
    in-band cancel needs a live session, which a wedged run may not have)
  - GET /health

Deliberately NOT the lowlevel decorator API of FastMCP: FastMCP generates
schemas from function signatures and validates arguments, which would reject
the credentials the backend injects per call (see FINDINGS.md).

MUST run single-worker: sessions and cancel flags live in-process.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import uuid
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Tuple

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp_server import tools_core

# -----------------------------------------------------------------------------
# Logging (file + console, same destination as before)
# -----------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Own name + own handler: tools_core already owns the "mcp_server" logger
# (handler → tools_core.log), so reusing that name silently rerouted this
# module's lines there on first deploy.
logger = logging.getLogger("mcp_server.server")
if not logger.handlers:
    logger.setLevel(os.getenv("FES_LOG_LEVEL", "INFO").upper())
    fh = TimedRotatingFileHandler(  # daily file, 7 kept — same policy as every other log
        LOG_DIR / "server.log", when="midnight", backupCount=7, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

MCP_TOOL_NAME_MODE = os.getenv("MCP_TOOL_NAME_MODE", "claude").strip().lower()

# -----------------------------------------------------------------------------
# Exposed schemas: registry parameters + per-call credential properties.
# The registry never declares credentials (they are injected by the caller);
# the EXPOSED schema advertises them so any client knows what a call needs.
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
    """Inject tenant credential properties into a tool's exposed input schema.

    Migration tools get source_*/target_*; everything else domain/token/ssl.
    `emit` is removed — MCP clients cannot pass callbacks.
    """
    schema = copy.deepcopy(row.get("parameters") or {})
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "required": []}

    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])

    if row.get("module") == "migration":
        for k, v in {**SOURCE_TENANT_PROPS, **TARGET_TENANT_PROPS}.items():
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
# Tool name modes: canonical dotted ids vs Claude-style underscores.
# Exposure follows MCP_TOOL_NAME_MODE; calls are accepted in EITHER form.
# -----------------------------------------------------------------------------
def _to_claude_tool_name(tool_id: str) -> str:
    return tool_id.replace(".", "_")


def _build_tool_name_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    canonical_to_alias: Dict[str, str] = {}
    alias_to_canonical: Dict[str, str] = {}
    for tid in getattr(tools_core, "TOOLS_BY_ID", {}):
        alias = _to_claude_tool_name(tid)
        canonical_to_alias[tid] = alias
        if alias not in alias_to_canonical:
            alias_to_canonical[alias] = tid
        elif alias_to_canonical[alias] != tid:
            logger.warning(
                "Claude tool alias collision: alias=%s maps to both %s and %s; keeping %s",
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
    return _ALIAS_TO_CANONICAL.get(name, name)


def _exposed_tool_name(canonical_tool_id: str) -> str:
    if MCP_TOOL_NAME_MODE == "claude":
        return _CANONICAL_TO_ALIAS.get(canonical_tool_id, canonical_tool_id)
    return canonical_tool_id


# -----------------------------------------------------------------------------
# The lowlevel MCP server: our two handlers
# -----------------------------------------------------------------------------
server: Server = Server("fes-mcp")

# The backend injects its Mcp-Session-Id under this argument name so the
# tools_core cancel flags (and the /mcp/cancel side-door) are keyed by the
# same id on both paths. Popped before dispatch — the SDK never sees it.
SESSION_ID_ARG = "fes_mcp_session_id"


@server.list_tools()
async def _list_tools() -> list[types.Tool]:
    rows = tools_core.list_tools()
    out: list[types.Tool] = []
    for row in rows:
        tid = row.get("tool_id")
        if not tid:
            continue
        out.append(
            types.Tool(
                name=_exposed_tool_name(tid),
                description=row.get("description") or "",
                inputSchema=_augment_input_schema(row),
            )
        )
    logger.info("tools/list served: %d tool(s), name_mode=%s", len(out), MCP_TOOL_NAME_MODE)
    return out


@server.call_tool()
async def _call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
    tool_id = _canonicalize_tool_name(name)
    args = dict(arguments or {})
    session_id = str(args.pop(SESSION_ID_ARG, "") or "") or f"anon-{uuid.uuid4().hex}"

    ctx = server.request_context
    progress_token = ctx.meta.progressToken if ctx.meta else None

    logger.info(
        "tools/call name=%s canonical=%s session=%s streaming=%s",
        name,
        tool_id,
        session_id,
        tool_id in tools_core.STREAMING_TOOL_IDS,
    )

    if tool_id not in tools_core.STREAMING_TOOL_IDS:
        payload = await tools_core.invoke_tool_async(tool_id=tool_id, arguments=args)
        return [types.TextContent(type="text", text=json.dumps(payload, default=str))]

    # Streaming tool: forward every emit() payload on the message channel (the
    # run-log / UI contract) and, when the caller sent a progressToken, mirror
    # the counters on the spec progress channel.
    final_payload: Dict[str, Any] = {"tool_id": tool_id, "ok": False, "error": "No result produced"}
    seq = 0.0
    try:
        async for item in tools_core.invoke_tool_stream_async(tool_id, args, session_id=session_id):
            if item.get("type") == "final":
                final_payload = item.get("payload") or final_payload
                continue
            # related_request_id routes the notification onto THIS request's
            # stream — without it, Streamable HTTP sends it to the standalone
            # GET stream and mid-call narration never reaches the caller
            # (found live 2026-08-16: spec progress arrived, narration didn't).
            with contextlib.suppress(Exception):
                await ctx.session.send_log_message(
                    level="info", data=item, logger="fes.progress", related_request_id=ctx.request_id
                )
            if progress_token is not None:
                seq += 1.0
                progress = item.get("processed_so_far")
                total = item.get("total_count")
                with contextlib.suppress(Exception):
                    await ctx.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=float(progress) if progress is not None else seq,
                        total=float(total) if total is not None else None,
                        message=item.get("message") or None,
                    )
    except (anyio.get_cancelled_exc_class(), GeneratorExit):
        # Spec cancellation (notifications/cancelled) cancelled this task. The
        # SDK thread underneath cannot be interrupted — set the flag so it
        # stops at its next emit() checkpoint, then let the cancellation fly.
        tools_core.request_cancel(session_id)
        logger.info("tools/call cancelled in-band; cancel flag set session=%s tool=%s", session_id, tool_id)
        raise

    return [types.TextContent(type="text", text=json.dumps(final_payload, default=str))]


# -----------------------------------------------------------------------------
# Transport: the SDK's session manager, mounted under /mcp
# -----------------------------------------------------------------------------
manager = StreamableHTTPSessionManager(app=server, json_response=False)


async def _mcp_asgi(scope, receive, send):
    await manager.handle_request(scope, receive, send)


# -----------------------------------------------------------------------------
# Ops extensions (deliberately outside the protocol)
# -----------------------------------------------------------------------------
async def cancel(request: Request) -> JSONResponse:
    """POST /mcp/cancel — out-of-band kill switch, session-scoped.

    The spec's notifications/cancelled is the primary path (in-band, sent by
    the backend client). This endpoint exists for the case where the session
    stream is wedged or gone — a plain POST with the session header works from
    anywhere. Both paths end at the same tools_core cancel flag.
    """
    session_id = request.headers.get("mcp-session-id") or request.headers.get("x-session-id")
    if not session_id:
        return JSONResponse({"ok": False, "error": "Missing Mcp-Session-Id header"}, status_code=400)
    try:
        tools_core.request_cancel(session_id)
        logger.info("Cancel requested via /mcp/cancel session_id=%s", session_id)
        return JSONResponse({"ok": True, "session_id": session_id})
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("request_cancel failed session_id=%s", session_id)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


async def health(_request: Request) -> JSONResponse:
    return JSONResponse(tools_core.health_summary())


@contextlib.asynccontextmanager
async def _lifespan(app: Starlette):
    async with manager.run():
        logger.info("MCP server up (official SDK transport), name_mode=%s", MCP_TOOL_NAME_MODE)
        yield


app = Starlette(
    routes=[
        Route("/health", endpoint=health, methods=["GET"]),
        Route("/mcp/cancel", endpoint=cancel, methods=["POST"]),
        Mount("/mcp", app=_mcp_asgi),
    ],
    lifespan=_lifespan,
)
