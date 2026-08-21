"""MCP client for the FES backend — on the OFFICIAL SDK (mcp==1.24).

Rebuilt 2026-08-16 on the official SDK (probe spike: the old
incompatibility no longer reproduces on mcp>=1.24). The hand-rolled
JSON-RPC-over-httpx client (~790 lines: manual initialize, manual SSE frame
parsing, manual session headers) is replaced by the SDK's
`streamablehttp_client` + `ClientSession`. What remains here is ONLY ours:

  - per-call credential injection (`_inject_credentials`): chat tools get the
    tenant config, migration tools get source_*/target_* — never from env
  - the session id rides along as `fes_mcp_session_id` in the arguments so the
    server keys its cancel flags consistently for BOTH cancel paths
  - narration: the server's notifications/message events (full emit() payloads)
    arrive via the SDK's logging_callback and are re-published to
    backend.runtime.publish_progress in the same envelope shape the UI has
    always consumed — the frontend contract is unchanged
  - spec progress: notifications/progress arrives via progress_callback and is
    logged (the UI's display feed is the message channel; this proves the spec
    channel without duplicating lines)
  - cancellation, two paths to the same server-side flag:
      1. spec (primary): notifications/cancelled for every in-flight request id
      2. ops side-door (fallback): POST /mcp/cancel with the session header —
         works even when the session stream is wedged

Public surface: connect / close / invoke_tool / cancel_session / health,
constructed with (base_url, tenant_config, migration_config). Only
backend.runtime builds it. (No `list_tools`: the agent's menu comes from the
generated registry in config/, not from MCP discovery — the server's
tools/list exists for MCP clients, and this client never asked.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, Optional, Set

import httpx
import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ._config import _make_module_logger

logger = _make_module_logger("backend.agent.mcp_client", "mcp_client.log")

# Must match mcp_server.server.SESSION_ID_ARG (popped server-side pre-dispatch).
SESSION_ID_ARG = "fes_mcp_session_id"


class McpClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        tenant_config: Optional[Dict[str, Any]] = None,
        migration_config: Optional[Dict[str, Any]] = None,
        ui_session_id: Optional[str] = None,
    ):
        # The UI session this client belongs to. Notifications are dispatched
        # by the SDK's receive-loop task (context snapshotted at connect), so
        # progress must be routed by session id, not by ContextVar.
        self._ui_session_id = ui_session_id
        self._base_url = (base_url or os.getenv("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")).rstrip("/")
        # 1800s: a migration can legitimately run for many minutes, and the
        # client MUST NOT time out before the server finishes — tool calls are
        # never retried, and abandoning a live migration mid-flight leaves the
        # target half-written. Reads finish in seconds regardless.
        self._timeout_seconds = float(os.getenv("PYSISENSE_MCP_HTTP_TIMEOUT", "1800"))

        self._tenant_config: Dict[str, Any] = tenant_config or {}
        self._migration_config: Dict[str, Any] = migration_config or {}

        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._get_session_id = None
        self._mcp_session_id: Optional[str] = None

        self._init_lock = asyncio.Lock()
        # Request ids currently in flight. Captured best-effort (the counter
        # read races under fan-out), which is fine: our cancellation semantics
        # are session-scoped — on cancel we notify EVERY in-flight id.
        self._inflight: Set[int] = set()

        logger.info(
            "McpClient initialized base_url=%s timeout=%ss (official SDK)", self._base_url, self._timeout_seconds
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        async with self._init_lock:
            if self._session is not None:
                return
            stack = AsyncExitStack()
            try:
                read, write, get_session_id = await stack.enter_async_context(
                    streamablehttp_client(
                        f"{self._base_url}/mcp/",
                        timeout=timedelta(seconds=self._timeout_seconds),
                        sse_read_timeout=timedelta(seconds=self._timeout_seconds),
                    )
                )
                session = await stack.enter_async_context(
                    ClientSession(read, write, logging_callback=self._on_log_message)
                )
                await session.initialize()
            except BaseException:
                with contextlib.suppress(Exception):
                    await stack.aclose()
                raise
            self._stack = stack
            self._session = session
            self._get_session_id = get_session_id
            self._mcp_session_id = get_session_id() if callable(get_session_id) else None
            logger.info("MCP session established session_id=%s", self._mcp_session_id)

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()
        logger.debug("McpClient.close() completed.")

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return self._session

    # ------------------------------------------------------------------
    # Notifications from the server
    # ------------------------------------------------------------------
    async def _on_log_message(self, params: types.LoggingMessageNotificationParams) -> None:
        """notifications/message → backend.runtime.publish_progress, in the
        exact envelope shape the UI has consumed since v1 (frontend extracts
        `params.data`)."""
        try:
            from backend import runtime as runtime_mod
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Cannot import backend.runtime for progress publish: %s", exc)
            return
        event = {
            "source": "mcp",
            "type": "notification",
            "method": "notifications/message",
            "params": {"level": params.level, "logger": params.logger, "data": params.data},
        }
        logger.debug("narration event received: %s", str(params.data)[:200])
        try:
            await runtime_mod.publish_progress_for(self._ui_session_id, event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("publish_progress failed: %s", exc, exc_info=True)

    @staticmethod
    async def _on_spec_progress(progress: float, total: float | None, message: str | None) -> None:
        """notifications/progress — the spec channel. The UI's display feed is
        the message channel (full payloads), so this is logged, not re-published."""
        logger.info("spec progress: %s/%s %s", progress, total, message or "")

    # ------------------------------------------------------------------
    # Credential injection (ours, unchanged semantics)
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
        module = tool_id.split(".", 1)[0] if "." in tool_id else tool_id.split("_", 1)[0]
        if module == "migration":
            return self._with_migration(arguments)
        return self._with_tenant(arguments)

    # ------------------------------------------------------------------
    # Public API (used by the agent / runtime)
    # ------------------------------------------------------------------
    async def invoke_tool(self, tool_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        session = await self._ensure_session()

        args = self._inject_credentials(tool_id, arguments or {})
        sid = self._get_session_id() if callable(self._get_session_id) else self._mcp_session_id
        if sid:
            args[SESSION_ID_ARG] = sid

        logger.info("tools/call name=%s (session=%s)", tool_id, sid)

        rid = session._request_id  # best-effort capture; see _inflight note above
        self._inflight.add(rid)
        try:
            result = await session.call_tool(
                tool_id,
                args,
                read_timeout_seconds=timedelta(seconds=self._timeout_seconds),
                progress_callback=self._on_spec_progress,
            )
        except asyncio.CancelledError:
            # The turn is being torn down (Stop / disconnect). Best-effort,
            # shielded: spec-cancel everything in flight, then the side-door.
            with contextlib.suppress(BaseException):
                await asyncio.shield(asyncio.create_task(self._cancel_all("turn cancelled")))
            raise
        finally:
            self._inflight.discard(rid)

        return self._parse_call_result(tool_id, result)

    @staticmethod
    def _parse_call_result(tool_id: str, result: types.CallToolResult) -> Dict[str, Any]:
        text: Optional[str] = None
        for block in result.content or []:
            if isinstance(block, types.TextContent):
                text = block.text
                break
        if result.isError:
            return {"tool_id": tool_id, "ok": False, "error": text or "Tool call failed"}
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
                return {"tool_id": tool_id, "result": parsed}
            except Exception:
                return {"tool_id": tool_id, "result": text}
        return {"tool_id": tool_id, "result": None}

    async def _cancel_all(self, reason: str) -> None:
        """Spec path first (notifications/cancelled per in-flight id), then the
        ops side-door — both end at the same server-side flag."""
        session = self._session
        if session is not None:
            for rid in sorted(self._inflight):
                with contextlib.suppress(Exception):
                    await session.send_notification(
                        types.ClientNotification(
                            types.CancelledNotification(
                                method="notifications/cancelled",
                                params=types.CancelledNotificationParams(requestId=rid, reason=reason),
                            )
                        )
                    )
                    logger.info("Sent notifications/cancelled requestId=%s (%s)", rid, reason)
        with contextlib.suppress(Exception):
            await self._post_cancel_endpoint()

    async def _post_cancel_endpoint(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or (self._get_session_id() if callable(self._get_session_id) else None) or self._mcp_session_id
        if not sid:
            return {"ok": False, "error": "Missing Mcp-Session-Id"}
        async with httpx.AsyncClient(base_url=self._base_url, timeout=10.0) as http:
            resp = await http.post("/mcp/cancel", headers={"Mcp-Session-Id": sid}, json={"session_id": sid})
            try:
                return resp.json()
            except Exception:
                return {"ok": resp.status_code == 200, "status_code": resp.status_code}

    async def cancel_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel whatever this session is running. Spec notifications for the
        in-flight ids (primary), then POST /mcp/cancel (works even when the
        session stream is wedged — see server docstring)."""
        session = self._session
        if session is not None and session_id is None:
            for rid in sorted(self._inflight):
                with contextlib.suppress(Exception):
                    await session.send_notification(
                        types.ClientNotification(
                            types.CancelledNotification(
                                method="notifications/cancelled",
                                params=types.CancelledNotificationParams(requestId=rid, reason="cancel_session"),
                            )
                        )
                    )
        return await self._post_cancel_endpoint(session_id)

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=10.0) as http:
            resp = await http.get("/health")
            data = resp.json()
            return data if isinstance(data, dict) else {}
