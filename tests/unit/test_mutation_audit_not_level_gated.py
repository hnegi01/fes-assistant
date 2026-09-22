"""The mutation audit log must never be silenced by FES_LOG_LEVEL.

Both audit loggers were created with `setLevel(_log_level)`, so the documented,
supported value `FES_LOG_LEVEL=WARNING` silently discarded every mutation
record — while `.env.example` stated the audit trail "has no off switch" and
`docs/security.md` stated "every execution is recorded". No error, no warning:
the files simply stopped growing. An operator turning down log noise turned off
their audit trail.
"""

from __future__ import annotations

import logging
import os

import pytest

from backend.agent import _config as backend_config
from mcp_server import tools_core

AUDIT_LOGGERS = [
    pytest.param(backend_config.audit_logger, id="backend"),
    pytest.param(tools_core.audit_logger, id="mcp"),
]


@pytest.mark.parametrize("audit_logger", AUDIT_LOGGERS)
def test_audit_logger_is_pinned_at_info(audit_logger: logging.Logger) -> None:
    assert audit_logger.level == logging.INFO, (
        "audit level must be INFO unconditionally, never FES_LOG_LEVEL — "
        "a WARNING level silently discards the whole audit trail"
    )


def _app_file_handlers(audit_logger: logging.Logger) -> list[logging.FileHandler]:
    """The handlers the APPLICATION created.

    pytest's own logging plugin attaches a `_FileHandler` pointed at os.devnull
    at NOTSET; asserting over it tests the harness, not the code.
    """
    return [
        h
        for h in audit_logger.handlers
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") not in (os.devnull, "/dev/null")
    ]


@pytest.mark.parametrize("audit_logger", AUDIT_LOGGERS)
def test_audit_handlers_are_pinned_at_info(audit_logger: logging.Logger) -> None:
    handlers = _app_file_handlers(audit_logger)
    assert handlers, "audit logger must have a file handler"
    for h in handlers:
        assert h.level == logging.INFO


@pytest.mark.parametrize("audit_logger", AUDIT_LOGGERS)
def test_a_mutation_line_survives_a_critical_root_level(audit_logger: logging.Logger) -> None:
    """The end-to-end property: raise the level, the record is still emitted."""
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler = _Collect(level=logging.INFO)
    logging.getLogger().setLevel(logging.CRITICAL)  # operator turns down the noise
    audit_logger.addHandler(handler)
    try:
        audit_logger.info("EXECUTING mutation tool=dashboard.delete_dashboard")
    finally:
        audit_logger.removeHandler(handler)
        logging.getLogger().setLevel(logging.WARNING)

    assert seen, "the audit record was swallowed — the trail is level-gated again"


@pytest.mark.parametrize("audit_logger", AUDIT_LOGGERS)
def test_audit_file_rotates_so_it_cannot_fill_the_disk(audit_logger: logging.Logger) -> None:
    from logging.handlers import TimedRotatingFileHandler

    file_handlers = _app_file_handlers(audit_logger)
    assert file_handlers
    assert all(isinstance(h, TimedRotatingFileHandler) for h in file_handlers), (
        "a plain FileHandler grows without bound; this is the file you least want to lose to a full disk"
    )
