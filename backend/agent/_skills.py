"""Skills — Sisense-authored procedures the planner plans from.

A skill is one directory under ``skills/`` holding a ``SKILL.md``: YAML
frontmatter (the machine contract — name, description, version, tools,
guardrails, requires_role) and a markdown body (the procedure the planner
reads). Design: ``docs/design/skills.md``.

This module is the LOADER and the planner's two touchpoints:

- ``load_skills()`` — every valid skill, by name. Parsed once and mtime-cached
  over the directory, like the allowlist, so editing a skill takes effect on
  the next turn without a restart. A skill that fails to parse or validate is
  EXCLUDED AND LOGGED AT ERROR, never shipped half-broken and never silently
  dropped. A missing directory (or ``FES_SKILLS_ENABLED=false``) means no
  skills — the agent behaves exactly as before this module existed.
- ``skills_index_text()`` — the one-line-per-skill index the planner always
  sees (progressive disclosure: names now, the body only on match).
- ``parse_skill_directive()`` — recognises the planner's opt-in, ``SKILL: <name>``
  on the first line, which triggers the second planning pass with the body.

Validation is deliberately strict about TOOLS. A skill's ``tools`` list is an
allowlist within the allowlist: every id must exist in the registry AND be
exposed, and every tool the body names must be declared. That is what turns
"never change ownership" from a request into an impossibility — the tool is
simply not in the list — and it is what the drift guard in
``tests/unit/test_skills.py`` pins so a rename or delisting fails CI instead of
breaking a customer's workflow months later.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import yaml

from ._config import ROOT_DIR, _make_module_logger

logger = _make_module_logger("backend.agent.llm_skills", "llm_skills.log")

_skills_env = os.getenv("FES_SKILLS_DIR")
SKILLS_DIR: Path = Path(_skills_env) if _skills_env else ROOT_DIR / "skills"
SKILLS_ENABLED: bool = os.getenv("FES_SKILLS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
SKILL_FILE = "SKILL.md"

# Frontmatter contract. Unknown keys are an error, not ignored: a typo'd key
# ("guardrail:") would otherwise silently disable the thing it was meant to
# enforce.
FRONTMATTER_KEYS = frozenset(
    {"name", "description", "version", "requires_role", "tools", "guardrails", "compensations", "step_labels"}
)
REQUIRED_KEYS = frozenset({"name", "description", "version", "tools"})
# Guardrails the runtime knows how to enforce. A skill declaring any other id
# is rejected — an unenforced guardrail is a promise the code cannot keep.
KNOWN_GUARDRAILS = frozenset({"validate-before-swap"})

_TOOL_ID_RE = re.compile(r"`([a-z_]+\.[a-z_][a-z0-9_]*)`")
_SKILL_DIRECTIVE_RE = re.compile(r"^\s*SKILL:\s*([A-Za-z0-9][A-Za-z0-9_-]*)\s*$")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n(.*)\Z", re.S)


class SkillError(ValueError):
    """A SKILL.md that cannot be used. The message says exactly why."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: int
    tools: Tuple[str, ...]
    body: str
    path: Path
    requires_role: Optional[str] = None
    guardrails: Tuple[Dict[str, Any], ...] = ()
    # tool_id -> {"tool": <undo tool_id>, "args": {param: "{args.x}" | "{result.y}" | literal}}
    # Declared in the skill, enforced by the runtime (design §9): what to run,
    # in reverse, for each completed step of that kind when a later step fails.
    compensations: Dict[str, Dict[str, Any]] = None  # type: ignore[assignment]
    # The `## Approval` section of the body: what the user is asked to approve,
    # in their words, with `{method.args.param}` / `{method.result.path}` /
    # `{method.count}` placeholders code fills from the plan and the read
    # results gathered before the gate (design §8/§11). Empty = generic dialog.
    approval: str = ""
    # tool_id -> what the progress line says while that tool runs, in the
    # user's words ("Building the model"). Falls back to the tool description.
    step_labels: Dict[str, str] = None  # type: ignore[assignment]
    # The `## Ask` section: how to ask the user for a value only they can give,
    # once the reads have run — same placeholders as `approval`. Empty = the
    # plan's own question text.
    ask: str = ""
    # The `## Report` section: the summary shown when the run ends, same
    # placeholders plus per-item ones (`{method.result|items_ran}` …).
    report: str = ""

    def __post_init__(self) -> None:
        if self.compensations is None:
            object.__setattr__(self, "compensations", {})
        if self.step_labels is None:
            object.__setattr__(self, "step_labels", {})

    def body_tool_ids(self) -> Set[str]:
        """Every ``package.method`` the procedure names in backticks."""
        return set(_TOOL_ID_RE.findall(self.body))

    @property
    def planner_body(self) -> str:
        """The body the PLANNER sees: without `## Ask` and `## Approval`, which
        are rendered by code with placeholders the model must never copy."""
        return _without_sections(self.body, ("Ask", "Approval"))


def parse_skill_file(path: Path) -> Skill:
    """Parse one SKILL.md. Raises SkillError with a precise reason on any defect.

    Structural checks only — registry/allowlist membership needs the registry
    and is applied by ``load_skills`` (or the drift guard) via
    ``validate_against_surface``.
    """
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise SkillError("missing YAML frontmatter (the file must start with a --- block)")
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(fm, dict):
        raise SkillError("frontmatter must be a mapping")

    unknown = set(fm) - FRONTMATTER_KEYS
    if unknown:
        raise SkillError(f"unknown frontmatter key(s): {sorted(unknown)}")
    missing = REQUIRED_KEYS - set(fm)
    if missing:
        raise SkillError(f"missing frontmatter key(s): {sorted(missing)}")

    name = str(fm["name"]).strip()
    if name != path.parent.name:
        raise SkillError(f"name {name!r} does not match its directory {path.parent.name!r}")
    description = " ".join(str(fm["description"]).split())
    if not description:
        raise SkillError("description is empty")

    version = fm["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SkillError("version must be a positive integer")

    tools = fm["tools"]
    if not isinstance(tools, list) or not tools:
        raise SkillError("tools must be a non-empty list")
    for t in tools:
        if not isinstance(t, str) or "." not in t:
            raise SkillError(f"tools entries must be 'package.method' ids, got {t!r}")
    if len(set(tools)) != len(tools):
        raise SkillError("tools contains duplicates")

    role = fm.get("requires_role")
    if role is not None and (not isinstance(role, str) or not role.strip()):
        raise SkillError("requires_role must be a non-empty string when present")

    guardrails_raw = fm.get("guardrails") or []
    if not isinstance(guardrails_raw, list):
        raise SkillError("guardrails must be a list")
    guardrails = []
    for g in guardrails_raw:
        if not isinstance(g, dict) or not isinstance(g.get("id"), str):
            raise SkillError("each guardrail must be a mapping with a string 'id'")
        if g["id"] not in KNOWN_GUARDRAILS:
            raise SkillError(f"unknown guardrail id {g['id']!r}; the runtime can enforce {sorted(KNOWN_GUARDRAILS)}")
        guardrails.append(dict(g))

    comps_raw = fm.get("compensations") or {}
    if not isinstance(comps_raw, dict):
        raise SkillError("compensations must be a mapping of tool_id -> {tool, args}")
    compensations: Dict[str, Dict[str, Any]] = {}
    for step_tool, spec in comps_raw.items():
        if not isinstance(step_tool, str) or "." not in step_tool:
            raise SkillError(f"compensations key must be a 'package.method' id, got {step_tool!r}")
        if not isinstance(spec, dict) or not isinstance(spec.get("tool"), str) or "." not in spec["tool"]:
            raise SkillError(f"compensation for {step_tool} needs a 'tool' id")
        args = spec.get("args") or {}
        if not isinstance(args, dict):
            raise SkillError(f"compensation for {step_tool}: 'args' must be a mapping")
        if step_tool not in tools or spec["tool"] not in tools:
            raise SkillError(f"compensation {step_tool} -> {spec['tool']}: both tools must be declared in `tools`")
        compensations[step_tool] = {"tool": spec["tool"], "args": dict(args)}

    labels_raw = fm.get("step_labels") or {}
    if not isinstance(labels_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in labels_raw.items()
    ):
        raise SkillError("step_labels must map tool ids to short label strings")
    unknown_labels = sorted(set(labels_raw) - set(tools))
    if unknown_labels:
        raise SkillError(f"step_labels names tools not declared in `tools`: {unknown_labels}")

    body = m.group(2).strip()
    if not body:
        raise SkillError("body is empty — a skill with no procedure teaches nothing")
    approval = _section(body, "Approval")
    ask = _section(body, "Ask")
    report_text = _section(body, "Report")

    return Skill(
        name=name,
        description=description,
        version=version,
        tools=tuple(tools),
        body=body,
        path=path,
        requires_role=role.strip() if isinstance(role, str) else None,
        guardrails=tuple(guardrails),
        compensations=compensations,
        approval=approval,
        ask=ask,
        report=report_text,
        step_labels={k: " ".join(v.split()) for k, v in labels_raw.items()},
    )


def _without_sections(body: str, headings: Tuple[str, ...]) -> str:
    out = body
    for h in headings:
        out = re.sub(rf"^##\s+{re.escape(h)}\s*$\n.*?(?=^##\s|\Z)", "", out, flags=re.M | re.S)
    return out.strip()


def _section(body: str, heading: str) -> str:
    """The text under `## <heading>` up to the next `## `, or ''."""
    m = re.search(rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s|\Z)", body, re.M | re.S)
    return m.group(1).strip() if m else ""


def validate_against_surface(
    skill: Skill, registry_ids: Iterable[str], allowed: Optional[Iterable[str]]
) -> Optional[str]:
    """Why this skill cannot run against the current tool surface, or None.

    ``allowed`` is None when no allowlist is in force (allow-all), mirroring
    ``allowed_tool_ids()``.
    """
    reg = set(registry_ids)
    not_in_registry = [t for t in skill.tools if t not in reg]
    if not_in_registry:
        return f"tools not in the registry: {not_in_registry}"
    if allowed is not None:
        hidden = [t for t in skill.tools if t not in set(allowed)]
        if hidden:
            return f"tools not allowlisted: {hidden}"
    undeclared = sorted(skill.body_tool_ids() - set(skill.tools))
    if undeclared:
        return f"body names tools the frontmatter does not declare: {undeclared}"
    return None


_cache_key: Optional[Tuple[Any, ...]] = None
_cache: Dict[str, Skill] = {}


def _dir_signature() -> Optional[Tuple[Tuple[str, float], ...]]:
    """Paths + mtimes of every SKILL.md, or None when there is no skills dir."""
    if not SKILLS_DIR.is_dir():
        return None
    sig = []
    for f in sorted(SKILLS_DIR.glob(f"*/{SKILL_FILE}")):
        try:
            sig.append((str(f), f.stat().st_mtime))
        except OSError:
            continue
    return tuple(sig)


def load_skills(
    *, registry_ids: Optional[Iterable[str]] = None, allowed: Optional[Iterable[str]] = None
) -> Dict[str, Skill]:
    """Every valid skill, keyed by name. Empty when disabled or absent.

    Cached on (file mtimes, allowlist contents, registry size) so both a skill
    edit and an allowlist edit re-validate on the next turn. Pass
    ``registry_ids``/``allowed`` from ``_registry`` to enforce the tool contract;
    omit them for a structure-only load (tests, tooling).
    """
    global _cache_key, _cache
    if not SKILLS_ENABLED:
        return {}
    sig = _dir_signature()
    if sig is None:
        return {}

    reg_key = None if registry_ids is None else len(set(registry_ids))
    allow_key = None if allowed is None else frozenset(allowed)
    key = (sig, reg_key, allow_key)
    if key == _cache_key:
        return _cache

    loaded: Dict[str, Skill] = {}
    for f in sorted(SKILLS_DIR.glob(f"*/{SKILL_FILE}")):
        try:
            skill = parse_skill_file(f)
            if registry_ids is not None:
                problem = validate_against_surface(skill, registry_ids, allowed)
                if problem:
                    raise SkillError(problem)
        except SkillError as exc:
            logger.error("Skill %s EXCLUDED: %s", f, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — a broken file must never take the planner down
            logger.exception("Skill %s EXCLUDED (unexpected): %s", f, exc)
            continue
        if skill.name in loaded:
            logger.error("Skill %s EXCLUDED: duplicate name %r", f, skill.name)
            continue
        loaded[skill.name] = skill

    _cache_key, _cache = key, loaded
    if loaded:
        logger.info("Loaded %d skill(s): %s", len(loaded), ", ".join(sorted(loaded)))
    return loaded


def skills_index_text(skills: Dict[str, Skill]) -> str:
    """The always-visible index: one ``- name: description`` line per skill."""
    if not skills:
        return ""
    return "\n".join(f"- {s.name}: {s.description}" for s in sorted(skills.values(), key=lambda s: s.name))


def parse_skill_directive(text: str) -> Optional[str]:
    """The planner's opt-in: ``SKILL: <name>`` as the FIRST non-empty line.

    Anywhere else it is just text — the planner is told to emit it alone, and
    a stray mention deeper in a plan must not hijack the turn.
    """
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = _SKILL_DIRECTIVE_RE.match(line)
        return m.group(1) if m else None
    return None


def strip_skill_directive(text: str) -> str:
    """Remove a leading ``SKILL:`` line so the rest can be parsed as a plan."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _SKILL_DIRECTIVE_RE.match(line):
            return "\n".join(lines[i + 1 :])
        break
    return text or ""
