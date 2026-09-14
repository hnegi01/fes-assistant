"""
Skill flow — plan from a procedure, approve once, execute with checkpoints and
compensation. Design: docs/design/skills.md §6–§9.

    chat       plan → execute → "what next?" → execute → …          (the reactive loop)
    migration  plan (ONE call) → ONE approval → execute in order      (migration_flow)
    skill      plan FROM A PROCEDURE (typed) → validate → ONE approval → runtime

The planner has already read the skill body (llm_agent._make_plan_detailed) and
produced a TYPED PLAN: tool ids, literal args, and `args_from` REFERENCES into
earlier results. This module owns everything after that:

- **validate** it before anything is shown: every tool inside the skill's own
  allowlist, every literal arg schema-valid, every reference pointing at an
  earlier step, every declared guardrail satisfied structurally.
- **gate** it once: a code-built dialog listing every mutating step, keyed on
  the canonical plan so editing a step re-gates, single use via
  `A._consume_approval` exactly like a lone mutation or a migration plan.
- **execute** it in code: resolve references at run time, expand loops from the
  live result, evaluate `when`, re-check guardrails with real values, invoke
  through `A._invoke_tool_traced`, checkpoint every result, and on the first
  failure run the skill's declared COMPENSATIONS in reverse.

The load-bearing idea is that data flows between steps by REFERENCE the runtime
resolves, never through the model. A fifteen-step skill therefore behaves
identically with summarization ON or OFF — where the reactive loop's dependent
chains block, because there the model has to read step 1 to write step 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import jsonschema

from . import llm_agent as A
from ._config import SKILL_MAX_STEPS, logger
from ._skills import Skill, load_skills
from .mcp_client import McpClient

# Synthetic tool_id for a whole-plan approval — never dispatched, exists so the
# plan can be keyed, stored and consumed by the same machinery a single
# mutation uses (`A._approval_key`). The UI matches on this string too.
PLAN_TOOL_ID = "skill.plan"


class PlanError(ValueError):
    """A plan that cannot be shown or run. The message says exactly why, in
    words meant for the user — plan errors are reported, never retried blind."""


# =============================================================================
# The path language — deliberately tiny (design §6)
# =============================================================================
_STEP_REF_RE = re.compile(r"^steps\[([A-Za-z0-9_-]+)\]\.result(.*)$")
_VAR_REF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$")
_TAIL_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\*|\d+)\]")
_WHEN_RE = re.compile(r"^\s*(.+?)\s*(==|!=)\s*(.+?)\s*$")


class Scope:
    """Results visible to a step: this scope's, then the enclosing one's.

    A `for_each` body runs in a child scope holding the loop variable and the
    sub-steps' results for THIS iteration; anything from before the loop is
    reached through the parent. Ids inside a loop are therefore per-iteration —
    `steps[5a]` in iteration 3 is iteration 3's copy.
    """

    def __init__(self, parent: Optional["Scope"] = None) -> None:
        self.parent = parent
        self.results: Dict[str, Any] = {}
        self.vars: Dict[str, Any] = {}

    def step_result(self, step_id: str) -> Any:
        if step_id in self.results:
            return self.results[step_id]
        if self.parent is not None:
            return self.parent.step_result(step_id)
        raise KeyError(step_id)

    def var(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.var(name)
        raise KeyError(name)


def _walk(value: Any, tail: str, expr: str) -> Any:
    """Apply `.key`, `[n]` and `[*]` tokens to a value."""
    pos = 0
    for m in _TAIL_TOKEN_RE.finditer(tail):
        if m.start() != pos:
            raise PlanError(f"cannot read reference {expr!r}: unexpected text at {tail[pos : m.start()]!r}")
        pos = m.end()
        key, idx = m.group(1), m.group(2)
        if key is not None:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                # `.key` over a list of records maps over it — what "[*].oid"
                # means after a for_each has already fanned out.
                value = [v.get(key) for v in value]
            else:
                raise PlanError(f"reference {expr!r}: no field {key!r} in the result")
        elif idx == "*":
            if not isinstance(value, list):
                raise PlanError(f"reference {expr!r}: [*] needs a list, got {type(value).__name__}")
        else:
            i = int(idx)
            if not isinstance(value, list) or i >= len(value):
                raise PlanError(f"reference {expr!r}: index {i} out of range")
            value = value[i]
    if pos != len(tail):
        raise PlanError(f"cannot read reference {expr!r}: unexpected text {tail[pos:]!r}")
    return value


def resolve_path(expr: str, scope: Scope) -> Any:
    """`steps[<id>].result...` or `<loopvar>...` → the value, or PlanError."""
    expr = (expr or "").strip()
    m = _STEP_REF_RE.match(expr)
    if m:
        step_id, tail = m.group(1), m.group(2)
        try:
            base = scope.step_result(step_id)
        except KeyError:
            raise PlanError(f"reference {expr!r}: step {step_id!r} has no result yet") from None
        return _walk(base, tail, expr)
    if expr.startswith("steps"):
        raise PlanError(f"malformed reference {expr!r} — expected steps[<id>].result...")
    m = _VAR_REF_RE.match(expr)
    if m:
        name, tail = m.group(1), m.group(2)
        try:
            base = scope.var(name)
        except KeyError:
            raise PlanError(f"reference {expr!r}: {name!r} is not a loop variable in scope") from None
        return _walk(base, tail, expr)
    raise PlanError(f"malformed reference {expr!r}")


def _referenced_step_ids(expr: str) -> Set[str]:
    m = _STEP_REF_RE.match((expr or "").strip())
    return {m.group(1)} if m else set()


def _referenced_var(expr: str) -> Optional[str]:
    e = (expr or "").strip()
    if _STEP_REF_RE.match(e):
        return None
    m = _VAR_REF_RE.match(e)
    return m.group(1) if m else None


def eval_when(expr: str, scope: Scope) -> bool:
    """`<path> == <json>`, `<path> != <json>`, or a bare truthy `<path>`."""
    m = _WHEN_RE.match(expr or "")
    if m:
        left, op, right = m.group(1), m.group(2), m.group(3)
        actual = resolve_path(left, scope)
        try:
            expected = json.loads(right)
        except json.JSONDecodeError:
            expected = right.strip("'\"")
        return (actual == expected) if op == "==" else (actual != expected)
    return bool(resolve_path(expr, scope))


# =============================================================================
# The plan model
# =============================================================================
@dataclass
class Step:
    id: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    args_from: Dict[str, str] = field(default_factory=dict)
    when: Optional[str] = None


@dataclass
class Loop:
    id: str
    over: str
    var: str
    steps: List[Step] = field(default_factory=list)


PlanItem = Union[Step, Loop]


def parse_plan(raw: Any) -> List[PlanItem]:
    """The planner's JSON → typed items. Structural only; validate_plan judges it."""
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise PlanError("the plan must be an object with a `steps` list")

    def _step(d: Dict[str, Any]) -> Step:
        if not isinstance(d, dict) or not isinstance(d.get("tool"), str):
            raise PlanError(f"a step needs a `tool`: {json.dumps(d)[:120]}")
        sid = str(d.get("id") or "").strip()
        if not sid:
            raise PlanError(f"step {d['tool']} has no id")
        args = d.get("args") or {}
        args_from = d.get("args_from") or {}
        if not isinstance(args, dict) or not isinstance(args_from, dict):
            raise PlanError(f"step {sid}: `args` and `args_from` must be objects")
        if not all(isinstance(v, str) for v in args_from.values()):
            raise PlanError(f"step {sid}: every `args_from` value must be a reference string")
        when = d.get("when")
        if when is not None and not isinstance(when, str):
            raise PlanError(f"step {sid}: `when` must be a string")
        return Step(id=sid, tool=d["tool"], args=dict(args), args_from=dict(args_from), when=when)

    items: List[PlanItem] = []
    for d in raw["steps"]:
        if isinstance(d, dict) and "for_each" in d:
            lid = str(d.get("id") or "").strip()
            if not lid:
                raise PlanError("a for_each block has no id")
            if not isinstance(d.get("for_each"), str) or not isinstance(d.get("as"), str):
                raise PlanError(f"loop {lid}: needs `for_each` (a reference) and `as` (a variable name)")
            subs = d.get("steps")
            if not isinstance(subs, list) or not subs:
                raise PlanError(f"loop {lid}: `steps` must be a non-empty list")
            for sub in subs:
                if isinstance(sub, dict) and "for_each" in sub:
                    raise PlanError(f"loop {lid}: nested loops are not supported")
            items.append(Loop(id=lid, over=d["for_each"], var=d["as"], steps=[_step(s) for s in subs]))
        else:
            items.append(_step(d))
    return items


def _iter_steps(items: Iterable[PlanItem]) -> Iterable[Tuple[Step, Optional[Loop]]]:
    for it in items:
        if isinstance(it, Loop):
            for s in it.steps:
                yield s, it
        else:
            yield it, None


def _tool_by_id(items: Iterable[PlanItem]) -> Dict[str, str]:
    return {s.id: s.tool for s, _ in _iter_steps(items)}


def _mutates(tool_id: str) -> bool:
    return bool((A.TOOL_REGISTRY.get(tool_id) or {}).get("mutates"))


def has_mutation(items: Iterable[PlanItem]) -> bool:
    return any(_mutates(s.tool) for s, _ in _iter_steps(items))


# -----------------------------------------------------------------------------
# Guardrails — structural check at plan time; re-checked with values at run time
# -----------------------------------------------------------------------------
def _derived_from_tool(step: Step, param: str, items: List[PlanItem], loop: Optional[Loop], tool_suffix: str) -> bool:
    """Does `step.args_from[param]` point at a result produced by a step whose
    tool ends with `tool_suffix`, or at a loop variable? (A loop variable is an
    item of a listing — never something this run created.)"""
    ref = step.args_from.get(param)
    if not ref:
        return False
    if _referenced_var(ref):
        return False
    tools = _tool_by_id(items)
    return any(tools.get(sid, "").endswith(tool_suffix) for sid in _referenced_step_ids(ref))


def guardrail_validate_before_swap(items: List[PlanItem]) -> Optional[str]:
    """`replace_datasource` on a dashboard this run did NOT create must be
    conditional on a `validate_dashboard_queries` result. A stage copy (made by
    `duplicate_dashboard` in this plan) is exempt — it exists to be tested."""
    tools = _tool_by_id(items)
    for step, loop in _iter_steps(items):
        if not step.tool.endswith(".replace_datasource"):
            continue
        if _derived_from_tool(step, "dashboard", items, loop, ".duplicate_dashboard"):
            continue  # swapping the copy — that IS the test
        refs = _referenced_step_ids(step.when or "")
        if not any(tools.get(r, "").endswith(".validate_dashboard_queries") for r in refs):
            return (
                f"step {step.id} swaps a dashboard this run did not create without a `when` on a "
                "validate_dashboard_queries result — the procedure requires validation first"
            )
    return None


_GUARDRAIL_CHECKS = {"validate-before-swap": guardrail_validate_before_swap}


def validate_plan(items: List[PlanItem], skill: Skill, max_steps: int = SKILL_MAX_STEPS) -> None:
    """Everything that can be judged before a value exists. Raises PlanError."""
    if not items:
        raise PlanError("the plan has no steps")
    seen: Set[str] = set()
    allowed = set(skill.tools)
    top_ids: List[str] = []

    def _check_refs(step: Step, visible: Set[str], loop: Optional[Loop]) -> None:
        for param, ref in step.args_from.items():
            var = _referenced_var(ref)
            if var is not None:
                if loop is None or var != loop.var:
                    raise PlanError(f"step {step.id}: {ref!r} uses a loop variable that is not in scope")
                continue
            for rid in _referenced_step_ids(ref):
                if rid not in visible:
                    raise PlanError(f"step {step.id}: `{param}` references step {rid!r}, which does not run before it")
            if not _referenced_step_ids(ref):
                raise PlanError(f"step {step.id}: malformed reference {ref!r}")
        if step.when:
            m = _WHEN_RE.match(step.when)
            path = m.group(1) if m else step.when
            var = _referenced_var(path)
            if var is not None and (loop is None or var != loop.var):
                raise PlanError(f"step {step.id}: `when` uses a variable not in scope")
            for rid in _referenced_step_ids(path):
                if rid not in visible:
                    raise PlanError(f"step {step.id}: `when` references step {rid!r}, which does not run before it")

    def _check_step(step: Step, visible: Set[str], loop: Optional[Loop]) -> None:
        if step.id in seen:
            raise PlanError(f"duplicate step id {step.id!r}")
        seen.add(step.id)
        if step.tool not in allowed:
            raise PlanError(
                f"step {step.id} uses `{step.tool}`, which the procedure '{skill.name}' does not allow "
                f"(allowed: {', '.join(sorted(allowed))})"
            )
        meta = A.TOOL_REGISTRY.get(step.tool)
        if not meta:
            raise PlanError(f"step {step.id}: `{step.tool}` is not an exposed tool")
        schema = meta.get("parameters") or {}
        props = schema.get("properties") or {}
        for param in list(step.args) + list(step.args_from):
            if props and param not in props:
                raise PlanError(f"step {step.id}: `{step.tool}` has no parameter `{param}`")
        overlap = set(step.args) & set(step.args_from)
        if overlap:
            raise PlanError(f"step {step.id}: {sorted(overlap)} given both literally and by reference")
        # Literal args must be valid on their own; required params supplied by
        # reference are checked when the value exists.
        partial = {
            **{k: v for k, v in schema.items() if k not in ("required",)},
            "required": [r for r in (schema.get("required") or []) if r not in step.args_from],
        }
        try:
            A._validate_tool_args(partial, step.args)
        except jsonschema.ValidationError as ve:
            missing = A._missing_required_fields(step.args, partial)
            if missing:
                raise PlanError(
                    f"step {step.id} (`{step.tool}`) is missing required value(s): {', '.join(missing)}"
                ) from ve
            raise PlanError(f"step {step.id} (`{step.tool}`): {ve.message}") from ve
        _check_refs(step, visible, loop)

    count = 0
    for it in items:
        if isinstance(it, Loop):
            if it.id in seen:
                raise PlanError(f"duplicate step id {it.id!r}")
            seen.add(it.id)
            if not _referenced_step_ids(it.over) or _referenced_var(it.over):
                raise PlanError(f"loop {it.id}: `for_each` must reference an earlier step's result")
            for rid in _referenced_step_ids(it.over):
                if rid not in top_ids:
                    raise PlanError(f"loop {it.id}: `for_each` references step {rid!r}, which does not run before it")
            inner: List[str] = []
            for s in it.steps:
                _check_step(s, set(top_ids) | set(inner), it)
                inner.append(s.id)
                count += 1
            top_ids.append(it.id)
        else:
            _check_step(it, set(top_ids), None)
            top_ids.append(it.id)
            count += 1
    if count > max_steps:
        raise PlanError(f"the plan has {count} steps before any loop expands; the ceiling is {max_steps}")

    for g in skill.guardrails:
        check = _GUARDRAIL_CHECKS.get(str(g.get("id")))
        if check is None:
            raise PlanError(f"guardrail {g.get('id')!r} is not enforceable by this runtime")
        problem = check(items)
        if problem:
            raise PlanError(f"guardrail '{g['id']}': {problem}")


# -----------------------------------------------------------------------------
# Rendering — the approval payload, the UI plan text, the dialog
# -----------------------------------------------------------------------------
def _step_payload(s: Step) -> Dict[str, Any]:
    d: Dict[str, Any] = {"id": s.id, "tool": s.tool, "args": A._scrub_secrets(s.args)}
    if s.args_from:
        d["args_from"] = dict(s.args_from)
    if s.when:
        d["when"] = s.when
    return d


def plan_arguments(items: List[PlanItem], skill: Skill) -> Dict[str, Any]:
    """The approval payload — and therefore the approval KEY. Change a step, an
    argument, a reference, or the order, and the plan must be approved again."""
    steps = []
    for it in items:
        if isinstance(it, Loop):
            steps.append(
                {"id": it.id, "for_each": it.over, "as": it.var, "steps": [_step_payload(s) for s in it.steps]}
            )
        else:
            steps.append(_step_payload(it))
    return {"skill": skill.name, "skill_version": skill.version, "steps": steps}


def _label(tool_id: str) -> str:
    desc = ((A.TOOL_REGISTRY.get(tool_id) or {}).get("description") or "").strip().splitlines()
    text = (desc[0].strip().rstrip(" .") if desc else "") or tool_id.split(".", 1)[-1].replace("_", " ")
    return text[:1].upper() + text[1:]


def _humanise(args: Dict[str, Any], args_from: Dict[str, str]) -> str:
    parts = []
    for k, v in args.items():
        shown = (
            ", ".join(map(str, v)) if isinstance(v, list) else ("yes" if v is True else "no" if v is False else str(v))
        )
        parts.append(f"{k.replace('_', ' ')}: {shown}")
    for k, ref in args_from.items():
        m = _STEP_REF_RE.match(ref.strip())
        src = f"step {m.group(1)}'s result" if m else f"each {ref.split('.', 1)[0]}"
        parts.append(f"{k.replace('_', ' ')}: from {src}")
    return " · ".join(parts)


def render_plan_lines(items: List[PlanItem]) -> List[str]:
    """Prose lines for the UI's plan block and the transcript."""
    lines: List[str] = []
    for it in items:
        if isinstance(it, Loop):
            m = _STEP_REF_RE.match(it.over.strip())
            lines.append(f"For each item from step {m.group(1) if m else '?'}:")
            for s in it.steps:
                lines.append(f"  {s.id}. {_label(s.tool)} ({s.tool})" + (f" — only if {s.when}" if s.when else ""))
        else:
            lines.append(f"{_label(it.tool)} ({it.tool})" + (f" — only if {it.when}" if it.when else ""))
    return lines


def render_dialog(items: List[PlanItem], skill: Skill) -> str:
    """The approval dialog, built entirely in code — zero LLM calls.

    Lists every step that WRITES, in order, with its arguments — literal ones as
    values, derived ones as where they come from. Reads are summarised in one
    line so the reader sees the shape without wading through them. Built from
    the plan that will run, never from the model's description of it."""
    writes: List[str] = []
    reads = 0
    for it in items:
        if isinstance(it, Loop):
            m = _STEP_REF_RE.match(it.over.strip())
            head = f"For each dashboard found in step {m.group(1) if m else '?'}:"
            block = []
            for s in it.steps:
                if _mutates(s.tool):
                    line = f"  - **{_label(s.tool)}** — {_humanise(s.args, s.args_from)}"
                    if s.when:
                        line += f" *(only if {s.when})*"
                    block.append(line)
                else:
                    reads += 1
            if block:
                writes.append(head)
                writes.extend(block)
        elif _mutates(it.tool):
            line = f"{len(writes) + 1}. **{_label(it.tool)}** — {_humanise(it.args, it.args_from)}"
            if it.when:
                line += f" *(only if {it.when})*"
            writes.append(line)
        else:
            reads += 1

    lines = [
        f"This follows the procedure **{skill.name}** (v{skill.version}). "
        "Approving covers every operation below, in this order — nothing runs until you do.",
        "",
    ]
    lines.extend(writes)
    if reads:
        lines.append("")
        lines.append(
            f"*Plus {reads} read-only step{'s' if reads != 1 else ''} the procedure uses to decide and verify.*"
        )
    if skill.compensations:
        lines.append("")
        lines.append(
            "If a step fails, the run stops and undoes what it created in this run "
            f"({', '.join(sorted(_label(t) for t in skill.compensations))}); the report says what ran, "
            "what failed, and what was reverted."
        )
    lines.append("")
    lines.append("Approve to run the whole sequence, or cancel and ask again with any changes.")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Role gate — courtesy, not enforcement (the token enforces)
# -----------------------------------------------------------------------------
_ROLE_RANK = {
    "viewer": 1,
    "consumer": 1,
    "viewerPlus": 2,
    "dashboardDesigner": 3,
    "contributor": 3,
    "designer": 3,
    "dataDesigner": 4,
    "dataAdmin": 5,
    "tenantAdmin": 6,
    "admin": 7,
    "sysAdmin": 8,
    "super": 8,
}


def role_satisfies(user_role: Optional[str], required: Optional[str]) -> Optional[bool]:
    """True/False, or None when either role is unknown to us (then don't block)."""
    if not required:
        return True
    need = _ROLE_RANK.get(str(required))
    have = _ROLE_RANK.get(str(user_role or ""))
    if need is None or have is None:
        return None
    return have >= need


async def _current_role(mcp_client: McpClient, mode: str) -> Optional[str]:
    if "access_management.get_my_user" not in A.TOOL_REGISTRY:
        return None
    try:
        res = await A._invoke_tool_traced(mcp_client, "access_management.get_my_user", {}, mode)
    except Exception as exc:  # noqa: BLE001 — a failed courtesy check must never block a turn
        logger.warning("Skill role check: get_my_user failed (%s); continuing without it.", exc)
        return None
    payload = (res or {}).get("result") if isinstance(res, dict) else None
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        for key in ("ROLE_NAME", "role_name", "roleName", "role"):
            if payload.get(key):
                return str(payload[key])
    return None


def _drop_mutations(items: List[PlanItem]) -> List[PlanItem]:
    """What a user without the role can still run: the reads, in order."""
    kept: List[PlanItem] = []
    for it in items:
        if isinstance(it, Loop):
            subs = [s for s in it.steps if not _mutates(s.tool)]
            if subs:
                kept.append(Loop(id=it.id, over=it.over, var=it.var, steps=subs))
        elif not _mutates(it.tool):
            kept.append(it)
    return kept


# =============================================================================
# Execution
# =============================================================================
def _payload_of(result: Any) -> Any:
    return result.get("result") if isinstance(result, dict) else result


def _count_of(payload: Any) -> Optional[int]:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for k in ("results", "items", "rows"):
            if isinstance(payload.get(k), list):
                return len(payload[k])
    return None


def _outcome_line(tool_id: str, ok: bool, payload: Any) -> str:
    label = _label(tool_id)
    if not ok:
        return f"{label} — failed"
    n = _count_of(payload)
    return f"{label} — {n} row{'s' if n != 1 else ''}" if n is not None else f"{label} — done"


def _ids_in(payload: Any) -> Set[str]:
    """Identifiers a create-shaped result reports, for the created-by-this-run set."""
    out: Set[str] = set()
    if isinstance(payload, dict):
        for k in ("oid", "_id", "id", "dashboard_id", "target_id"):
            v = payload.get(k)
            if isinstance(v, str) and v:
                out.add(v)
    return out


def _render_template(value: Any, step_args: Dict[str, Any], payload: Any) -> Any:
    """`{args.x}` / `{result.y}` placeholders in a compensation's args."""
    if not isinstance(value, str):
        return value
    m = re.fullmatch(r"\{(args|result)\.([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())
    if not m:
        return value
    src = step_args if m.group(1) == "args" else (payload if isinstance(payload, dict) else {})
    return src.get(m.group(2)) if isinstance(src, dict) else None


@dataclass
class _Done:
    step_id: str
    tool: str
    args: Dict[str, Any]
    payload: Any


@dataclass
class _Report:
    ran: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: Optional[Tuple[str, str]] = None
    not_attempted: List[str] = field(default_factory=list)
    compensated: List[str] = field(default_factory=list)
    compensation_failed: Optional[Tuple[str, str]] = None
    handoff: Optional[str] = None
    outcome: str = "ok"

    def text(self, body: str) -> str:
        lines: List[str] = []
        if self.handoff:
            lines.append(self.handoff)
        if self.failed:
            tool, reason = self.failed
            lines.append(f"**Stopped** — `{tool}` failed: {reason}")
        if self.ran:
            lines.append("**Completed:** " + ", ".join(f"`{t}`" for t in self.ran))
        if self.skipped:
            lines.append("**Skipped (condition not met):** " + ", ".join(f"`{t}`" for t in self.skipped))
        if self.not_attempted:
            lines.append("**Not attempted:** " + ", ".join(f"`{t}`" for t in self.not_attempted))
        if self.compensated:
            lines.append("**Reverted:** " + ", ".join(f"`{t}`" for t in self.compensated))
        if self.compensation_failed:
            tool, reason = self.compensation_failed
            lines.append(
                f"**Could not revert** `{tool}`: {reason} — stopped unwinding; "
                "what remains is listed above and needs a look."
            )
        head = "\n\n".join(lines)
        return f"{head}\n\n{body}" if body else head


async def _execute(
    items: List[PlanItem],
    skill: Skill,
    *,
    mcp_client: McpClient,
    mode: str,
    summ_on: bool,
    transcript: List[Dict[str, Any]],
    raw_results: List[Tuple[str, Any]],
    steps_executed: int,
    trace: Dict[str, Any],
    report: _Report,
) -> Tuple[_Report, int]:
    root = Scope()
    done: List[_Done] = []
    created: Set[str] = set()
    # Stage copies this run created AND already deleted in-plan (a later
    # cleanup step). Compensating their creation again would 404 and, worse,
    # stop the unwind before anything after them is reverted.
    removed: Set[str] = set()
    all_ids = [s.id for s, _ in _iter_steps(items)]

    async def _run_step(step: Step, scope: Scope, loop_pos: Optional[Tuple[int, int]]) -> bool:
        nonlocal steps_executed
        if steps_executed >= SKILL_MAX_STEPS:
            raise PlanError(f"skill step ceiling reached ({SKILL_MAX_STEPS}); the rest was not attempted")

        # Resolve references and the condition with real values.
        args = dict(step.args)
        for param, ref in step.args_from.items():
            args[param] = resolve_path(ref, scope)
        if step.when and not eval_when(step.when, scope):
            report.skipped.append(step.tool)
            logger.info("Skill step %s (%s) skipped: `%s` is false", step.id, step.tool, step.when)
            return True

        # Runtime guardrail with values: swapping a dashboard this run did NOT
        # create requires a passed validation — the plan-time check proved a
        # `when` exists; this makes sure nothing slipped past it.
        if step.tool.endswith(".replace_datasource"):
            target = str(args.get("dashboard") or "")
            if target and target not in created and not step.when:
                raise PlanError(f"step {step.id} would swap dashboard {target!r} without a validation gate")

        meta = A.TOOL_REGISTRY.get(step.tool) or {}
        try:
            A._validate_tool_args(meta.get("parameters") or {}, args)
        except jsonschema.ValidationError as ve:
            raise PlanError(f"step {step.id} (`{step.tool}`) resolved to invalid arguments: {ve.message}") from ve

        n = steps_executed + 1
        if _mutates(step.tool):
            A.audit_logger.info(
                "EXECUTING mutation (skill %s) tool=%s args=%s",
                skill.name,
                step.tool,
                json.dumps(A._scrub_secrets(args), ensure_ascii=False),
            )
        ev: Dict[str, Any] = {"phase": "executing", "step": n, "max_steps": SKILL_MAX_STEPS, "tool_id": step.tool}
        if loop_pos:
            ev["loop_index"], ev["loop_total"] = loop_pos
        await A._emit_agent_progress(ev)

        result = await A._invoke_tool_traced(mcp_client, step.tool, args, mode)
        ok = A._effective_ok(result)
        payload = _payload_of(result)
        A._record_tool_result(result)
        A._record_step(n, step.tool, result)
        raw_results.append((step.tool, result))
        transcript.extend(
            A._transcript_step(
                {"function": {"name": step.tool, "arguments": json.dumps(args, default=str)}},
                step.tool,
                result,
                summ_on,
            )
        )
        steps_executed += 1
        trace["tool_selected"] = trace.get("tool_selected") or step.tool
        await A._emit_agent_progress(
            {
                "phase": "completed",
                "step": n,
                "max_steps": SKILL_MAX_STEPS,
                "tool_id": step.tool,
                "ok": ok,
                "outcome": _outcome_line(step.tool, ok, payload),
                **({"loop_index": loop_pos[0], "loop_total": loop_pos[1]} if loop_pos else {}),
            }
        )
        if not ok:
            reason = str(
                (result or {}).get("error") or A._payload_failure_reason(payload) or "no reason reported"
            ).strip()
            report.failed = (step.tool, reason)
            return False

        scope.results[step.id] = payload
        report.ran.append(step.tool)
        if _mutates(step.tool):
            done.append(_Done(step.id, step.tool, args, payload))
            if step.tool.endswith(".duplicate_dashboard"):
                created.update(_ids_in(payload))
            elif step.tool.endswith(".delete_dashboard"):
                gone = {str(v) for k, v in args.items() if k in ("dashboard_id", "dashboard", "oid") and v}
                removed.update(gone & created)
        return True

    async def _compensate() -> None:
        for d in reversed(done):
            spec = skill.compensations.get(d.tool)
            if not spec:
                continue
            # Swapping a copy back is pointless when the copy is about to be
            # deleted — and it is what "covered by deleting the copy" means.
            if d.tool.endswith(".replace_datasource") and str(d.args.get("dashboard") or "") in created:
                continue
            # The copy was already deleted by the plan itself — nothing to undo.
            if d.tool.endswith(".duplicate_dashboard") and (_ids_in(d.payload) & removed):
                continue
            cargs = {k: _render_template(v, d.args, d.payload) for k, v in (spec.get("args") or {}).items()}
            if any(v is None for v in cargs.values()):
                report.compensation_failed = (spec["tool"], "the result did not carry the values needed to undo it")
                logger.error("Skill compensation %s for %s: unresolvable args %s", spec["tool"], d.tool, cargs)
                return
            A.audit_logger.info(
                "EXECUTING compensation (skill %s) tool=%s args=%s",
                skill.name,
                spec["tool"],
                json.dumps(A._scrub_secrets(cargs), ensure_ascii=False),
            )
            try:
                res = await A._invoke_tool_traced(mcp_client, spec["tool"], cargs, mode)
            except Exception as exc:  # noqa: BLE001
                report.compensation_failed = (spec["tool"], str(exc)[:200])
                return
            raw_results.append((spec["tool"], res))
            if not A._effective_ok(res):
                reason = str((res or {}).get("error") or A._payload_failure_reason(_payload_of(res)) or "failed")
                report.compensation_failed = (spec["tool"], reason[:200])
                logger.error("Skill compensation %s failed: %s — stopping the unwind", spec["tool"], reason)
                return
            report.compensated.append(spec["tool"])

    try:
        for it in items:
            if isinstance(it, Loop):
                seq = resolve_path(it.over, root)
                if not isinstance(seq, list):
                    raise PlanError(f"loop {it.id}: `{it.over}` did not resolve to a list")
                for i, item in enumerate(seq):
                    child = Scope(root)
                    child.vars[it.var] = item
                    for s in it.steps:
                        if not await _run_step(s, child, (i + 1, len(seq))):
                            raise _Stop(s.id)
                root.results[it.id] = seq
            else:
                if not await _run_step(it, root, None):
                    raise _Stop(it.id)
    except _Stop as stop:
        idx = all_ids.index(stop.step_id) if stop.step_id in all_ids else len(all_ids)
        report.not_attempted = [s.tool for s, _ in list(_iter_steps(items))[idx + 1 :]]
        report.outcome = "skill_failed"
        await _compensate()
    except PlanError as pe:
        report.failed = report.failed or ("plan", str(pe))
        report.outcome = "skill_failed"
        await _compensate()
    return report, steps_executed


class _Stop(Exception):
    def __init__(self, step_id: str) -> None:
        super().__init__(step_id)
        self.step_id = step_id


# =============================================================================
# The turn
# =============================================================================
async def run(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    planning_context: str,
    mode: str,
    passed_tools: List[Dict[str, Any]],
    user_text: str,
    mcp_client: McpClient,
    approved_mutations: Set[Tuple[str, str]],
    summ_on: bool,
    turn_trace_id: str,
    trace: Dict[str, Any],
    transcript: Optional[List[Dict[str, Any]]] = None,
    raw_results: Optional[List[Tuple[str, Any]]] = None,
    steps_executed: int = 0,
    skill: Optional[Skill] = None,
    plan: Optional[Dict[str, Any]] = None,
    pending_plan: Optional[Dict[str, Any]] = None,
    **_ignored: Any,
) -> str:
    """One skill turn: validate → (gate) → execute → report. Pauses via pending_loop."""
    transcript = transcript if transcript is not None else []
    raw_results = raw_results if raw_results is not None else []
    report = _Report()

    def _finish(outcome: str, reply: str) -> str:
        trace["outcome"] = outcome
        trace["agent_steps"] = steps_executed
        trace["skill"] = skill.name if skill else (pending_plan or {}).get("skill", {}).get("name")
        # Name the procedure in the turn's response (`skill` field) — fresh
        # and resume alike, and only once skill_flow actually owns the turn,
        # so a planner that named a skill and then fell back never claims one.
        _out = A.turn_output()
        if _out is not None and skill is not None:
            _out["skill"] = {"name": skill.name, "version": skill.version}
        A._write_llm_trace(trace)
        return reply

    # ---------------------------------------------------------------- resume
    if pending_plan:
        meta = pending_plan.get("skill") or {}
        skills = load_skills(registry_ids=A.all_registry_tool_ids(), allowed=A.exposed_tool_ids())
        skill = skills.get(str(meta.get("name") or ""))
        if skill is None:
            return _finish(
                "skill_missing",
                f"The procedure `{meta.get('name')}` this plan came from is no longer available, so I did not run it.",
            )
        if skill.version != meta.get("version"):
            return _finish(
                "skill_changed",
                f"The procedure `{skill.name}` changed (v{meta.get('version')} → v{skill.version}) after this plan was "
                "proposed, so I did not run the old plan. Ask again and I will plan from the current version.",
            )
        if not A._consume_approval(approved_mutations, PLAN_TOOL_ID, pending_plan.get("plan_arguments") or {}):
            logger.info("Dropping paused skill plan (no matching approval this turn).")
            return _finish(
                "plan_dropped", "That plan was not approved, so nothing ran. Ask again if you want to proceed."
            )
        try:
            items = parse_plan(pending_plan.get("plan") or {})
        except PlanError as pe:
            return _finish("plan_invalid", f"I could not run the approved plan: {pe}")
        logger.info("Resuming approved skill plan '%s' (%d item(s)).", skill.name, len(items))
    else:
        # ------------------------------------------------------------ fresh
        assert skill is not None and plan is not None, "run() needs skill+plan or pending_plan"
        try:
            items = parse_plan(plan)
            if not items:
                return _finish("no_plan", "The procedure did not apply to this request after all — nothing planned.")
            validate_plan(items, skill)
        except PlanError as pe:
            logger.warning("Skill '%s' plan rejected: %s", skill.name, pe)
            return _finish("plan_invalid", f"I could not build a safe plan from the procedure `{skill.name}`: {pe}")

        # Courtesy role check BEFORE showing a plan whose first write would fail.
        if skill.requires_role and has_mutation(items) and mode == "chat":
            role = await _current_role(mcp_client, mode)
            ok = role_satisfies(role, skill.requires_role)
            if ok is False:
                reads = _drop_mutations(items)
                report.handoff = (
                    f"Making these changes needs the **{skill.requires_role}** role; your token has **{role}**. "
                    + ("I ran the read-only part so you can hand the findings to someone who can." if reads else "")
                )
                if not reads:
                    return _finish("role_insufficient", report.handoff)
                items = reads

        plan_text = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(render_plan_lines(items)))
        transcript.append({"role": "assistant", "content": f"PLAN ({skill.name}):\n{plan_text}"})
        await A._emit_agent_progress(
            {"phase": "planned", "step": 1, "max_steps": SKILL_MAX_STEPS, "plan": plan_text, "skill": skill.name}
        )

        # ------------------------------------------------------------ gate
        if has_mutation(items) and A.REQUIRE_MUTATION_CONFIRM:
            plan_args = plan_arguments(items, skill)
            if not A._consume_approval(approved_mutations, PLAN_TOOL_ID, plan_args):
                explanation = render_dialog(items, skill)
                A._record_tool_result(
                    {
                        "ok": False,
                        "pending_confirmation": {
                            "tool_id": PLAN_TOOL_ID,
                            "arguments": plan_args,
                            "reason": explanation,
                        },
                    }
                )
                A._record_pending_loop(
                    {
                        "transcript": transcript,
                        "raw_results": raw_results,
                        "steps_executed": steps_executed,
                        "tool_id": PLAN_TOOL_ID,
                        "arguments": plan_args,
                        "plan": {"steps": plan_args["steps"]},
                        "plan_arguments": plan_args,
                        "skill": {"name": skill.name, "version": skill.version},
                    }
                )
                logger.info("Skill plan '%s' awaiting approval (%d item(s)).", skill.name, len(items))
                return _finish("pending_mutation", explanation)

    # --------------------------------------------------------------- execute
    report, steps_executed = await _execute(
        items,
        skill,
        mcp_client=mcp_client,
        mode=mode,
        summ_on=summ_on,
        transcript=transcript,
        raw_results=raw_results,
        steps_executed=steps_executed,
        trace=trace,
        report=report,
    )
    # Code-built in BOTH summarization modes, like migration: a screen that
    # says what was WRITTEN must be deterministic.
    body = A._describe_results_local(raw_results)
    return _finish(report.outcome, report.text(body))
