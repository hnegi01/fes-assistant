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
_STEP_REF_RE = re.compile(r"^steps\[([A-Za-z0-9_-]+)\]\.(result|args)(.*)$")
_VAR_REF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$")
_TAIL_TOKEN_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\*|\d+)\]")
_WHEN_RE = re.compile(r"^\s*(.+?)\s*(==|!=)\s*(.+?)\s*$")


# A step whose `when` was false leaves this marker as its result. A later
# `when` that reads it is false too; a later `args_from` that needs it stops the
# run — a value that was never produced cannot be passed on. JSON-safe so a
# paused plan's scope survives the approval round trip.
SKIPPED: Dict[str, bool] = {"__skipped__": True}


def is_skipped(value: Any) -> bool:
    return isinstance(value, dict) and value.get("__skipped__") is True


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
        self.args: Dict[str, Dict[str, Any]] = {}  # the arguments each step actually ran with
        self.vars: Dict[str, Any] = {}

    def step_result(self, step_id: str) -> Any:
        if step_id in self.results:
            return self.results[step_id]
        if self.parent is not None:
            return self.parent.step_result(step_id)
        raise KeyError(step_id)

    def step_args(self, step_id: str) -> Dict[str, Any]:
        if step_id in self.args:
            return self.args[step_id]
        if self.parent is not None:
            return self.parent.step_args(step_id)
        raise KeyError(step_id)

    def var(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.var(name)
        raise KeyError(name)

    def snapshot(self) -> Dict[str, Any]:
        """This scope's own results and args, for a paused plan (root only)."""
        return {"results": dict(self.results), "args": dict(self.args)}

    @classmethod
    def restore(cls, snap: Optional[Dict[str, Any]]) -> "Scope":
        sc = cls()
        if isinstance(snap, dict):
            sc.results = dict(snap.get("results") or {})
            sc.args = dict(snap.get("args") or {})
        return sc


def _labeller(skill: Skill) -> Any:
    """tool_id → the skill's label for it, else the tool description's first line."""
    return lambda t: skill.step_labels.get(t) or _label(t)


def scope_var_name(scope: Scope) -> Optional[str]:
    """The loop variable a child scope carries, for progress lines ("dashboard 2 of 4")."""
    return next(iter(scope.vars), None) if scope.vars else None


def _walk(value: Any, tail: str, expr: str, *, missing_ok: bool = False) -> Any:
    """Apply `.key`, `[n]` and `[*]` tokens to a value. With `missing_ok`, a key
    that is not there yields None instead of an error — conditions read
    optional keys (a warning count that is absent when there is no warning)."""
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
            elif missing_ok:
                return None
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


_TERNARY_RE = re.compile(r"^(.+?)\s+\?\s+(.+?)\s+:\s+(.+)$")


def _subexprs(expr: str) -> List[str]:
    """The plain references inside an expression: a ternary's condition path
    and both branches; otherwise the expression itself."""
    e = (expr or "").strip()
    t = _TERNARY_RE.match(e)
    if not t:
        return [e]
    cond = t.group(1).strip()
    w = _WHEN_RE.match(cond)
    return [w.group(1).strip() if w else cond, t.group(2).strip(), t.group(3).strip()]


def resolve_path(expr: str, scope: Scope, *, skipped_ok: bool = False, missing_ok: bool = False) -> Any:
    """`steps[<id>].result...`, `steps[<id>].args...` or `<loopvar>...` → the value, or PlanError.

    An expression may also CHOOSE: `<condition> ? <path-if-true> : <path-if-false>`,
    the condition in `when` syntax. This is how a plan takes the safe superset
    of tables when the analysis flags an ambiguous join, and the exact list
    otherwise — decided by code from the result, never by the model.

    `.args` reads back what an earlier step RAN WITH — the perspective name the
    plan chose in the create step feeds the build step — so a value is written
    once and cannot drift between steps. A reference into a SKIPPED step returns
    the SKIPPED marker when `skipped_ok` (conditions), else it is an error: an
    argument cannot come from a step that never ran.
    """
    expr = (expr or "").strip()
    t = _TERNARY_RE.match(expr)
    if t:
        chosen = t.group(2) if eval_when(t.group(1).strip(), scope) else t.group(3)
        return resolve_path(chosen.strip(), scope, skipped_ok=skipped_ok, missing_ok=missing_ok)
    m = _STEP_REF_RE.match(expr)
    if m:
        step_id, kind, tail = m.group(1), m.group(2), m.group(3)
        try:
            base = scope.step_args(step_id) if kind == "args" else scope.step_result(step_id)
        except KeyError:
            try:
                probe = scope.step_result(step_id)
            except KeyError:
                raise PlanError(f"reference {expr!r}: step {step_id!r} has not run yet") from None
            if not is_skipped(probe):
                raise PlanError(f"reference {expr!r}: step {step_id!r} has not run yet") from None
            base = probe  # skipped: reported below
        if is_skipped(base):
            if skipped_ok:
                return SKIPPED
            raise PlanError(f"step {step_id!r} was skipped (its condition was not met), so {expr!r} has no value")
        return _walk(base, tail, expr, missing_ok=missing_ok)
    if expr.startswith("steps"):
        raise PlanError(f"malformed reference {expr!r} — expected steps[<id>].result... or steps[<id>].args...")
    m = _VAR_REF_RE.match(expr)
    if m:
        name, tail = m.group(1), m.group(2)
        try:
            base = scope.var(name)
        except KeyError:
            raise PlanError(f"reference {expr!r}: {name!r} is not a loop variable in scope") from None
        return _walk(base, tail, expr, missing_ok=missing_ok)
    raise PlanError(f"malformed reference {expr!r}")


def _referenced_step_ids(expr: str) -> Set[str]:
    out: Set[str] = set()
    for sub in _subexprs(expr):
        m = _STEP_REF_RE.match(sub)
        if m:
            out.add(m.group(1))
    return out


def _referenced_var(expr: str) -> Optional[str]:
    e = (expr or "").strip()
    if _TERNARY_RE.match(e):
        for sub in _subexprs(e):
            v = _referenced_var(sub)
            if v is not None:
                return v
        return None
    if _STEP_REF_RE.match(e) or e.startswith("steps"):
        # A `steps...` path is never a loop variable — a malformed one must be
        # reported as malformed, not as "variable not in scope".
        return None
    m = _VAR_REF_RE.match(e)
    return m.group(1) if m else None


def _split_when(expr: str) -> List[str]:
    """The conjuncts of a condition: `a == [] && b != null` → two clauses."""
    return [c.strip() for c in (expr or "").split("&&") if c.strip()]


def parse_when(expr: str) -> List[Tuple[str, Optional[str], Any]]:
    """A condition → [(path, op, expected)], op None for a bare truthy path.

    Strict: the right-hand side must be JSON (`[]`, `null`, `true`, `0`,
    `"text"`). A clause that does not parse raises, so a plan with a condition
    the runtime cannot judge is rejected at validation instead of being
    evaluated to false by accident (live 2026-09-16: an `&&` the grammar did
    not know was swallowed into a string comparison that was always false).
    """
    clauses = _split_when(expr)
    if not clauses:
        raise PlanError("empty condition")
    out: List[Tuple[str, Optional[str], Any]] = []
    for c in clauses:
        m = _WHEN_RE.match(c)
        if not m:
            out.append((c, None, None))
            continue
        left, op, right = m.group(1).strip(), m.group(2), m.group(3).strip()
        try:
            expected = json.loads(right)
        except json.JSONDecodeError:
            raise PlanError(
                f"condition {c!r} not understood: the value after {op} must be JSON "
                '(`[]`, `null`, `true`, `0`, `"text"`)'
            ) from None
        out.append((left, op, expected))
    return out


def eval_when(expr: str, scope: Scope) -> bool:
    """`<path> == <json>`, `<path> != <json>`, a bare truthy `<path>`, or several
    of those joined with `&&` (all must hold).

    A condition on a skipped step is false — whatever the operator — so a chain
    of dependent conditional steps collapses cleanly when its first link does.
    A key that is absent reads as null.
    """
    for path, op, expected in parse_when(expr):
        actual = resolve_path(path, scope, skipped_ok=True, missing_ok=True)
        if is_skipped(actual):
            return False
        if op is None:
            if not actual:
                return False
        elif op == "==" and actual != expected:
            return False
        elif op == "!=" and actual == expected:
            return False
    return True


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
    # param -> the question to ask the user for it. Asked by code once the
    # reads before the first write have run; filled from the answer; then the
    # plan is validated and gated with the value in place.
    args_ask: Dict[str, str] = field(default_factory=dict)


@dataclass
class Loop:
    id: str
    over: str
    var: str
    steps: List[Step] = field(default_factory=list)
    when: Optional[str] = None


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
        args_ask = d.get("args_ask") or {}
        if not isinstance(args_ask, dict) or not all(isinstance(v, str) for v in args_ask.values()):
            raise PlanError(f"step {sid}: `args_ask` must map parameter names to question strings")
        # A literal that reads `steps[<id>].result…` / `.args…` is a reference the
        # planner filed under the wrong key — no user value ever looks like that.
        # Promote it instead of failing the plan on a bookkeeping slip.
        args = dict(args)
        args_from = dict(args_from)
        for k, v in list(args.items()):
            if isinstance(v, str) and _STEP_REF_RE.match(v.strip()):
                logger.info("Plan step %s: `%s` given as a literal reference; treating it as args_from.", sid, k)
                args_from.setdefault(k, v.strip())
                del args[k]
        when = d.get("when")
        if when is not None and not isinstance(when, str):
            raise PlanError(f"step {sid}: `when` must be a string")
        return Step(id=sid, tool=d["tool"], args=args, args_from=args_from, when=when, args_ask=dict(args_ask))

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
            lwhen = d.get("when")
            if lwhen is not None and not isinstance(lwhen, str):
                raise PlanError(f"loop {lid}: `when` must be a string")
            body = [_step(s) for s in subs]
            _normalise_loop_var(lid, d["as"], body)
            items.append(Loop(id=lid, over=d["for_each"], var=d["as"], steps=body, when=lwhen))
        else:
            items.append(_step(d))
    return items


def _normalise_loop_var(loop_id: str, var: str, body: List[Step]) -> None:
    """Inside a `for_each` only ONE variable exists. A bare name that is not it
    (the prompt's example says `item`, the plan said `as: dashboard`) can only
    mean it — rewrite instead of failing the plan on a naming slip."""

    def _fix(ref: str) -> str:
        e = ref.strip()
        other = _referenced_var(e)
        if other is None or other == var:
            return ref
        logger.info("Plan loop %s: `%s` uses %r for its variable %r; treating it as %r.", loop_id, e, other, var, var)
        return var + e[len(other) :]

    for st in body:
        # A literal argument that names the loop item ("item.dashboard_id" under
        # `args`) is a reference filed under the wrong key AND the wrong name.
        # Only the loop variable and the prompt's example name `item` qualify —
        # a real value can contain a dot.
        for k, v in list(st.args.items()):
            if isinstance(v, str):
                head = _referenced_var(v.strip())
                if head in (var, "item") and _VAR_REF_RE.match(v.strip()) and "." in v:
                    logger.info(
                        "Plan loop %s: `%s` given as a literal for `%s`; treating it as args_from.", loop_id, v, k
                    )
                    st.args_from.setdefault(k, _fix(v))
                    del st.args[k]
        st.args_from = {k: _fix(v) for k, v in st.args_from.items()}
        if st.when:
            fixed = []
            for clause in _split_when(st.when):
                m = _WHEN_RE.match(clause)
                fixed.append(f"{_fix(m.group(1))} {m.group(2)} {m.group(3)}" if m else _fix(clause))
            st.when = " && ".join(fixed)


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


_SWAP_PROOFS = (".validate_dashboard_queries", ".compare_dashboard_values")


def guardrail_validate_before_swap(items: List[PlanItem]) -> Optional[str]:
    """`replace_datasource` on a dashboard this run did NOT create must be
    conditional on a result that proves the perspective answers for it: a
    `validate_dashboard_queries` run on a stage copy, or a
    `compare_dashboard_values` run of the dashboard against both datasources
    (no copy needed). A stage copy (made by `duplicate_dashboard` in this plan)
    is exempt — it exists to be tested."""
    tools = _tool_by_id(items)
    for step, loop in _iter_steps(items):
        if not step.tool.endswith(".replace_datasource"):
            continue
        if _derived_from_tool(step, "dashboard", items, loop, ".duplicate_dashboard"):
            continue  # swapping the copy — that IS the test
        refs = _referenced_step_ids(step.when or "")
        if not any(tools.get(r, "").endswith(_SWAP_PROOFS) for r in refs):
            return (
                f"step {step.id} swaps a dashboard this run did not create without a `when` on a "
                "validate_dashboard_queries or compare_dashboard_values result — the skill requires that proof first"
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
            for sub in _subexprs(ref):
                if not (_STEP_REF_RE.match(sub) or (_referenced_var(sub) and loop is not None)):
                    raise PlanError(f"step {step.id}: malformed reference {sub!r} in {ref!r}")
        if step.when:
            try:
                clauses = parse_when(step.when)
            except PlanError as pe:
                raise PlanError(f"step {step.id}: {pe}") from None
            for path, _op, _exp in clauses:
                var = _referenced_var(path)
                if var is not None and (loop is None or var != loop.var):
                    raise PlanError(f"step {step.id}: `when` uses a variable not in scope")
                if not _referenced_step_ids(path) and var is None:
                    raise PlanError(f"step {step.id}: `when` clause {path!r} is not a reference")
                for rid in _referenced_step_ids(path):
                    if rid not in visible:
                        raise PlanError(f"step {step.id}: `when` references step {rid!r}, which does not run before it")

    def _check_step(step: Step, visible: Set[str], loop: Optional[Loop]) -> None:
        if step.id in seen:
            raise PlanError(f"duplicate step id {step.id!r}")
        seen.add(step.id)
        if step.tool not in allowed:
            # Right method, wrong package (`datamodel.get_dashboards_by_datasource`
            # for `dashboard.…`): method names are unique within a skill, so an
            # exact method match is unambiguous — correct it, say so in the log.
            same_method = [t for t in allowed if t.rsplit(".", 1)[-1] == step.tool.rsplit(".", 1)[-1]]
            if len(same_method) == 1:
                logger.info(
                    "Plan step %s: `%s` is not a skill tool; treating it as `%s`.", step.id, step.tool, same_method[0]
                )
                step.tool = same_method[0]
        if step.tool not in allowed:
            raise PlanError(
                f"step {step.id} uses `{step.tool}`, which the skill '{skill.name}' does not allow "
                f"(allowed: {', '.join(sorted(allowed))})"
            )
        meta = A.TOOL_REGISTRY.get(step.tool)
        if not meta:
            raise PlanError(f"step {step.id}: `{step.tool}` is not an exposed tool")
        schema = meta.get("parameters") or {}
        props = schema.get("properties") or {}
        for param in list(step.args) + list(step.args_from) + list(step.args_ask):
            if props and param not in props:
                raise PlanError(f"step {step.id}: `{step.tool}` has no parameter `{param}`")
        sources = [set(step.args), set(step.args_from), set(step.args_ask)]
        overlap = (sources[0] & sources[1]) | (sources[0] & sources[2]) | (sources[1] & sources[2])
        if overlap:
            raise PlanError(f"step {step.id}: {sorted(overlap)} given by more than one of args/args_from/args_ask")
        # Literal args must be valid on their own; required params supplied by
        # reference or asked of the user are checked when the value exists.
        deferred = set(step.args_from) | set(step.args_ask)
        partial = {
            **{k: v for k, v in schema.items() if k not in ("required",)},
            "required": [r for r in (schema.get("required") or []) if r not in deferred],
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
            if it.when:
                try:
                    lclauses = parse_when(it.when)
                except PlanError as pe:
                    raise PlanError(f"loop {it.id}: {pe}") from None
                for wpath, _op, _exp in lclauses:
                    if _referenced_var(wpath) is not None:
                        raise PlanError(f"loop {it.id}: `when` cannot use a loop variable")
                    for rid in _referenced_step_ids(wpath):
                        if rid not in top_ids:
                            raise PlanError(
                                f"loop {it.id}: `when` references step {rid!r}, which does not run before it"
                            )
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
    if s.args_ask:
        d["args_ask"] = dict(s.args_ask)
    if s.when:
        d["when"] = s.when
    return d


def pending_asks(items: List[PlanItem]) -> List[Tuple[Step, str]]:
    """(step, param) for every value still to be asked of the user, in plan order."""
    return [(st, param) for st, _ in _iter_steps(items) for param in st.args_ask]


def fill_ask(items: List[PlanItem], step_id: str, param: str, value: Any) -> None:
    """Move an asked parameter into the step's literal args with the user's value."""
    for st, _ in _iter_steps(items):
        if st.id == step_id and param in st.args_ask:
            st.args_ask.pop(param)
            st.args[param] = value
            return
    raise PlanError(f"step {step_id!r} has no pending question for `{param}`")


def plan_arguments(items: List[PlanItem], skill: Skill) -> Dict[str, Any]:
    """The approval payload — and therefore the approval KEY. Change a step, an
    argument, a reference, or the order, and the plan must be approved again."""
    steps = []
    for it in items:
        if isinstance(it, Loop):
            entry: Dict[str, Any] = {
                "id": it.id,
                "for_each": it.over,
                "as": it.var,
                "steps": [_step_payload(s) for s in it.steps],
            }
            if it.when:
                entry["when"] = it.when
            steps.append(entry)
        else:
            steps.append(_step_payload(it))
    return {"skill": skill.name, "skill_version": skill.version, "steps": steps}


def _label(tool_id: str) -> str:
    desc = ((A.TOOL_REGISTRY.get(tool_id) or {}).get("description") or "").strip().splitlines()
    text = (desc[0].strip().rstrip(" .") if desc else "") or tool_id.split(".", 1)[-1].replace("_", " ")
    return text[:1].upper() + text[1:]


def _humanise(args: Dict[str, Any], args_from: Dict[str, str], args_ask: Optional[Dict[str, str]] = None) -> str:
    args_ask = args_ask or {}
    parts = []
    for k, v in args.items():
        shown = (
            ", ".join(map(str, v)) if isinstance(v, list) else ("yes" if v is True else "no" if v is False else str(v))
        )
        parts.append(f"{k.replace('_', ' ')}: {shown}")
    for k in args_ask:
        parts.append(f"{k.replace('_', ' ')}: (you will be asked)")
    for k, ref in args_from.items():
        m = _STEP_REF_RE.match(ref.strip())
        if m and m.group(2) == "args":
            src = f"step {m.group(1)}'s {m.group(3).lstrip('.').replace('_', ' ') or 'arguments'}"
        elif m:
            src = f"step {m.group(1)}'s result"
        else:
            src = f"each {ref.split('.', 1)[0]}"
        parts.append(f"{k.replace('_', ' ')}: from {src}")
    return " · ".join(parts)


def render_plan_lines(items: List[PlanItem]) -> List[str]:
    """Prose lines for the UI's plan block and the transcript."""
    lines: List[str] = []
    for it in items:
        if isinstance(it, Loop):
            m = _STEP_REF_RE.match(it.over.strip())
            lines.append(
                f"For each item from step {m.group(1) if m else '?'}:" + (f" (only if {it.when})" if it.when else "")
            )
            for s in it.steps:
                lines.append(f"  {s.id}. {_label(s.tool)} ({s.tool})" + (f" — only if {s.when}" if s.when else ""))
        else:
            lines.append(f"{_label(it.tool)} ({it.tool})" + (f" — only if {it.when}" if it.when else ""))
    return lines


def render_dialog(items: List[PlanItem], skill: Skill, root: Optional[Scope] = None) -> str:
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
            head = f"For each item found in step {m.group(1) if m else '?'}:" + (
                f" *(only if {it.when})*" if it.when else ""
            )
            block = []
            for s in it.steps:
                if _mutates(s.tool):
                    line = f"  - **{_label(s.tool)}** — {_humanise(s.args, s.args_from, s.args_ask)}"
                    if s.when:
                        line += f" *(only if {s.when})*"
                    block.append(line)
                else:
                    reads += 1
            if block:
                writes.append(head)
                writes.extend(block)
        elif _mutates(it.tool):
            line = f"{len(writes) + 1}. **{_label(it.tool)}** — {_humanise(it.args, it.args_from, it.args_ask)}"
            if it.when:
                line += f" *(only if {it.when})*"
            writes.append(line)
        else:
            reads += 1

    lines = [
        f"Using the skill **{skill.name}**. Approving covers every operation below, in this order.",
        "",
    ]
    lines.extend(writes)
    if reads:
        lines.append("")
        lines.append(f"*Plus {reads} read-only step{'s' if reads != 1 else ''} the skill uses to decide and verify.*")
    if skill.compensations:
        lines.append("")
        lines.append(
            "If a step fails, the run stops and undoes what it created in this run "
            f"({', '.join(sorted(_label(t) for t in skill.compensations))}); the report says what ran, "
            "what failed, and what was reverted."
        )
    if root is not None:
        # The per-item lists the loops will run over, from the reads already
        # done — the full roster the summary abbreviates.
        for it in items:
            if not isinstance(it, Loop):
                continue
            try:
                seq = resolve_path(it.over, root, missing_ok=True)
            except PlanError:
                continue
            if isinstance(seq, list) and seq:
                names = [_item_label(x) for x in seq]  # "Title (owner)" when the row carries an owner
                lines.append("")
                lines.append(
                    f"**{it.var.replace('_', ' ').capitalize()}s in scope ({len(names)}):** " + ", ".join(names)
                )
    lines.append("")
    lines.append("Approve to run the whole sequence, or cancel and ask again with any changes.")
    return "\n".join(lines)


_APPROVAL_PLACEHOLDER_RE = re.compile(
    r"\{([a-z_][a-z0-9_]*)\.(args|result|count)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[\*\]|\[\d+\])*)"
    r"(?:\|(s|count|pairs|unique|head:\d+|items_ran|items_skipped|items_failed|items_on_behalf|items_ran_owners|or:[^}]*))?\}"
)


def render_approval(items: List[PlanItem], skill: Skill, root: Scope) -> str:
    """What the user is asked to approve — the skill's own `## Approval` text
    with its placeholders filled (see `render_text`). Falls back to the
    operation list when the skill has no such section."""
    if not skill.approval.strip():
        return render_dialog(items, skill)
    text = render_text(skill.approval, items, skill, root)
    if "approve" not in text.lower():
        text += "\n\nApprove to proceed, or cancel."
    return text


def render_ask(items: List[PlanItem], skill: Skill, root: Scope, fallback_question: str) -> str:
    """The question for a user-supplied value, asked after the reads have run —
    the skill's `## Ask` text with the read results filled in, else the plan's
    own question. Always ends as a question."""
    text = render_text(skill.ask, items, skill, root) if skill.ask.strip() else fallback_question.strip()
    return text.rstrip("?. ") + "?"


# Innermost conditional block: a body with no nested `{?`/`{!` inside. Applied
# repeatedly, so blocks may nest ("found X — {all resolved | Y unresolved}").
_COND_RE = re.compile(
    r"\{(\?|!)([a-z_][a-z0-9_]*\.(?:args|result|count)(?:[^}|]*)(?:\|[a-z_]+)?)\}((?:(?!\{[?!]).)*?)\{/\}", re.S
)
_SPEC_RE = re.compile(
    r"^([a-z_][a-z0-9_]*)\.(args|result|count)((?:\.[A-Za-z_][A-Za-z0-9_]*|\[\*\]|\[\d+\])*)"
    r"(?:\|(count|items_ran|items_skipped|items_failed|items_on_behalf))?$"
)


_ITEMS_SHOWN = 10  # per-item lists in summaries show this many, then "… and K more"


def render_report(items: List[PlanItem], skill: Skill, root: Scope, report: "_Report") -> str:
    """The end-of-run summary from the skill's `## Report` section, or '' when it has none."""
    return render_text(skill.report, items, skill, root, report) if skill.report.strip() else ""


def _item_label(item: Any) -> str:
    if isinstance(item, dict):
        name = item.get("title") or item.get("name") or item.get("email") or item.get("oid") or item.get("id") or "?"
        owner = item.get("owner_email")
        return f"{name} ({owner})" if owner else str(name)
    return str(item)


def render_text(
    template: str, items: List[PlanItem], skill: Skill, root: Scope, report: Optional["_Report"] = None
) -> str:
    """Fill a skill-authored text from the plan and the results gathered so far.
    Code only: nothing here comes from the model.

    Placeholders name a tool by its method: `{create_perspective.args.name}`
    (a literal argument of the plan step using that tool, or what it ran
    with), `{get_dashboards_by_datasource.result[*].title}` (a path into that
    step's result — lists join with commas), `{get_dashboards_by_datasource.count}`
    (rows in a list result). Filters: `|s` (plural s when not 1), `|count`
    (length of a list), `|or:<text>` (fallback when missing), `|pairs`
    (many-to-many entries as "A ↔ B (keys)"), `|head:N` (first N items, then
    "… and K more"), `|unique` (distinct values, first occurrence order).
    With a run report: `{method.result|items_ran}` / `|items_skipped` /
    `|items_failed` list the loop items (dashboards) for which that step ran,
    was skipped, or failed — "Title (owner)" each — and `|items_ran_owners`
    the distinct owners of the ran items.

    Conditional blocks keep a warning line off the screen when there is
    nothing to warn about: `{?spec}…{/}` renders its body only when the value
    is present and non-zero/non-empty; `{!spec}…{/}` only when it is not.
    """
    by_method: Dict[str, Step] = {}
    for st, _ in _iter_steps(items):
        by_method.setdefault(st.tool.rsplit(".", 1)[-1], st)

    def _lookup(method: str, kind: str, tail: str, *, missing_ok: bool) -> Tuple[bool, Any]:
        """(found, value) for a placeholder; found=False when it cannot be resolved."""
        st = by_method.get(method)
        if st is None:
            logger.warning("Skill %s text names %r, which the plan does not use", skill.name, method)
            return False, None
        try:
            if kind == "args":
                try:
                    base: Any = root.step_args(st.id)
                except KeyError:
                    # Not run yet: its literal args, plus whatever of its
                    # references can already be resolved from the reads.
                    base = dict(st.args)
                    for k, ref in st.args_from.items():
                        try:
                            base[k] = resolve_path(ref, root)
                        except PlanError:
                            pass
                value = _walk(base, tail, method, missing_ok=missing_ok)
            else:
                base = root.step_result(st.id)
                if is_skipped(base):
                    return False, None
                value = _count_of(base) if kind == "count" else _walk(base, tail, method, missing_ok=missing_ok)
                if kind == "count" and value is None:
                    value = len(base) if isinstance(base, list) else None
        except (KeyError, PlanError):
            return False, None
        return value is not None, value

    def _truthy(spec: str) -> bool:
        m = _SPEC_RE.match(spec.strip())
        if not m:
            return False
        if m.group(4) and m.group(4).startswith("items_"):
            st = by_method.get(m.group(1))
            bucket = (report.items.get(st.tool) if (report is not None and st is not None) else None) or {}
            return bool(bucket.get(m.group(4)[len("items_") :]))
        found, value = _lookup(m.group(1), m.group(2), m.group(3) or "", missing_ok=True)
        if not found:
            return False
        if m.group(4) == "count":
            return isinstance(value, (list, dict)) and len(value) > 0
        return bool(value)

    def _cond(m: "re.Match[str]") -> str:
        show = _truthy(m.group(2))
        return m.group(3) if (show if m.group(1) == "?" else not show) else ""

    def _fill(m: "re.Match[str]") -> str:
        method, kind, tail, filt = m.group(1), m.group(2), m.group(3) or "", m.group(4)
        if filt and filt.startswith("items_"):
            st = by_method.get(method)
            bucket = (report.items.get(st.tool) if (report is not None and st is not None) else None) or {}
            if filt == "items_ran_owners":
                owners: List[str] = []
                for it_ in bucket.get("ran", []):
                    o = it_.get("owner_email") if isinstance(it_, dict) else None
                    if o and o not in owners:
                        owners.append(o)
                return ", ".join(owners) if owners else "none"
            chosen = bucket.get(filt[len("items_") :], [])
            if not chosen:
                return "none"
            shown = ", ".join(_item_label(x) for x in chosen[:_ITEMS_SHOWN])
            return shown + (f" … and {len(chosen) - _ITEMS_SHOWN} more" if len(chosen) > _ITEMS_SHOWN else "")
        fallback = filt[3:] if filt and filt.startswith("or:") else None
        found, value = _lookup(method, kind, tail, missing_ok=fallback is not None)
        if not found:
            return "?" if fallback is None else fallback
        if filt == "s":
            return "" if value == 1 else "s"
        if filt == "count":
            return str(len(value)) if isinstance(value, (list, dict)) else "?"
        if filt == "unique" and isinstance(value, list):
            seen: List[str] = []
            for v in value:
                if v is not None and str(v) not in seen:
                    seen.append(str(v))
            return ", ".join(seen)
        if filt and filt.startswith("head:") and isinstance(value, list):
            n = int(filt[5:])
            shown = ", ".join(str(v) for v in value[:n])
            return shown + (f" … and {len(value) - n} more" if len(value) > n else "")
        if filt == "pairs" and isinstance(value, list):
            out = []
            for e in value:
                if isinstance(e, dict) and e.get("table_a") and e.get("table_b"):
                    keys = ", ".join(map(str, e.get("columns_a") or []))
                    out.append(f"{e['table_a']} ↔ {e['table_b']}" + (f" ({keys})" if keys else ""))
            return "; ".join(out) if out else "none"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return "yes" if value is True else "no" if value is False else str(value)

    text = template
    for _ in range(8):  # resolve innermost blocks first; nesting deeper than this is a template bug
        nxt = _COND_RE.sub(_cond, text)
        if nxt == text:
            break
        text = nxt
    return _APPROVAL_PLACEHOLDER_RE.sub(_fill, text).strip()


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


def _outcome_line(tool_id: str, ok: bool, payload: Any, label: Optional[str] = None) -> str:
    """'<label> — <what happened>', from metadata only (safe in both summarization modes)."""
    label = label or _label(tool_id)
    if not ok:
        return f"{label} — failed"
    if isinstance(payload, dict) and "all_match" in payload:
        n = payload.get("compared")
        return f"{label} — {'all ' + str(n) + ' widgets match' if payload['all_match'] else 'differences found'}"
    if isinstance(payload, dict) and "published" in payload and "widgets_updated" in payload:
        # A dashboard change: viewers see it only once the shared copy is
        # published. Say so honestly when the SDK could not confirm that.
        if payload.get("published") is True:
            return f"{label} — done and published"
        owner = payload.get("owner")
        who = f"only the owner can publish it ({owner})" if owner else (payload.get("publish_error") or "")
        return f"{label} — done for the owner; publish pending" + (f" — {who}" if who else "")
    n = _count_of(payload)
    return f"{label} — {n} row{'s' if n != 1 else ''}" if n is not None else f"{label} — done"


def _loop_item(scope: Scope) -> Any:
    """The current loop item (a dashboard row, say), or None outside a loop."""
    return next(iter(scope.vars.values()), None) if scope.vars else None


def _item_title(scope: Scope) -> Optional[str]:
    """A loop item's display name, for per-item report lines ('Sales Overview: …')."""
    item = next(iter(scope.vars.values()), None) if scope.vars else None
    if isinstance(item, dict):
        for k in ("title", "name", "email", "dashboard_id", "oid", "id"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(item) if isinstance(item, str) else None


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
    """`{args.x}` / `{result.y}` / `{result.y.z}` placeholders in a compensation's args.

    Dotted paths walk nested dicts — `replace_datasource` reports the old
    datasource as an object, and the revert call takes its `title`. A missing
    key at any depth yields None, which schema validation then rejects loudly.
    """
    if not isinstance(value, str):
        return value
    m = re.fullmatch(r"\{(args|result)((?:\.[A-Za-z_][A-Za-z0-9_]*)+)\}", value.strip())
    if not m:
        return value
    cur: Any = step_args if m.group(1) == "args" else (payload if isinstance(payload, dict) else {})
    for key in m.group(2).lstrip(".").split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


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
    skipped_because: List[str] = field(default_factory=list)  # parallel to `skipped`: the condition that was false
    outcomes: List[str] = field(default_factory=list)  # one line per executed step, in the skill's words
    # tool_id -> {"ran" | "skipped" | "failed": [loop item dicts]} for steps that ran
    # inside a for_each — what the end-of-run summary lists per dashboard.
    items: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)

    def note_item(self, tool: str, status: str, item: Any) -> None:
        if item is None:
            return
        self.items.setdefault(tool, {"ran": [], "skipped": [], "failed": [], "on_behalf": []})[status].append(item)

    failed: Optional[Tuple[str, str]] = None
    not_attempted: List[str] = field(default_factory=list)
    compensated: List[str] = field(default_factory=list)
    compensation_failed: Optional[Tuple[str, str]] = None
    handoff: Optional[str] = None
    outcome: str = "ok"

    def text(self, body: str, label_of: Optional[Any] = None) -> str:
        """The report, in the user's words: the skill's step labels (or the
        tool's description) instead of tool ids; a skipped step says only that
        its condition was not met — the condition itself is in the logs."""
        lab = label_of or (lambda t: t)

        def _names(tools: List[str]) -> str:
            seen: List[str] = []
            for t in tools:
                n = lab(t)
                if n not in seen:
                    seen.append(n)
            return ", ".join(seen)

        lines: List[str] = []
        if self.handoff:
            lines.append(self.handoff)
        if self.failed:
            tool, reason = self.failed
            lines.append(f"**Stopped** — {lab(tool) if tool != 'plan' else 'the plan'} failed: {reason}")
        if self.outcomes:
            lines.append("**Done**\n" + "\n".join(f"- {o}" for o in self.outcomes))
        elif self.ran:
            lines.append("**Done:** " + _names(self.ran))
        if self.skipped:
            lines.append(
                "**Skipped, condition not met:** " + _names([t for t in self.skipped if not t.startswith("loop ")])
            )
        if self.not_attempted:
            lines.append("**Not attempted:** " + _names(self.not_attempted))
        if self.compensated:
            lines.append("**Reverted:** " + _names(self.compensated))
        if self.compensation_failed:
            tool, reason = self.compensation_failed
            lines.append(
                f"**Could not revert** {lab(tool)}: {reason} — stopped unwinding; "
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
    root: Optional[Scope] = None,
    until_write: bool = False,
) -> Tuple[_Report, int, Scope, int]:
    """Run `items` in order. Returns (report, steps_executed, root scope, items consumed).

    `until_write` is the look-first pass before the approval gate: reads run,
    a write whose `when` is already false is skipped, and the pass stops at
    the first write that WOULD run — that is where the dialog goes. The scope
    it returns is what the resume turn continues from.
    """
    root = root if root is not None else Scope()
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

        # The condition first — a step gated on a skipped step is itself
        # skipped, and its arguments (which may point at that step) are never
        # resolved. Then the references, with real values.
        if step.when and not eval_when(step.when, scope):
            _skip(step.id, step.tool, step.when, scope)
            if loop_pos:
                report.note_item(step.tool, "skipped", _loop_item(scope))
            return True
        if step.args_ask:
            raise PlanError(f"step {step.id} still needs the user's answer for {sorted(step.args_ask)}")
        args = dict(step.args)
        for param, ref in step.args_from.items():
            args[param] = resolve_path(ref, scope)

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
        scope.args[step.id] = dict(args)

        n = steps_executed + 1
        if _mutates(step.tool):
            A.audit_logger.info(
                "EXECUTING mutation (skill %s) tool=%s args=%s",
                skill.name,
                step.tool,
                json.dumps(A._scrub_secrets(args), ensure_ascii=False),
            )
        ev: Dict[str, Any] = {
            "phase": "executing",
            "step": n,
            "max_steps": SKILL_MAX_STEPS,
            "tool_id": step.tool,
            "label": skill.step_labels.get(step.tool) or _label(step.tool),
            "skill": skill.name,
        }
        if loop_pos:
            ev["loop_index"], ev["loop_total"] = loop_pos
            ev["loop_var"] = scope_var_name(scope)
        await A._emit_agent_progress(ev)

        result = await A._invoke_tool_traced(mcp_client, step.tool, args, mode)
        ok = A._effective_ok(result)
        payload = _payload_of(result)
        A._record_tool_result(result)
        A._record_step(n, step.tool, result, label=skill.step_labels.get(step.tool) or _label(step.tool))
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
                "label": skill.step_labels.get(step.tool) or _label(step.tool),
                "outcome": _outcome_line(step.tool, ok, payload),
                **({"loop_index": loop_pos[0], "loop_total": loop_pos[1]} if loop_pos else {}),
            }
        )
        if not ok:
            reason = str(
                (result or {}).get("error") or A._payload_failure_reason(payload) or "no reason reported"
            ).strip()
            report.failed = (step.tool, reason)
            if loop_pos:
                report.note_item(step.tool, "failed", _loop_item(scope))
            return False

        scope.results[step.id] = payload
        report.ran.append(step.tool)
        _lbl = skill.step_labels.get(step.tool) or _label(step.tool)
        _title = _item_title(scope) if loop_pos else None
        report.outcomes.append((f"{_title}: " if _title else "") + _outcome_line(step.tool, ok, payload, _lbl))
        if loop_pos:
            report.note_item(step.tool, "ran", _loop_item(scope))
            # "Moved on the owner's behalf" matters to the reader only under
            # co-authoring, where the owner's private copy lags the shared one.
            # With the feature off there is a single copy and the outcome is the
            # same as an owner run, whatever the SDK had to do to get there.
            if (
                isinstance(payload, dict)
                and payload.get("ownership_transferred_temporarily") is True
                and payload.get("co_authoring") is True
            ):
                report.note_item(step.tool, "on_behalf", _loop_item(scope))
        if _mutates(step.tool):
            done.append(_Done(step.id, step.tool, args, payload))
            if step.tool.endswith(".duplicate_dashboard"):
                created.update(_ids_in(payload))
            elif step.tool.endswith(".delete_dashboard"):
                gone = {str(v) for k, v in args.items() if k in ("dashboard_id", "dashboard", "oid") and v}
                removed.update(gone & created)
        return True

    def _skip(item_id: str, tool_or_label: str, when: str, scope: Scope) -> None:
        report.skipped.append(tool_or_label)
        report.skipped_because.append(when)
        scope.results[item_id] = SKIPPED
        logger.info("Skill item %s (%s) skipped: `%s` is false", item_id, tool_or_label, when)

    def _is_write(it: PlanItem) -> bool:
        return any(_mutates(st.tool) for st in it.steps) if isinstance(it, Loop) else _mutates(it.tool)

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
            await A._emit_agent_progress(
                {"phase": "compensating", "tool_id": spec["tool"], "label": _label(spec["tool"]), "skill": skill.name}
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

    consumed = 0
    try:
        for it in items:
            if until_write and _is_write(it):
                # Look-first pass: a write whose condition is already false is
                # skipped here, so the dialog never asks approval for it.
                if it.when and not eval_when(it.when, root):
                    label = it.steps[0].tool if isinstance(it, Loop) else it.tool
                    _skip(it.id, f"loop {it.id}" if isinstance(it, Loop) else label, it.when, root)
                    consumed += 1
                    continue
                break  # this is where approval is asked
            if isinstance(it, Loop):
                if it.when and not eval_when(it.when, root):
                    _skip(it.id, f"loop {it.id}", it.when, root)
                    consumed += 1
                    continue
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
            consumed += 1
    except _Stop as stop:
        idx = all_ids.index(stop.step_id) if stop.step_id in all_ids else len(all_ids)
        report.not_attempted = [s.tool for s, _ in list(_iter_steps(items))[idx + 1 :]]
        report.outcome = "skill_failed"
        await _compensate()
    except PlanError as pe:
        report.failed = report.failed or ("plan", str(pe))
        report.outcome = "skill_failed"
        await _compensate()
    return report, steps_executed, root, consumed


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
                f"The skill `{meta.get('name')}` this plan came from is no longer available, so I did not run it.",
            )
        if skill.version != meta.get("version"):
            return _finish(
                "skill_changed",
                f"The skill `{skill.name}` changed after this plan was proposed, so I did not run the old plan. "
                "Ask again and I will plan from the current version.",
            )
        if not A._consume_approval(approved_mutations, PLAN_TOOL_ID, pending_plan.get("plan_arguments") or {}):
            logger.info("Dropping paused skill plan (no matching approval this turn).")
            return _finish(
                "plan_dropped", "That plan was not approved, so nothing ran. Ask again if you want to proceed."
            )
        try:
            items = parse_plan(pending_plan.get("plan") or {})
            # Re-validate on resume, not just at plan time. The frontmatter
            # `tools` list is what makes "never change ownership" impossible
            # rather than discouraged, and validate_plan is what enforces it —
            # checking only once meant the control held at exactly one moment.
            # Cheap, and the alternative is a guarantee with a gap in it.
            validate_plan(items, skill)
        except PlanError as pe:
            return _finish("plan_invalid", f"I could not run the approved plan: {pe}")
        # Continue from where the look-first pass stopped: its read results and
        # arguments are the scope the remaining steps reference.
        root = Scope.restore(pending_plan.get("scope"))
        consumed = int(pending_plan.get("consumed") or 0)
        report.ran = list(pending_plan.get("ran") or [])
        report.skipped = list(pending_plan.get("skipped") or [])
        report.skipped_because = list(pending_plan.get("skipped_because") or [])
        # Outcomes of the pre-gate reads; a paused plan without them (older
        # pause) still lists what ran, from `ran`.
        report.outcomes = list(pending_plan.get("outcomes") or []) or [
            f"{_labeller(skill)(t)} — done" for t in report.ran
        ]
        already_approved = True  # consumed above — the continuation must not gate again
        logger.info(
            "Resuming approved skill plan '%s' (%d item(s) left after %d already run).",
            skill.name,
            len(items) - consumed,
            consumed,
        )
    else:
        # ------------------------------------------------------------ fresh
        assert skill is not None and plan is not None, "run() needs skill+plan or pending_plan"
        if "error" in plan and "steps" not in plan:
            logger.warning("Skill '%s' pass produced no plan: %r", skill.name, str(plan.get("error"))[:120])
            return _finish(
                "plan_unparsable",
                "I recognised this as something I have a skill for, but I could not turn it into a plan this time. "
                "Nothing was changed. Please rephrase the request, or name the data model explicitly.",
            )
        if isinstance(plan.get("ask"), str) and plan["ask"].strip():
            # The procedure needs a value only the user can supply. Ask through
            # the ordinary clarification channel; the next turn re-plans with
            # the answer in history (see call_llm_with_tools). Capped like any
            # other clarification so a non-answer cannot loop.
            question = plan["ask"].strip()
            attempts = int(plan.get("attempts") or 1)
            if attempts > A.CLARIFY_MAX_ATTEMPTS:
                logger.info("Skill '%s' clarification cap (%d) reached; giving up.", skill.name, A.CLARIFY_MAX_ATTEMPTS)
                return _finish(
                    "clarification_cap",
                    f"I can't do this without that information. {question} "
                    "Ask again with it included and I will take it from there.",
                )
            # One thing is missing, so ask one direct question — a lead line and
            # a single bullet (the multi-field shape) read as a list of one.
            question = question.rstrip("?. ") + "?"
            A._record_pending_clarification(
                {
                    "tool_id": PLAN_TOOL_ID,
                    "skill": {"name": skill.name, "version": skill.version},
                    "missing_fields": [],
                    "filled_args": {},
                    "attempts": attempts,
                    "question": question,
                }
            )
            logger.info("Skill '%s' awaiting the user's answer (attempt %d).", skill.name, attempts)
            return _finish("awaiting_clarification", question)
        try:
            items = parse_plan(plan)
            if not items:
                return _finish("no_plan", "This skill did not apply to the request after all — nothing planned.")
            validate_plan(items, skill)
        except PlanError as pe:
            logger.warning("Skill '%s' plan rejected: %s", skill.name, pe)
            return _finish("plan_invalid", f"I could not build a safe plan for this: {pe}")
        root = Scope()
        consumed = 0
        already_approved = False

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
        # No plan text for the UI: a skill run shows its stages as they happen
        # (labels, checklist, status line); the internal plan is for the
        # transcript, the logs and the approval's details expander.
        await A._emit_agent_progress({"phase": "planned", "step": 1, "max_steps": SKILL_MAX_STEPS, "skill": skill.name})

        # ------------------------------------------------------------ look first
        if (has_mutation(items) or pending_asks(items)) and A.REQUIRE_MUTATION_CONFIRM:
            # Run the reads that precede the first write, so a question or the
            # dialog can say what WAS FOUND (which dashboards, how many tables)
            # and a blocking finding is reported instead of asked about.
            report, steps_executed, root, consumed = await _execute(
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
                root=root,
                until_write=True,
            )
            if report.outcome != "ok" or consumed >= len(items):
                # A read failed, or every write's condition was already false:
                # there is nothing to ask or approve. Report what was found.
                return _finish(report.outcome, report.text("", _labeller(skill)))

    return await _continue(
        items,
        skill,
        root=root,
        consumed=consumed,
        report=report,
        attempts=1,
        finish=_finish,
        mcp_client=mcp_client,
        mode=mode,
        summ_on=summ_on,
        transcript=transcript,
        raw_results=raw_results,
        steps_executed=steps_executed,
        trace=trace,
        approved_mutations=approved_mutations,
        approved=already_approved,
    )


def _paused_state(
    skill: Skill,
    items: List[PlanItem],
    root: Scope,
    consumed: int,
    report: _Report,
    transcript: List[Dict[str, Any]],
    raw_results: List[Tuple[str, Any]],
    steps_executed: int,
) -> Dict[str, Any]:
    """Everything a later turn needs to carry on from here without re-running anything."""
    plan_args = plan_arguments(items, skill)
    return {
        "transcript": transcript,
        "raw_results": raw_results,
        "steps_executed": steps_executed,
        "tool_id": PLAN_TOOL_ID,
        "arguments": plan_args,
        "plan": {"steps": plan_args["steps"]},
        "plan_arguments": plan_args,
        "skill": {"name": skill.name, "version": skill.version},
        "scope": root.snapshot(),
        "consumed": consumed,
        "ran": list(report.ran),
        "skipped": list(report.skipped),
        "skipped_because": list(report.skipped_because),
        "outcomes": list(report.outcomes),
    }


async def _continue(
    items: List[PlanItem],
    skill: Skill,
    *,
    root: Scope,
    consumed: int,
    report: _Report,
    attempts: int,
    finish: Any,
    mcp_client: McpClient,
    mode: str,
    summ_on: bool,
    transcript: List[Dict[str, Any]],
    raw_results: List[Tuple[str, Any]],
    steps_executed: int,
    trace: Dict[str, Any],
    approved_mutations: Set[Tuple[str, str]],
    approved: bool = False,
) -> str:
    """After the look-first reads: ask what must be asked, then gate, then run.

    Shared by the fresh path and the answer path so a question and the dialog
    always come from the same code, in the same order: every `args_ask` is
    asked (one per turn, with the read results in hand), then the approval, then
    execution of what is left.
    """
    # Never ask for a value a step will not use: a `when` that can already be
    # judged from the reads (no dashboards found, an analysis error) and is
    # false marks the step skipped, and the question with it.
    asks = []
    for st, _ in pending_asks(items):
        if st.when:
            try:
                if not eval_when(st.when, root):
                    report.skipped.append(st.tool)
                    report.skipped_because.append(st.when)
                    root.results[st.id] = SKIPPED
                    logger.info("Skill item %s (%s) skipped before asking: `%s` is false", st.id, st.tool, st.when)
                    continue
            except PlanError:
                pass  # depends on something not run yet — ask, judge later
        asks.append((st, next(iter(st.args_ask))))
    if asks:
        step, param = asks[0]
        question = render_ask(items, skill, root, step.args_ask[param])
        A._record_pending_clarification(
            {
                **_paused_state(skill, items, root, consumed, report, transcript, raw_results, steps_executed),
                "missing_fields": [param],
                "filled_args": {},
                "attempts": attempts,
                "question": question,
                "ask": {"step_id": step.id, "param": param},
            }
        )
        logger.info("Skill '%s' asking for `%s` of step %s (attempt %d).", skill.name, param, step.id, attempts)
        await A._emit_agent_progress({"phase": "awaiting_answer", "skill": skill.name})
        return finish("awaiting_clarification", question)

    remaining = items[consumed:]
    if has_mutation(remaining) and A.REQUIRE_MUTATION_CONFIRM and not approved:
        plan_args = plan_arguments(items, skill)
        if not A._consume_approval(approved_mutations, PLAN_TOOL_ID, plan_args):
            summary = render_approval(items, skill, root)
            A.record_issued(approved_mutations, PLAN_TOOL_ID, plan_args)
            A._record_tool_result(
                {
                    "ok": False,
                    "pending_confirmation": {
                        "tool_id": PLAN_TOOL_ID,
                        "arguments": plan_args,
                        "reason": summary,
                        # The exact operation list + the full per-item roster, for the details expander.
                        "details": render_dialog(items, skill, root),
                    },
                }
            )
            A._record_pending_loop(
                _paused_state(skill, items, root, consumed, report, transcript, raw_results, steps_executed)
            )
            logger.info(
                "Skill plan '%s' awaiting approval (%d read step(s) done, %d item(s) to go).",
                skill.name,
                steps_executed,
                len(remaining),
            )
            await A._emit_agent_progress({"phase": "awaiting_approval", "skill": skill.name})
            return finish("pending_mutation", summary)

    # --------------------------------------------------------------- execute
    report, steps_executed, root, _ = await _execute(
        remaining,
        skill,
        mcp_client=mcp_client,
        mode=mode,
        summ_on=summ_on,
        transcript=transcript,
        raw_results=raw_results,
        steps_executed=steps_executed,
        trace=trace,
        report=report,
        root=root,
    )
    # Code-built in BOTH summarization modes, like migration: a screen that
    # says what was WRITTEN must be deterministic.
    await A._emit_agent_progress({"phase": "done", "outcome": report.outcome, "skill": skill.name})
    summary = render_report(items, skill, root, report) if report.outcome == "ok" else ""
    steps_text = report.text("", _labeller(skill))
    # A clean run ends with the summary alone (the per-step results sit in the
    # expanders); the step list is for runs where something was skipped or failed.
    clean = report.outcome == "ok" and not report.skipped and not report.failed and not report.compensated
    if summary and clean:
        return finish(report.outcome, summary)
    return finish(report.outcome, f"{summary}\n\n{steps_text}" if summary else steps_text)


async def answer(
    pending: Dict[str, Any],
    *,
    latest_user_message: Dict[str, Any],
    user_text: str,
    mode: str,
    mcp_client: McpClient,
    approved_mutations: Set[Tuple[str, str]],
    summ_on: bool,
    turn_trace_id: str,
    trace: Dict[str, Any],
    **_ignored: Any,
) -> Optional[str]:
    """The turn after a skill asked for a value. Fills the plan from the reply
    and continues (next question, or the approval) WITHOUT re-planning or
    re-running the reads. Returns None when the reply is a different request —
    the caller then drops the question and plans the message fresh.
    """
    meta = pending.get("skill") or {}
    skills = load_skills(registry_ids=A.all_registry_tool_ids(), allowed=A.exposed_tool_ids())
    skill = skills.get(str(meta.get("name") or ""))

    def _finish(outcome: str, reply: str) -> str:
        trace["outcome"] = outcome
        trace["skill"] = meta.get("name")
        _out = A.turn_output()
        if _out is not None and skill is not None:
            _out["skill"] = {"name": skill.name, "version": skill.version}
        A._write_llm_trace(trace)
        return reply

    if skill is None or skill.version != meta.get("version"):
        return _finish(
            "skill_changed",
            "The skill behind that question changed before you answered, so I did not continue. "
            "Ask again and I will start from the current version.",
        )
    ask = pending.get("ask") or {}
    try:
        items = parse_plan(pending.get("plan") or {})
    except PlanError as pe:
        return _finish("plan_invalid", f"I could not continue the paused plan: {pe}")
    step = next((st for st, _ in _iter_steps(items) if st.id == ask.get("step_id")), None)
    param = str(ask.get("param") or "")
    if step is None or param not in step.args_ask:
        return _finish("plan_invalid", "I could not find the question that was pending, so I did not continue.")

    root = Scope.restore(pending.get("scope"))
    consumed = int(pending.get("consumed") or 0)
    report = _Report(
        ran=list(pending.get("ran") or []),
        skipped=list(pending.get("skipped") or []),
        skipped_because=list(pending.get("skipped_because") or []),
        outcomes=list(pending.get("outcomes") or [])
        or [f"{_labeller(skill)(t)} — done" for t in (pending.get("ran") or [])],
    )
    attempts = int(pending.get("attempts") or 1)
    question = str(pending.get("question") or step.args_ask[param])
    prop = ((A.TOOL_REGISTRY.get(step.tool) or {}).get("parameters") or {}).get("properties", {}).get(param) or {}

    value = await _interpret_answer(question, param, prop, user_text, turn_trace_id, summ_on=summ_on)
    if value is None:
        # Not an answer. A real change of subject stands on its own words;
        # a non-answer ("hmm", "why?") does not — re-ask, up to the cap.
        _, fresh_pkg, _, _ = await A._navigate_to_tools(latest_user_message, [], turn_trace_id, mode)
        if fresh_pkg != "__unclear__":
            logger.info("Skill '%s' question dropped: the reply is a new request.", skill.name)
            return None
        if attempts + 1 > A.CLARIFY_MAX_ATTEMPTS:
            return _finish(
                "clarification_cap",
                f"I can't continue without that. {question} Ask again with it included and I will take it from there.",
            )
        return await _continue(
            items,
            skill,
            root=root,
            consumed=consumed,
            report=report,
            attempts=attempts + 1,
            finish=_finish,
            mcp_client=mcp_client,
            mode=mode,
            summ_on=summ_on,
            transcript=list(pending.get("transcript") or []),
            raw_results=list(pending.get("raw_results") or []),
            steps_executed=int(pending.get("steps_executed") or 0),
            trace=trace,
            approved_mutations=approved_mutations,
        )

    fill_ask(items, step.id, param, value)
    try:
        validate_plan(items, skill)
    except PlanError as pe:
        logger.info("Skill '%s': answer %r rejected by validation: %s", skill.name, value, pe)
        if attempts + 1 > A.CLARIFY_MAX_ATTEMPTS:
            return _finish(
                "clarification_cap", f"That value does not work here ({pe}). Ask again with a different one."
            )
        step.args.pop(param, None)
        step.args_ask[param] = question
        return await _continue(
            items,
            skill,
            root=root,
            consumed=consumed,
            report=report,
            attempts=attempts + 1,
            finish=_finish,
            mcp_client=mcp_client,
            mode=mode,
            summ_on=summ_on,
            transcript=list(pending.get("transcript") or []),
            raw_results=list(pending.get("raw_results") or []),
            steps_executed=int(pending.get("steps_executed") or 0),
            trace=trace,
            approved_mutations=approved_mutations,
        )
    logger.info("Skill '%s': `%s` of step %s = %r from the user's answer.", skill.name, param, step.id, value)
    return await _continue(
        items,
        skill,
        root=root,
        consumed=consumed,
        report=report,
        attempts=1,
        finish=_finish,
        mcp_client=mcp_client,
        mode=mode,
        summ_on=summ_on,
        transcript=list(pending.get("transcript") or []),
        raw_results=list(pending.get("raw_results") or []),
        steps_executed=int(pending.get("steps_executed") or 0),
        trace=trace,
        approved_mutations=approved_mutations,
    )


async def _interpret_answer(
    question: str,
    param: str,
    prop: Dict[str, Any],
    reply: str,
    trace_id: str,
    *,
    summ_on: bool = False,
) -> Any:
    """One small model call: the user's reply → the parameter's value, or None.
    The value is what the user WROTE — the prompt forbids inventing or normalising.

    The rendered question is filled from LIVE RESULTS (`{method.result.path}`,
    `{method.count}`), so passing it verbatim sent Sisense data to the model
    even with summarization OFF — the one place a skill run was NOT identical
    in both modes, contrary to what CLAUDE.md and docs/security.md say. With
    the switch off the model gets the parameter's own metadata instead, which
    is all an extraction task needs: it is reading the USER's reply, not the
    question.
    """
    from ._prompts import SKILL_ANSWER_SYSTEM_PROMPT

    reply = (reply or "").strip()
    if not reply:
        return None
    safe_question = question if summ_on else f"What value should be used for `{param}`?"
    system = SKILL_ANSWER_SYSTEM_PROMPT.format(
        question=safe_question,
        param=param,
        description=str(prop.get("description") or "").split(". ")[0],
        type=prop.get("type") or "string",
    )
    try:
        data = await A.call_llm_raw(
            [{"role": "system", "content": system}, {"role": "user", "content": reply}],
            tools=None,
            trace_id=trace_id,
            label="skill_answer",
        )
        text, _ = A._pick_tool_calls_from_llm_response(data)
        obj = A._parse_skill_plan_json(text or "") if "steps" in (text or "") else None
        if obj is None:
            raw = (text or "").strip().strip("`")
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
            obj = json.loads(raw) if raw else {}
    except Exception as exc:  # noqa: BLE001 — a failed interpretation is a non-answer, never a crash
        logger.warning("Skill answer interpretation failed (%s); treating the reply as a non-answer.", exc)
        return None
    value = obj.get("value") if isinstance(obj, dict) else None
    return None if value in (None, "") else value
