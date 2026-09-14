---
name: optimize-datamodel-for-ai-assistant
description: Optimize a data model for Sisense AI Assistant — build a perspective
  containing only what its dashboards actually use, move those dashboards onto
  it, and verify every widget still answers. Use when someone asks to prepare,
  optimize, slim down or make a data model ready for Sisense AI Assistant.
version: 1
requires_role: dataDesigner
tools:
  - datamodel.analyze_perspective_requirements
  - datamodel.create_perspective
  - datamodel.deploy_datamodel
  - dashboard.get_dashboards_by_datasource
  - dashboard.duplicate_dashboard
  - dashboard.replace_datasource
  - dashboard.validate_dashboard_queries
  - dashboard.delete_dashboard
guardrails:
  - id: validate-before-swap
    rule: replace_datasource on a dashboard NOT created by this run requires a
      prior validate_dashboard_queries on its stage copy with zero failed widgets
---

## When this applies
The user wants to optimize or prepare a data model for **Sisense AI Assistant**
— phrased as "make it AI-ready", "optimize it for the AI assistant", "slim it
down for AI", or "build a perspective from what's actually used". The point of
the perspective is that Sisense AI Assistant answers better over a model that
carries only the tables and columns dashboards actually use.
If the user only asks whether a model IS ready, do not run this procedure —
answer with the two reads alone (`analyze_perspective_requirements` and
`get_dashboards_by_datasource`) and report what a perspective would need, which
dashboards are affected and who owns them, and anything that would block it.

## Procedure
1. **Analyze what the dashboards need**
   (`datamodel.analyze_perspective_requirements`). Its `perspectives` output
   is the exact payload for step 2 — use it as-is. It already includes join
   columns and custom-column sources the dashboards never reference directly.
   Do NOT derive the column list from unused-column analysis: that omits
   dependencies and the perspective will build but its dashboards will fail
   at query time. If `errors` contains `unresolved_reference`, stop and
   report — a dashboard references a column the model no longer has, and no
   perspective fixes that.
2. **Create the perspective** (`datamodel.create_perspective`) from step 1's
   payload. Name it `<model>_AI_Assistant` unless the user gave a name. It fails fast on
   a duplicate name — treat that as "already exists", and adopt it.
3. **Build it** (`datamodel.deploy_datamodel`). It is not queryable until built.
4. **Find every dashboard on the root model**
   (`dashboard.get_dashboards_by_datasource`). Note each one's `owner_email`.
5. **For each dashboard**, in order:
   a. **Duplicate it** (`dashboard.duplicate_dashboard`) — the copy is titled
      `<title>_perspective_stage`, which is how a re-run finds its own leftovers.
   b. **Point the copy at the perspective** (`dashboard.replace_datasource`).
   c. **Validate the copy** (`dashboard.validate_dashboard_queries` with the
      perspective as `datasource`). Every widget answers, or the perspective is
      missing something that widget needs.
   d. **Only if c reported zero failed widgets**, point the ORIGINAL at the
      perspective (`dashboard.replace_datasource`). Otherwise leave the original
      exactly as it was and record why.
   e. **Delete the stage copy** (`dashboard.delete_dashboard`, id AND title).

## Never
- Change a dashboard's owner, for any reason. Only the owner can publish; a
  dashboard the user cannot publish is REPORTED with its owner's name.
- Touch a dashboard on a different root model.
- Swap an original whose copy did not validate clean.
- Leave a stage copy behind on success.

## On failure
- Delete every stage copy this run created.
- Swap back every ORIGINAL this run swapped, to its `previous_datasource`.
- Leave the perspective in place if it built — it is harmless and the next
  run adopts it. Delete it only if step 2 or 3 is what failed.
- Report ran / failed (with the SDK's error verbatim) / not attempted.

## Report
- The perspective's name and how many tables/columns it kept vs the model.
- Dashboards swapped, by title.
- Dashboards NOT swapped, by title, with the failing widgets.
- Dashboards that now need publishing, grouped by owner email — the user
  will forward this.
