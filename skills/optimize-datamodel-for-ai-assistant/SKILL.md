---
name: optimize-datamodel-for-ai-assistant
description: Optimize a data model for Sisense AI Assistant — build a perspective
  containing only what its dashboards actually use, move those dashboards onto
  it, and verify every widget returns the same values. Use when someone asks to
  prepare, optimize, slim down or make a data model ready for Sisense AI Assistant.
version: 2
requires_role: dataDesigner
tools:
  - datamodel.analyze_perspective_requirements
  - datamodel.create_perspective
  - datamodel.deploy_datamodel
  - datamodel.delete_perspective
  - dashboard.get_dashboards_by_datasource
  - dashboard.compare_dashboard_values
  - dashboard.replace_datasource
compensations:
  datamodel.create_perspective:
    tool: datamodel.delete_perspective
    args: {perspective: "{args.name}", datamodel: "{args.datamodel}"}
  dashboard.replace_datasource:
    tool: dashboard.replace_datasource
    args: {dashboard: "{args.dashboard}", datasource: "{result.previous_datasource_title}", act_as_owner: true}
step_labels:
  datamodel.analyze_perspective_requirements: Analysing which tables and columns your dashboards use
  dashboard.get_dashboards_by_datasource: Finding the dashboards on this model
  datamodel.create_perspective: Creating the perspective
  datamodel.deploy_datamodel: Building the model and waiting for the build to finish
  dashboard.compare_dashboard_values: Comparing every widget on the model and the perspective
  dashboard.replace_datasource: Moving the dashboard onto the perspective
  datamodel.delete_perspective: Removing the perspective
guardrails:
  - id: validate-before-swap
    rule: replace_datasource on a dashboard requires a prior compare_dashboard_values
      of that dashboard against the root model and the perspective with all_match true
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
   (`datamodel.analyze_perspective_requirements` with `detailed: true`, so the
   many-to-many pairs are reported). It returns two table lists
   in the exact `tables` shape step 3 takes: `perspective_tables` (the columns
   the dashboards use plus the join columns of the relation paths the query
   engine actually uses) and `perspective_tables_all_paths` (the same, keeping
   every equally short path — a safe superset). When a join path could not be
   determined it says so: the warnings carry an ambiguous_join_path count.
   Choose the list in code, never by hand — give step 3's `tables` as
   `steps[1].result.warnings.ambiguous_join_path != null ? steps[1].result.perspective_tables_all_paths : steps[1].result.perspective_tables`.
   Do NOT derive the column list from unused-column analysis: it omits the
   join columns and the perspective will build but its dashboards will fail at
   query time. `errors` lists dashboards that reference fields the model no
   longer has (`unresolved_reference`) or could not be exported. Those
   dashboards are already broken on the root model; a perspective neither
   fixes nor worsens them. Do NOT stop for them: proceed for the rest. They
   are not moved — the per-dashboard comparison cannot pass for them — and the
   report lists them with their owners so someone can repair them.
2. **Find every dashboard on the root model**
   (`dashboard.get_dashboards_by_datasource`). Each row carries `dashboard_id`,
   `title` and `owner_email` — note the owner. Steps 1 and 2 are the reads:
   they always come first, before anything is created, so the question and the
   approval can show what was found. If it finds NO dashboards there is nothing
   to move — stop and report it (they may already sit on a perspective). Make
   the create step and everything after it conditional on this list being
   non-empty (`steps[2].result != []`).
3. **Create the perspective** (`datamodel.create_perspective`) from step 1's
   payload. **The name must come from the user** — a perspective cannot be
   renamed once created, so never choose one. If the request and the earlier
   messages do not give a name, plan this step and everything after it as
   usual and put `name` under `args_ask` with the question "What should the
   perspective be called?" — the user is asked once the reads have run, and
   the plan then continues with the answer. It fails fast on a duplicate name;
   that is correct — the name is the user's, so the run stops and reports the
   clash, and the user asks again with a different name. Never adopt or rename.
4. **Build the ROOT model and wait for it** (`datamodel.deploy_datamodel` with
   the root model's title as `datamodel_name`, `build_type: "schema_changes"`
   and `wait: true`). A perspective is not queryable until its parent has been
   built after the change, and the build call only starts the build unless it
   waits — so `wait: true` is what makes step 5 safe to run right after.
   Building the perspective by name fails with "Elasticube not found".
5. **For each dashboard**, in order — nothing is copied or modified for the test:
   a. **Compare it on both datasources** (`dashboard.compare_dashboard_values`
      with the dashboard, `datasource_a` = the root model's title,
      `datasource_b` = the perspective's name). It runs every widget's own
      query against both and reports `all_match`, which is true only when at
      least one widget was compared and every compared widget returned the same
      values. It changes nothing.
   b. **Only if a reported `all_match == true`**, point the dashboard at the
      perspective (`dashboard.replace_datasource` with `act_as_owner: true`,
      so dashboards owned by other users are moved as well). Otherwise leave
      it exactly as it was and record which widgets differed or failed.

## Never
- Change a dashboard's owner, for any reason. Only the owner can publish; a
  dashboard the user cannot publish is REPORTED with its owner's name.
- Touch a dashboard on a different root model.
- Move a dashboard whose comparison did not come back `all_match` — including
  dashboards the analysis listed under `errors`; they stay and are reported.

## On failure
- Swap back every dashboard this run moved, to its `previous_datasource`.
- Leave the perspective in place if it built — it is harmless and the next
  run adopts it. Delete it (`datamodel.delete_perspective`) only if step 3 or 4
  is what failed.
- Report ran / failed (with the SDK's error verbatim) / not attempted.

## Ask
**{analyze_perspective_requirements.args.datamodel}** — what I found:

- Model: {analyze_perspective_requirements.result.summary.model_tables} tables, {analyze_perspective_requirements.result.summary.model_columns} columns
- Dashboards on it: {get_dashboards_by_datasource.count} — {get_dashboards_by_datasource.result[*].title|head:5}
- All these dashboards together need only **{analyze_perspective_requirements.result.summary.tables_required_in_perspective} tables and {analyze_perspective_requirements.result.summary.columns_required_in_perspective} columns** — the fields their widgets query, plus the join keys those queries rely on
- Random paths (two tables joinable more than one way): {!analyze_perspective_requirements.result.join_path_choices|count}**none found**{/}{?analyze_perspective_requirements.result.join_path_choices|count}**{analyze_perspective_requirements.result.join_path_choices|count} found** — {!analyze_perspective_requirements.result.warnings.ambiguous_join_path}**all resolved** using the path the query engine takes for the current widgets{/}{?analyze_perspective_requirements.result.warnings.ambiguous_join_path}**{analyze_perspective_requirements.result.warnings.ambiguous_join_path} unresolved** (every possible path kept for those); the rest resolved using the path the query engine takes for the current widgets{/}{/}
- Dashboards using fields the model no longer has: {!analyze_perspective_requirements.result.errors|count}**none**{/}{?analyze_perspective_requirements.result.errors|count}**{analyze_perspective_requirements.result.errors|count}** — already broken on the model; they stay where they are and are named in the summary at the end of the run{/}
- Many-to-many joins among the kept tables: {!analyze_perspective_requirements.result.warnings.many_to_many_in_perspective}**none**{/}{?analyze_perspective_requirements.result.warnings.many_to_many_in_perspective}**{analyze_perspective_requirements.result.many_to_many|pairs}**{/}

Optimizing for the Sisense AI Assistant means:

- Creating a **perspective**: a trimmed view of the model with exactly those tables and columns
- Moving the dashboards onto it — keeping everything they depend on is what lets them move without breaking
- Leaving out the rest, so the AI Assistant has less to sift through and answers better

A perspective can't be renamed after it is created. What should this one be called?

## Approval
Approving runs this on **{create_perspective.args.datamodel}**:

- Create the perspective **{create_perspective.args.name}** with {create_perspective.args.tables|count} tables{?analyze_perspective_requirements.result.warnings.ambiguous_join_path} (every candidate join path kept){/}
- Rebuild the model's schema so the perspective becomes queryable
- Check {get_dashboards_by_datasource.count} dashboard{get_dashboards_by_datasource.count|s} — {get_dashboards_by_datasource.result[*].title|head:10} — by running every widget against both the model and the perspective
- Move onto the perspective only the dashboards whose widgets return identical values, including dashboards owned by other users; any dashboard with a difference stays where it is and is listed in the report

Nothing is copied or changed for the check. If anything fails, the changes made in this run are undone.

Approve to proceed.

## Report
**{create_perspective.args.datamodel}** is now optimized for the Sisense AI Assistant.

- Perspective **{create_perspective.args.name}** was created with **{create_perspective.args.tables|count} tables** and built.
- {?replace_datasource.result|items_ran}**Now using the perspective:** {replace_datasource.result|items_ran}.{/}{!replace_datasource.result|items_ran}**No dashboard was moved** onto it.{/}{?replace_datasource.result|items_on_behalf}
- Dashboard Co-Authoring is enabled and you are not the owner of these dashboards, so their **shared copy** (what everyone sees) now uses the perspective while the owner's **private copy** is still on the root model until they open the dashboard: {replace_datasource.result|items_on_behalf}.{/}
- {?replace_datasource.result|items_skipped}**Left on the root model** because their widgets did not return the same values on the perspective: {replace_datasource.result|items_skipped}.{/}{!replace_datasource.result|items_skipped}**No dashboard was left behind** — every dashboard checked returned identical values on the perspective.{/}
- {?analyze_perspective_requirements.result.errors|count}**{analyze_perspective_requirements.result.errors|count} dashboard reference(s) to fields the model no longer has** — already broken, left as they were: {analyze_perspective_requirements.result.errors|head:5}.{/}{!analyze_perspective_requirements.result.errors|count}**No dashboard** uses fields the model no longer has.{/}
- {?analyze_perspective_requirements.result.warnings.many_to_many_in_perspective}**Many-to-many joins among the kept tables:** {analyze_perspective_requirements.result.many_to_many|pairs}.{/}{!analyze_perspective_requirements.result.warnings.many_to_many_in_perspective}**No many-to-many joins** among the kept tables.{/}
