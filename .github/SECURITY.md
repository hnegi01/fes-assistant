# Security policy

## Reporting a vulnerability

Please report security issues **privately**, not as a public issue.

Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability). If that is unavailable, contact the
maintainers through your usual Sisense channel.

Include what you did, what you expected, what happened, and the commit or
release you tested. A minimal reproduction is worth more than a scanner
export.

## Scope

This repository is the FES Assistant application: a Streamlit UI, a FastAPI
backend, and an MCP server that calls the PySisense SDK.

**In scope:** anything in this repository, the published
`hnegi01/fes-{backend,mcp,ui}` images, and the release pipeline.

**Out of scope:** the Sisense platform itself, the PySisense SDK (report those
upstream), and any deployment's own infrastructure.

## What this application does and does not protect

Read `docs/security.md` for the full model. Four things matter most when
assessing a finding:

**The Sisense API token is the authorization boundary.** The agent can do
exactly what the supplied token permits, and nothing more. Credentials are
supplied per request and injected server-side; they are never stored on the
server and never read from the environment.

**The mutation approval gate is an oversight control, not an authorization
control.** Every write is gated by a dialog that names the operation and its
arguments, and an approval is single-use and bound to a dialog this server
actually issued, in that session, within a time limit. It cannot stop a
scripted client from driving its own dialog and answering it — that is what
the API is for. It stops writes that no dialog ever proposed.

**Summarization is a data-visibility switch.** With it off, Sisense result data
never reaches the model. It governs the model's view, not the screen.

**The tool allowlist is the surface.** A tool absent from
`config/allowed_tools.txt` is unreachable, enforced independently in three
places. A missing file means "no policy configured" and allows all tools, by
design and documented; a file that exists but cannot be read is a config error
and denies everything.

## What this repository does NOT ship

There is **no authentication and no TLS** in the reference deployment. The
shipped nginx config listens on port 80 with no auth. Putting an
authenticating, TLS-terminating layer in front of it is the operator's job, and
a deployment that skips it exposes an agent that can write to a BI platform.
See `docs/security.md`.
