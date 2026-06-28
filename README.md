<div align="center">

# 🔁 OpenLoop

### The open-source control plane & runtime for **team AI agents**

Persistent, asynchronous teammates that work across your channels — Slack,
Discord, Zoom, GitHub, Linear — each with their own memory, tools, model
policy, budget, approval rules, and audit trail.

<br/>

[![CI](https://github.com/p1c2u/openloop/actions/workflows/ci.yml/badge.svg)](https://github.com/p1c2u/openloop/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyopenloop?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/pyopenloop/)
[![Python](https://img.shields.io/pypi/pyversions/pyopenloop?logo=python&logoColor=white)](https://pypi.org/project/pyopenloop/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Model-agnostic](https://img.shields.io/badge/models-LiteLLM-7c3aed.svg)](https://github.com/BerriAI/litellm)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Status: early/WIP](https://img.shields.io/badge/status-early%2FWIP-orange.svg)](#)

<br/>

[**Why**](#why) ·
[**vs. Claude Tag**](#how-this-differs-from-claude-tag) ·
[**Features**](#what-it-does-mvp) ·
[**How it works**](#how-it-works) ·
[**Quickstart**](#quickstart-preliminary--commands-are-placeholders) ·
[**Roadmap**](#roadmap) ·
[**Contributing**](#contributing)

</div>

> **Status: early / WIP.** Not production-ready. APIs, config formats, and the
> commands below will change. Commands marked *(preliminary)* are placeholders.

## Why

Most AI assistants are personal and single-player. Real work happens across
shared **channels, threads, calls, repos, and issues** — and it spans more than
one tool. The useful primitive isn't "a Slack bot" — it's a persistent team
teammate that remembers your team's decisions, uses a scoped set of tools,
routes tasks to the right model, stays in budget, asks before doing anything
risky, and leaves an audit trail.

OpenLoop gives your team such an agent, reachable across the surfaces you
already use, with an open-source runtime you can self-host and inspect.
Slack/Discord/Zoom/GitHub/Linear are *surfaces* — the product is the agent
runtime and control plane behind them. Memory, tools, and policy can be scoped
per channel so context doesn't leak across teams.

**For:** engineering, data, and platform teams; AI-heavy startups; OSS
maintainers; anyone who wants Claude Tag–style workflows without single-vendor
lock-in.

## How this differs from Claude Tag

> Comparison based on Anthropic's [Claude Tag announcement](https://www.anthropic.com/news/introducing-claude-tag)
> (June 2026). Claude Tag is evolving quickly — check the source for current details.

OpenLoop shares Claude Tag's core model — persistent, asynchronous,
multiplayer team agents with tools and memory — and extends it in a
model-agnostic, configurable, multi-surface, open-source direction. The first
rows below are common ground; the rest are where OpenLoop goes further.

| Dimension | Claude Tag | OpenLoop |
| --- | --- | --- |
| **Persistent** | Long-lived agent that stays active and retains team context | Long-lived agent that stays active and retains team context |
| **Asynchronous** | Event- and mention-driven; works in the background, replies when ready | Event- and mention-driven; works in the background, replies when ready |
| **Multiplayer** | Shared in team channels, not a personal assistant | Shared across team surfaces, not a personal assistant |
| **Team context** | Channel-level Claude identities with scoped memories and shared conversations | Configurable memory scope across channel, agent, or workspace |
| **Tools** | Admin-approved tool access inside Claude's managed product | Explicit MCP/native tool allowlists, approval gates, and auditable actions |
| **Models** | Claude-native / Anthropic-first | Model-agnostic routing across Anthropic, OpenAI, Gemini, Ollama, OpenRouter, or any LiteLLM-compatible provider |
| **Configuration** | Admin-managed product settings | Config-as-code agents for surfaces, memory, model policy, tools, approvals, and budgets |
| **Surfaces** | Slack first, with other workplace integrations planned | Designed as a surface-agnostic runtime for Slack, Discord, GitHub, Linear, Zoom, and more |
| **Deployment** | Managed Anthropic experience | Self-hostable, inspectable, open-source infrastructure |
| **Best for** | Teams that want a managed Claude-centered teammate | Teams that want multiplayer agents with provider choice, deeper policy control, and open infrastructure |

## What it does (MVP)

- Summarize Slack/Discord threads and remember team decisions
- Create GitHub issues from discussions
- Investigate failing CI/builds
- Draft PRs via a coding worker, gated on human approval
- Track token/cost usage per channel, agent, task, and model
- Route tasks across providers (OpenAI, Anthropic, Gemini, Ollama, OpenRouter,
  any LiteLLM-compatible) — not only Claude
- Require human approval before risky write actions

## Concepts

| Concept | Meaning |
| --- | --- |
| **Workspace** | Top-level tenant; owns agents, integrations, policy defaults. |
| **Agent** | A team agent reachable across surfaces — the unit of identity, memory, and policy. |
| **Surface** | Where you reach the agent: Slack, Discord, Zoom, GitHub, Linear. |
| **Memory Scope** | What an agent recalls; can be scoped per channel so context doesn't leak across teams. |
| **Tool Policy** | Explicit allowlist of tools (MCP / native) and their permissions. |
| **Model Policy** | Which model handles which kind of task. |
| **Human Approval** | Gates requiring a human before write/risky actions. |
| **Budget** | Per-agent (optionally per-channel) spend and token limits, tracked and enforced. |

## How it works

```mermaid
flowchart TB
    subgraph SURFACES["🗣️ &nbsp;Surfaces &nbsp;<sub>(shared team channels)</sub>"]
        direction LR
        S1[Slack] ~~~ S2[Discord] ~~~ S3[Zoom] ~~~ S4[GitHub] ~~~ S5[Linear]
    end

    subgraph SELF["🏠 &nbsp;Self-hosted — your infra, your keys, inspectable"]
        RT["⚙️ &nbsp;Agent Runtime<br/><sub>shared team agent · loads policy · budget</sub>"]
        MG["🧠 &nbsp;Model Gateway<br/><sub>LiteLLM · model-agnostic</sub>"]
        TG["🔧 &nbsp;Tool Gateway<br/><sub>MCP + native · allowlist</sub>"]
        DB[("🗄️ &nbsp;Postgres + pgvector<br/><sub>channel-scoped memory · budget · audit</sub>")]
        HA{{"✋ &nbsp;Human approval"}}
    end

    subgraph MODELS["☁️ &nbsp;Model providers <sub>(external)</sub>"]
        direction TB
        M1[OpenAI] ~~~ M2[Anthropic] ~~~ M3[Gemini] ~~~ M4[Ollama] ~~~ M5[OpenRouter]
    end

    subgraph TOOLS["🛠️ &nbsp;Tool targets <sub>(external)</sub>"]
        direction LR
        T1[GitHub] ~~~ T2[CI] ~~~ T3[Coding worker]
    end

    SURFACES -- "mention / event" --> RT
    RT -- "model policy" --> MG
    RT -- "tool policy" --> TG
    RT <-. "read / write" .-> DB
    MG -- "your provider keys" --> MODELS
    TG -- "reads" --> TOOLS
    TG -- "write actions" --> HA
    HA -- "approved" --> TOOLS

    classDef runtime fill:#7c3aed,stroke:#5b21b6,color:#fff,font-weight:bold
    classDef gateway fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef store fill:#0ea5e9,stroke:#0369a1,color:#fff
    classDef gate fill:#f59e0b,stroke:#b45309,color:#1f2937,font-weight:bold
    class RT runtime
    class MG,TG gateway
    class DB store
    class HA gate
    style SELF fill:#faf5ff,stroke:#7c3aed,stroke-width:2px,stroke-dasharray:7 4
```

A mention or event on any surface triggers the team's agent. The runtime loads
its memory, tool policy, model policy, and budget; routes the task to a model;
uses only allowed tools; pauses for approval on write actions; then records
usage, cost, and outcome to the audit trail and memory.

## Example: Slack

```
@dev-platform  @agent summarize this thread and capture the decision

agent  📝 Summary (last 24 messages):
       • Moving ingestion off the legacy queue.
       • Redis Streams chosen over Kafka for v1 (lower ops overhead).
       • Owner: @priya, target: end of next sprint.

       🧠 Saved to channel memory: "Use Redis Streams for ingestion v1. Owner: @priya."
       Open a GitHub issue to track it?

@dev-platform  yes

agent  ⏳ Write action (create issue in openloop/ingestion) — approval required.
       @priya / @maintainers, approve?   💸 est. $0.004 · gpt-4o-mini
```

## Example: agent config

Agents are config-as-code (`apiVersion v1alpha1`, preliminary). Full example:
[`agents/dev-platform.yaml`](agents/dev-platform.yaml).

```yaml
apiVersion: openloop.ai/v1alpha1
kind: Agent
metadata: { name: dev-platform, workspace: acme }
spec:
  surfaces:
    - { type: slack, channel: "#dev-platform" }
    - { type: discord, channel: "eng" }
  memory: { scope: channel, backend: postgres, retention_days: 90 }
  model_policy:
    default: openai/gpt-4o-mini
    routes:
      - { match: { task: code }, model: anthropic/claude-sonnet-4-6 }
  tools:
    - { name: github, type: native, permissions: ["issues:write", "pulls:read"] }
  approvals:
    require_for: ["github.issues:write", "github.pulls:write"]
    approvers: ["@priya", "@maintainers"]
  budget: { monthly_usd: 50, per_task_usd: 0.50, on_exceeded: block }
```

## Architecture

| Layer | Now | Later |
| --- | --- | --- |
| API / backend | FastAPI | — |
| Agent runtime | durable workflow engine (worker + chat pipeline as workflows) | model-call replay/caching semantics; retry policies |
| Model gateway | LiteLLM | routing analytics |
| Tools | MCP gateway + native GitHub/Slack | more native connectors |
| Storage | Postgres + pgvector | — |
| Queue | Redis | — |
| Surfaces | Slack | Discord / Zoom / GitHub / Linear |
| Delivery | persisted Slack sessions, background progress/final postbacks, approval + thread-reply continuation, startup reconciler | provider idempotency keys, cross-process delivery locks, more surface adapters |
| Coding worker | draft PRs (clone → edit → push) | OpenHands-style |
| Dashboard | — | Next.js |
| Observability | — | OpenTelemetry / Langfuse traces |

The open-source runtime covers: agent runtime, Slack integration with async delivery, model adapters (LiteLLM), MCP tool gateway,
local Postgres channel/thread memory, approval flow, basic token/cost tracking,
Docker Compose deploy, and config-as-code agents.

## Quickstart *(preliminary — commands are placeholders)*

Requires Docker + Compose and at least one model provider key.

```bash
git clone https://github.com/p1c2u/openloop.git
cd openloop
cp .env.example .env          # set provider keys + Slack credentials
docker compose up -d          # Postgres + Redis + runtime
openloop agents apply -f agents/dev-platform.yaml
# invite the bot to a channel, then: @agent summarize this thread
```

Nothing leaves your machine except calls to the providers/tools you configure.

### Local development *(preliminary)*

The runtime is a Python/FastAPI app. Local dev uses [mise](https://mise.jdx.dev)
to pin the toolchain and manage a project virtualenv — no global installs.

```bash
mise install                 # Python 3.12 + uv + an auto-created .venv
mise run install             # install the runtime + dev deps (uv) into .venv
mise run test                # run the test suite
mise run dev                 # FastAPI runtime with autoreload on :8000
mise run apply -- -f agents/dev-platform.yaml   # validate an agent config
```

#### Try it against real Slack (Socket Mode)

Socket Mode opens an outbound WebSocket, so you can test a real mention →
reply → approval round-trip locally — no public URL or tunnel. In your `.env`
set a model key, `SLACK_BOT_TOKEN` (`xoxb-…`), and `SLACK_APP_TOKEN` (`xapp-…`,
created with the `connections:write` scope and Socket Mode enabled), then:

```bash
mise exec -- openloop slack socket   # connects; mention the bot in a channel
```

Read-only observability while it runs: `GET /usage` (spend vs. budget) and
`GET /audit` (recent token/cost records).

#### End-to-end tests

The E2E suite is layered so each slower or less deterministic dependency is
isolated.

**A. Unit and integration tests**: isolated logic tests plus in-process runtime,
memory, usage, tool gateway, approval, Slack, MCP, and HTTP API tests. External
providers, GitHub, and Postgres are faked, so this runs without credentials,
Docker, or network:

```bash
mise run test
mise run test-unit          # optional: only tests/unit
mise run test-integration   # optional: only tests/integration
```

**B. Postgres E2E**: real Postgres/pgvector stores, fake model and fake GitHub.
This validates SQL, asyncpg type handling, vector recall, approval persistence,
and usage persistence:

```bash
mise run test-e2e
```

**C. Runtime/GitHub live E2E**: real model, approval gate, and real GitHub issue
creation. It creates and then closes one issue, so it is explicitly opt-in:

```bash
export E2E_CONFIRM=1
export E2E_MODEL=groq/llama-3.3-70b-versatile
export GROQ_API_KEY=gsk_…             # or set another LiteLLM provider key
export GITHUB_TOKEN=github_pat_…       # issues:write on the repo
export E2E_GITHUB_REPO=you/sandbox
export DATABASE_URL=postgresql://openloop:change-me@localhost:5432/openloop  # optional
mise run test-e2e-runtime-github-live
```

You can also invoke the live pytest target directly:

```bash
E2E_LIVE=1 GITHUB_TOKEN=… E2E_GITHUB_REPO=you/sandbox \
  E2E_MODEL=groq/llama-3.3-70b-versatile GROQ_API_KEY=gsk_… \
  mise exec -- python -m pytest tests/e2e/test_runtime_github_live.py -v
```

GitHub Actions runs unit and integration tests on pull requests and pushes to `main`.
The runtime/GitHub live E2E workflow is separate, runs nightly or manually, and
uses a protected `live-e2e` environment. Configure repository secrets
`GROQ_API_KEY` and `LIVE_E2E_GITHUB_TOKEN`; the latter should be a fine-grained
token with issue write access to `openloop/openloop-e2e-sandbox`.

**D. Interactive Slack smoke** (a human in the loop).
With `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` set, run `openloop slack socket`,
mention the bot in a channel, and click **Approve** on a held write action.

**E. Automated Slack live E2E**: real Socket Mode round-trip with a stubbed
model, so it proves the Slack wire (auth, event delivery, in-thread reply)
without LLM nondeterminism. The triggering mention is posted with a **user**
token, not the bot token — Slack suppresses `app_mention` for messages an app
posts as itself, so a bot mentioning itself never fires the event. Add the
user-token scope `chat:write` (OAuth & Permissions → **User Token Scopes**) and
reinstall to get an `xoxp-…` token:

```bash
export E2E_LIVE=1
export SLACK_BOT_TOKEN=xoxb-…         # chat:write, app_mentions:read, channels:history
export SLACK_APP_TOKEN=xapp-…         # connections:write, Socket Mode enabled
export E2E_SLACK_USER_TOKEN=xoxp-…    # user-token chat:write — posts the mention
export E2E_SLACK_CHANNEL=C0…          # a channel the bot and that user are both in
export E2E_CONFIRM=1
mise run test-e2e-slack-live
```

The Slack live E2E GitHub Actions workflow is manual-only and uses a protected
`slack-live-e2e` environment. Configure environment secrets `SLACK_BOT_TOKEN`,
`SLACK_APP_TOKEN`, and `E2E_SLACK_USER_TOKEN`; set `E2E_SLACK_CHANNEL` as an
environment/repository variable, or pass a channel ID when manually dispatching
the workflow.

## Roadmap

- [x] Core async runtime + task pipeline
- [x] Slack surface (mentions, thread replies, approvals)
- [x] LiteLLM gateway + model-policy routing
- [x] MCP tool gateway + native GitHub connector
- [x] Channel/thread memory (Postgres + pgvector)
- [x] Human approval flow + token/cost tracking
- [x] Docker Compose + config-as-code
- [ ] Discord / Zoom / GitHub / Linear surfaces
- [x] Coding worker (draft PRs) — connector + approval gate + crash-resumable
- [x] Durable workflows — engine + approval-as-wait-node; worker resumes on crash;
  chat pipeline runs as a workflow (bounded: persisted turn state + idempotent
  writes; model calls are not replayed on crash)
- [x] Claude Tag-like Slack async delivery — persisted surface sessions, progress
  + final postbacks, approval/thread-reply continuation, startup reconciler,
  conversation-history threading (a follow-up turn replays the thread's prior
  exchanges), idempotency-keyed delivery (a crash between a successful post and
  recording its id is recovered by key instead of re-posting; best-effort, falls
  back to at-least-once if the surface lookup can't run), and delivery outside the
  original request lifecycle
- [x] Cross-process coordination — a distributed lock so that when several
  replicas boot together only one leads recovery, re-run on an interval so a
  leader that dies mid-sweep is healed by a survivor. `LOCK_BACKEND=auto` (default)
  uses Postgres advisory locks when the deploy already runs Postgres — no extra
  service — with Redis and process-local backends also available
- [ ] Hardening for full production parity — more surface adapters and an explicit
  model-call replay/caching policy
- [ ] Next.js dashboard, OTel/Langfuse tracing

## Scope

This repository is the **open-source runtime and control plane** — self-hosted,
inspectable, and the whole product for now.

It covers the agent runtime, model gateway (LiteLLM), MCP tool gateway + native
connectors, channel/thread memory, approval flow, token/cost tracking, Slack
async delivery, config-as-code, and Docker Compose deployment. The roadmap above
tracks what's planned.

## Security

Agents act on your behalf — treat their credentials like a production service
account.

- **Least privilege:** give each agent the narrowest tool scope it needs; never
  broad org-wide credentials.
- **Approve writes:** require human approval for creating issues/PRs, posting
  externally, or deleting anything.
- **Scoped tokens:** use fine-grained, per-integration tokens.
- **Memory isolation:** scope memory per channel so context doesn't leak.
- **Secrets:** keep them in `.env` or a secrets manager — never commit them.
- **Inspect:** self-host the runtime to see exactly what the agent does.

Early-stage software, no warranty. Don't connect sensitive production systems
until you've reviewed the threat model for your environment.

## Contributing

Early project — shaping the foundations is the most valuable work. Open an issue
before anything non-trivial. Help wanted on the runtime, model adapters, MCP
connectors, Slack surface, memory layer, and docs. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
