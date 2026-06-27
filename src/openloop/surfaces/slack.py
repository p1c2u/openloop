"""Slack surface.

Handles `app_mention` events with Phase D's async-delivery contract: a mention
creates a persisted :class:`~openloop.sessions.store.SurfaceSession`, the handler
posts a short in-thread progress message and returns fast, and the
:class:`~openloop.sessions.runner.SessionRunner` works the turn in the background,
posting the final answer (or an approval card) back to the thread later. Built on
slack-bolt's async app, exposed to FastAPI via the request handler.
"""

from __future__ import annotations

import asyncio
import logging
import re

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from openloop.runtime import Runtime, Task
from openloop.sessions import (
    SessionRunner,
    SlackSurfaceDelivery,
    SurfaceSessionStore,
    SurfaceTarget,
)
from openloop.surfaces.approvals import APPROVE_ACTION, DENY_ACTION

logger = logging.getLogger(__name__)

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _strip_mentions(text: str) -> str:
    return _MENTION_RE.sub("", text or "").strip()


def _approver_handle(user: dict) -> str:
    """Best-effort Slack identity → approver handle (e.g. '@priya')."""
    name = user.get("username") or user.get("name") or user.get("id", "")
    return f"@{name}" if name else "unknown"


def _target_from_event(runtime: Runtime, event: dict, thread_ts: str | None) -> SurfaceTarget:
    agent = runtime.agent
    return SurfaceTarget(
        surface="slack",
        workspace=agent.metadata.workspace,
        agent=agent.metadata.name,
        channel=event.get("channel"),
        thread=thread_ts,
        # The event ts is the idempotency key — Slack re-delivers the same event
        # on a delivery timeout, and the runner dedupes on it.
        event_id=event.get("event_ts") or event.get("ts"),
    )


async def handle_mention(runner: SessionRunner, event: dict, say) -> None:  # type: ignore[no-untyped-def]
    """Core ``app_mention`` logic: mention → session → background runner.

    Kept module-level (rather than a closure in :func:`build_slack_app`) so it
    can be driven directly with a synthetic event and a fake delivery — the full
    mention path without a live Slack connection. The runner owns progress/final
    delivery; only the empty-mention help reply uses ``say`` directly.
    """
    text = _strip_mentions(event.get("text", ""))
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not text:
        await say(text="Hi — mention me with a request.", thread_ts=thread_ts)
        return

    target = _target_from_event(runner.runtime, event, thread_ts)
    task = Task(
        text=text,
        surface="slack",
        channel=event.get("channel"),
        user=event.get("user"),
    )
    await runner.run(task, target)


def build_slack_app(
    runtime: Runtime,
    sessions: SurfaceSessionStore,
    *,
    bot_token: str,
    signing_secret: str | None = None,
) -> AsyncApp:
    """Build the Bolt app (mention + approval handlers) bound to a runtime.

    Shared by both transports: the FastAPI HTTP handler and Socket Mode. With
    no signing secret (Socket-Mode-only), request verification is disabled. A
    :class:`SessionRunner` over a :class:`SlackSurfaceDelivery` (bound to the
    app's web client) handles the async delivery.
    """
    if signing_secret:
        app = AsyncApp(token=bot_token, signing_secret=signing_secret)
    else:
        app = AsyncApp(token=bot_token, request_verification_enabled=False)

    runner = SessionRunner(runtime, sessions, SlackSurfaceDelivery(app.client))
    # Exposed so the app lifespan can repoint the runner's session store after a
    # Postgres-setup fallback (mirrors how the workflow engine's store is swapped)
    # — and so the approval handler can reach the runner in a later slice.
    app._session_runner = runner  # type: ignore[attr-defined]

    @app.event("app_mention")
    async def on_mention(event, say):  # type: ignore[no-untyped-def]
        # Return fast: the whole turn runs in the background so Slack's event
        # request isn't held open for the agent's (possibly long) work.
        asyncio.create_task(_run_mention(runner, event, say))

    async def _on_decision(ack, body, action, respond, approve):  # type: ignore[no-untyped-def]
        await ack()
        if runtime.tools is None:
            return
        approver = _approver_handle(body.get("user", {}))
        # The runner resolves the approval *and* continues the owning session,
        # posting the eventual answer back in the original thread.
        message = await runner.resolve_approval(
            action["value"], approver, approve=approve
        )
        await respond(text=message, replace_original=False)

    @app.action(APPROVE_ACTION)
    async def on_approve(ack, body, action, respond):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, respond, approve=True)

    @app.action(DENY_ACTION)
    async def on_deny(ack, body, action, respond):  # type: ignore[no-untyped-def]
        await _on_decision(ack, body, action, respond, approve=False)

    return app


async def _run_mention(runner: SessionRunner, event: dict, say) -> None:  # type: ignore[no-untyped-def]
    """Background wrapper around :func:`handle_mention` that swallows errors.

    The runner already records + delivers its own failures *once a session and
    its progress message exist*. This guard covers the earlier handoff steps
    (session-store write, the very first ``post_progress``) whose failure would
    otherwise leave the user staring at a mention that silently went nowhere — so
    on any escape it posts a best-effort error notice in-thread.
    """
    try:
        await handle_mention(runner, event, say)
    except Exception:
        logger.exception("Slack mention handling failed for event %s", event.get("ts"))
        thread_ts = event.get("thread_ts") or event.get("ts")
        try:
            await say(
                text="⚠️ Something went wrong starting that. Please try again.",
                thread_ts=thread_ts,
            )
        except Exception:
            logger.exception("failed to post mention-handoff error to the thread")


def build_slack_handler(
    runtime: Runtime,
    sessions: SurfaceSessionStore,
    *,
    bot_token: str,
    signing_secret: str,
) -> AsyncSlackRequestHandler:
    """Wrap the Bolt app in a FastAPI request handler (HTTP events transport)."""
    app = build_slack_app(
        runtime, sessions, bot_token=bot_token, signing_secret=signing_secret
    )
    return AsyncSlackRequestHandler(app)
