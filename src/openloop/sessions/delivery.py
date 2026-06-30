"""Surface delivery — posting status, approvals, and final answers to a surface.

Phase D decouples the *answer* from the inbound request lifecycle. A
:class:`SurfaceDelivery` is the surface-agnostic seam the session runner uses to
set transient progress status, post approval cards when human input is needed,
and post the final answer (or an error) later — possibly long after the original
HTTP/Bolt request returned.

Delivery must be **idempotent**: the runner persists the ids returned here on the
session, so a crash-and-resume or a duplicate inbound event reuses the existing
approval/final message instead of posting a second one. The protocol returns the
provider message id from each durable post; ``update_approval`` takes one back.

That persisted id is the primary guard, but it leaves one window open: between a
provider accepting a post and the runner recording the returned id, a crash means
the id is lost and a retry can't tell the post already landed — at-least-once. To
close it, each post carries a deterministic ``key``: the post is *tagged* with the
key so a later attempt can find it, and when ``recover`` is set the implementation
first looks for an already-posted message with that key and returns its id instead
of posting a duplicate. Tagging is free (no extra call); the lookup runs only on
the recovery path, so the happy path is unaffected. Surfaces with no native dedup
(Slack) realize this best-effort and degrade to at-least-once if the lookup can't
run — the persisted id + startup reconciler remain the primary mechanism.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from openloop.sessions.store import SurfaceTarget

if TYPE_CHECKING:
    from openloop.approvals.store import ApprovalRequest

logger = logging.getLogger(__name__)


@runtime_checkable
class SurfaceDelivery(Protocol):
    async def set_progress_status(self, target: SurfaceTarget, text: str) -> None:
        """Set a transient, surface-native progress indicator.

        This is best-effort UI polish: implementations should not let missing
        provider support or transient provider errors block the actual answer.
        """
        ...

    async def update_approval(
        self,
        target: SurfaceTarget,
        message_id: str,
        text: str,
        requests: "list[ApprovalRequest]",
    ) -> None:
        """Update an existing approval card with buttons or resolution text.

        Editing the approval card in place keeps button handling tidy; the final
        answer is delivered separately so a cosmetic update failure cannot lose
        the outcome.
        """
        ...

    async def post_approval(
        self,
        target: SurfaceTarget,
        text: str,
        requests: "list[ApprovalRequest]",
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        """Post a new approval card with buttons; return its provider id.

        ``key`` tags the message for idempotent recovery; ``recover`` asks the
        implementation to first return an already-posted message with this key
        (closing the post-succeeded-but-id-lost window) rather than duplicating.
        """
        ...

    async def post_final(
        self, target: SurfaceTarget, text: str, *, blocks: list[dict] | None = None,
        key: str | None = None, recover: bool = False,
    ) -> str:
        """Post the final answer; return its provider id.

        See :meth:`post_approval` for ``key`` / ``recover``.
        """
        ...

    async def post_error(
        self, target: SurfaceTarget, text: str, *, key: str | None = None,
        recover: bool = False,
    ) -> str:
        """Post an error/interrupted notice; return its provider id.

        See :meth:`post_approval` for ``key`` / ``recover``.
        """
        ...


# Slack has no native idempotency key on chat.postMessage, so we tag each posted
# message with the delivery key in Slack `metadata` and, on recovery, scan the
# thread for a message already bearing it. Bounded scan: only the most recent page
# is checked — enough for the crash-retry window (our message is among the latest),
# best-effort for very long threads. Needs the bot's `*:history` read scope; if it
# lacks it (or the call fails) the lookup degrades to None and we post fresh.
_DELIVERY_EVENT_TYPE = "openloop_delivery"
_LOOKUP_LIMIT = 200


class SlackSurfaceDelivery:
    """Delivers to Slack via a Bolt/`AsyncWebClient` ``client``.

    Uses the stored channel + thread and message timestamps Slack returns so the
    runner can dedupe and update. Threading is best-effort: if a target has no
    thread it posts at the channel root.
    """

    def __init__(self, client) -> None:  # AsyncWebClient
        self.client = client

    @staticmethod
    def _metadata(key: str | None) -> dict | None:
        """Slack message metadata that tags a post with its delivery key."""
        if not key:
            return None
        return {"event_type": _DELIVERY_EVENT_TYPE, "event_payload": {"key": key}}

    async def _find_by_key(self, target: SurfaceTarget, key: str) -> str | None:
        """Return the ts of an already-posted message tagged with ``key``, if any.

        Scans the thread (or channel root) for a message carrying our delivery
        metadata. Defensive: any failure (missing history scope, transient error)
        degrades to ``None`` so the caller posts fresh rather than crashing.
        """
        try:
            if target.thread:
                resp = await self.client.conversations_replies(
                    channel=target.channel,
                    ts=target.thread,
                    include_all_metadata=True,
                    limit=_LOOKUP_LIMIT,
                )
            else:
                resp = await self.client.conversations_history(
                    channel=target.channel,
                    include_all_metadata=True, limit=_LOOKUP_LIMIT,
                )
        except Exception:  # noqa: BLE001 — best-effort dedup; fall back to posting
            logger.warning(
                "delivery idempotency lookup failed for key %s; posting fresh",
                key, exc_info=True,
            )
            return None
        for msg in resp.get("messages", []) or []:
            md = msg.get("metadata") or {}
            payload = md.get("event_payload") or {}
            if (
                md.get("event_type") == _DELIVERY_EVENT_TYPE
                and payload.get("key") == key
            ):
                return msg.get("ts")
        return None

    async def set_progress_status(self, target: SurfaceTarget, text: str) -> None:
        """Set Slack's assistant-thread status, e.g. "<App> is thinking..."."""
        if not target.channel or not target.thread:
            return
        try:
            setter = getattr(self.client, "assistant_threads_setStatus", None)
            if setter is not None:
                await setter(
                    channel_id=target.channel,
                    thread_ts=target.thread,
                    status=text,
                    loading_messages=[text],
                )
            else:
                await self.client.api_call(
                    "assistant.threads.setStatus",
                    json={
                        "channel_id": target.channel,
                        "thread_ts": target.thread,
                        "status": text,
                        "loading_messages": [text],
                    },
                )
        except Exception:  # noqa: BLE001 — status must never block delivery
            logger.warning(
                "failed to set Slack assistant status for thread %s",
                target.thread,
                exc_info=True,
            )

    async def _post(
        self,
        target: SurfaceTarget,
        text: str,
        *,
        blocks: list[dict] | None,
        key: str | None,
        recover: bool,
    ) -> str:
        """Tagged, idempotent post: recover an existing keyed message or post one."""
        if key and recover:
            existing = await self._find_by_key(target, key)
            if existing is not None:
                return existing
        resp = await self.client.chat_postMessage(
            channel=target.channel,
            thread_ts=target.thread,
            text=text,
            blocks=blocks,
            metadata=self._metadata(key),
        )
        return resp["ts"]

    async def update_approval(self, target, message_id, text, requests) -> None:
        # Local import keeps the Block Kit helper out of the surface-agnostic core.
        from openloop.surfaces.approvals import approval_blocks

        await self.client.chat_update(
            channel=target.channel,
            ts=message_id,
            text=text,
            blocks=approval_blocks(requests),
        )

    async def post_approval(
        self, target, text, requests, *, key=None, recover=False
    ) -> str:
        # Local import keeps the Block Kit helper out of the surface-agnostic core.
        from openloop.surfaces.approvals import approval_blocks

        return await self._post(
            target,
            text,
            blocks=approval_blocks(requests),
            key=key,
            recover=recover,
        )

    async def post_final(
        self,
        target: SurfaceTarget,
        text: str,
        *,
        blocks: list[dict] | None = None,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        return await self._post(target, text, blocks=blocks, key=key, recover=recover)

    async def post_error(
        self,
        target: SurfaceTarget,
        text: str,
        *,
        key: str | None = None,
        recover: bool = False,
    ) -> str:
        return await self._post(target, text, blocks=None, key=key, recover=recover)
