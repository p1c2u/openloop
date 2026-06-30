"""Unit tests for SlackSurfaceDelivery's idempotency-key dedup (Finding #1).

Closes the post-succeeded-but-id-lost window: each post is tagged with a
deterministic key in Slack `metadata`, and a recovery post looks the thread up by
that key so a crashed first attempt isn't duplicated. The happy path never scans;
a missing history scope degrades to posting fresh.
"""

import pytest

from openloop.sessions import SlackSurfaceDelivery, SurfaceTarget

pytestmark = pytest.mark.unit


class FakeSlackClient:
    """Minimal AsyncWebClient stand-in: records posts, serves metadata back."""

    def __init__(self, *, lookup_error: bool = False) -> None:
        self.posted: list[dict] = []
        self.statuses: list[dict] = []
        self.lookups = 0
        self.lookup_error = lookup_error
        self._seq = 0

    async def chat_postMessage(
        self, *, channel, thread_ts=None, text=None, blocks=None, metadata=None
    ):
        self._seq += 1
        ts = f"{self._seq}.0001"
        self.posted.append(
            {
                "channel": channel,
                "thread_ts": thread_ts,
                "text": text,
                "blocks": blocks,
                "metadata": metadata,
                "ts": ts,
            }
        )
        return {"ts": ts}

    async def assistant_threads_setStatus(
        self, *, channel_id, thread_ts, status, loading_messages=None, **kwargs
    ):
        if self.lookup_error:
            raise RuntimeError("missing assistant scope")
        self.statuses.append(
            {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "status": status,
                "loading_messages": loading_messages,
            }
        )
        return {"ok": True}

    async def conversations_replies(
        self, *, channel, ts, include_all_metadata=False, limit=200
    ):
        self.lookups += 1
        if self.lookup_error:
            raise RuntimeError("missing channels:history scope")
        msgs = [
            {"ts": p["ts"], "metadata": p["metadata"], "text": p["text"]}
            for p in self.posted
            if p["channel"] == channel and (p["thread_ts"] == ts or p["ts"] == ts)
        ]
        return {"messages": msgs}

    async def conversations_history(
        self, *, channel, include_all_metadata=False, limit=200
    ):
        self.lookups += 1
        if self.lookup_error:
            raise RuntimeError("missing channels:history scope")
        msgs = [
            {"ts": p["ts"], "metadata": p["metadata"]}
            for p in self.posted
            if p["channel"] == channel
        ]
        return {"messages": msgs}


class ApiCallOnlySlackClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def api_call(self, api_method, *, json=None, **kwargs):
        self.calls.append({"api_method": api_method, "json": json})
        return {"ok": True}


def _target(thread="100.1"):
    return SurfaceTarget(
        surface="slack", workspace="w", agent="a", channel="C1", thread=thread
    )


async def test_tagged_post_records_metadata_without_scanning():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_final(_target(), "answer", key="s1:final")

    # Happy path: the message is tagged for later recovery, but no lookup runs.
    assert client.lookups == 0
    assert ts == client.posted[0]["ts"]
    md = client.posted[0]["metadata"]
    assert md["event_type"] == "openloop_delivery"
    assert md["event_payload"]["key"] == "s1:final"


async def test_progress_status_uses_assistant_thread_indicator():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.statuses == [
        {
            "channel_id": "C1",
            "thread_ts": "100.1",
            "status": "is thinking...",
            "loading_messages": ["is thinking..."],
        }
    ]
    assert client.posted == []


async def test_progress_status_fallback_sets_loading_messages():
    client = ApiCallOnlySlackClient()
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.calls == [
        {
            "api_method": "assistant.threads.setStatus",
            "json": {
                "channel_id": "C1",
                "thread_ts": "100.1",
                "status": "is thinking...",
                "loading_messages": ["is thinking..."],
            },
        }
    ]


async def test_progress_status_failure_is_non_blocking():
    client = FakeSlackClient(lookup_error=True)
    delivery = SlackSurfaceDelivery(client)

    await delivery.set_progress_status(_target(), "is thinking...")

    assert client.statuses == []


async def test_recover_returns_existing_message_instead_of_duplicating():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    first = await delivery.post_final(_target(), "answer", key="s1:final")
    # The crash-retry path finds the tagged message and returns its id.
    again = await delivery.post_final(
        _target(), "answer", key="s1:final", recover=True
    )

    assert again == first
    assert len(client.posted) == 1  # no duplicate
    assert client.lookups == 1


async def test_recover_at_channel_root_uses_history():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)
    target = _target(thread=None)

    first = await delivery.post_final(target, "answer", key="k")
    again = await delivery.post_final(target, "answer", key="k", recover=True)

    assert again == first
    assert len(client.posted) == 1


async def test_recover_posts_fresh_when_no_prior_message():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_final(_target(), "answer", key="s1:final", recover=True)

    assert len(client.posted) == 1
    assert ts == client.posted[0]["ts"]


async def test_lookup_failure_degrades_to_posting():
    # Missing history scope / transient error must not crash delivery — post fresh
    # and fall back to at-least-once rather than dropping the answer.
    client = FakeSlackClient(lookup_error=True)
    delivery = SlackSurfaceDelivery(client)

    ts = await delivery.post_error(_target(), "boom", key="s1:error", recover=True)

    assert len(client.posted) == 1
    assert ts == client.posted[0]["ts"]


async def test_unkeyed_posts_never_dedupe():
    client = FakeSlackClient()
    delivery = SlackSurfaceDelivery(client)

    a = await delivery.post_final(_target(), "x")
    b = await delivery.post_final(_target(), "x")

    assert a != b
    assert len(client.posted) == 2
    assert client.posted[0]["metadata"] is None
