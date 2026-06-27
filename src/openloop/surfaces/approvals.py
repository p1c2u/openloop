"""Surface-agnostic approval UI helpers (Slack Block Kit + resolution).

Kept separate from the Bolt wiring so the rendering and resolution logic can be
unit-tested without constructing a Slack app or talking to Slack.
"""

from __future__ import annotations

from openloop.approvals.store import ApprovalRequest
from openloop.tools import ToolGateway

APPROVE_ACTION = "openloop_approve"
DENY_ACTION = "openloop_deny"


def approval_blocks(requests: list[ApprovalRequest]) -> list[dict]:
    """Block Kit for one or more pending write actions, each with buttons."""
    blocks: list[dict] = []
    for req in requests:
        approvers = ", ".join(req.approvers) or "any approver"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⏳ *Approval required:* {req.summary}\n_{approvers}_",
                },
            }
        )
        blocks.append(
            {
                "type": "actions",
                "block_id": f"approval:{req.id}",
                "elements": [
                    {
                        "type": "button",
                        "action_id": APPROVE_ACTION,
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": req.id,
                    },
                    {
                        "type": "button",
                        "action_id": DENY_ACTION,
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "value": req.id,
                    },
                ],
            }
        )
    return blocks


def resolution_message(inv, approver: str) -> str:
    """Status line for a resolved approval — shared by the button reply and the
    session continuation so they never drift."""
    if inv.status == "executed":
        detail = inv.result.summary if inv.result else (inv.message or "done")
        return f"✅ Approved by {approver} — {detail}"
    if inv.status == "denied":
        return f"🚫 Denied by {approver}."
    # forbidden (not an approver / unknown / already resolved)
    return f"⛔ {inv.message}"


async def resolve_from_action(
    gateway: ToolGateway, approval_id: str, approver: str, *, approve: bool
) -> str:
    """Resolve an approval from a button click; return a status message."""
    inv = await gateway.resolve(approval_id, approver, approve=approve)
    return resolution_message(inv, approver)
