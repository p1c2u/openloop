"""Native coding-worker connector — opens *draft* PRs from an instruction.

Exposes a single write action, ``coding_worker.pr:write``. On execution it runs a
pluggable :class:`CodingWorker` (clone → model-edit → commit → push) and then
opens a **draft** pull request from the pushed branch.

Phase A runs the whole pipeline inside :meth:`execute`, **after** approval. Two
distinct gates, never conflated:

1. the human **approval** lets the worker **start**;
2. the **draft PR itself** is the review gate before **merge**.

There is no "approve the generated diff before opening the PR" step here — the
approval summary says *run worker + open draft PR* and must never imply diff
review.

``job_id`` is the stable thread through the whole system. It is minted in
:meth:`prepare_args` **before** the approval is created, so it is carried in the
approval args, the worker state, the branch name + idempotency keys, and the
final PR metadata — giving one identity through the whole system.

**Phase B — durability.** Given a :class:`CheckpointStore`, the connector
persists a checkpoint after each named step and resumes from it on a mid-flight
crash. The two durable side effects (branch push, PR open) are made idempotent so
a replay never duplicates them: a pushed branch is detected via ``completed_steps``
(the local sandbox is ephemeral, so only the push survives a crash), and an
already-open PR is reused via :meth:`GitHubClient.find_pull`. Without a store the
connector behaves exactly as in Phase A (outcome-only, no resume).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from openloop.checkpoints.store import CheckpointStore, WorkerCheckpoint
from openloop.tools.base import ActionSpec, ToolResult
from openloop.tools.github import GitHubClient

# Persist-after-each-step callback the worker invokes so a crash leaves an
# accurate mid-phase record (completed_steps + state_json), not just a status.
StepCallback = Callable[["WorkerState"], Awaitable[None]]

logger = logging.getLogger(__name__)

_REPO = {"type": "string", "description": "owner/repo, e.g. acme/ingestion"}

# Named steps the worker walks through. In code now; checkpointed in Phase B.
STEPS = ("clone", "branch", "edit", "commit", "push")


@dataclass(slots=True)
class WorkerState:
    """Worker progress for one job — serialized into a checkpoint's state_json.

    ``title`` / ``body`` are filled once the worker generates the change so they
    survive in the checkpoint; on resume after a push they let the PR be opened
    without re-running the worker.
    """

    job_id: str
    repo: str
    instruction: str
    base: str
    branch: str
    completed_steps: list[str] = field(default_factory=list)
    title: str | None = None
    body: str | None = None

    def push_key(self) -> str:
        """Idempotency key for the branch push — never a single global key."""
        return f"{self.job_id}:push:{self.branch}"

    def open_pr_key(self) -> str:
        """Idempotency key for opening the PR."""
        return f"{self.job_id}:open_pr:{self.repo}:{self.branch}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerState":
        fields = {
            "job_id", "repo", "instruction", "base", "branch",
            "completed_steps", "title", "body",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass(slots=True)
class WorkerOutcome:
    """What the worker produced: a pushed branch ready for a draft PR.

    Carries the model spend so it is at least *observable* in the tool result.
    NOTE: this runs inside ``ToolGateway.resolve()``, which is outside
    ``Runtime.handle``'s usage accounting — so this spend is not yet recorded in
    ``/usage`` nor checked against per-task/monthly budgets. Enforcing that means
    threading a UsageStore + agent scope through the approval-resolution path,
    which lands with Phase B/C (where approval becomes an event on the workflow).
    """

    branch: str
    title: str
    body: str
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class CodingWorker(Protocol):
    """Clones a repo, applies model-generated edits, commits and pushes.

    Implementations must push ``state.branch`` and return a :class:`WorkerOutcome`
    describing the PR to open. They own the ``state.push_key()`` idempotency key,
    set ``state.title`` / ``state.body`` once generated, and call ``on_step`` (if
    given) after appending each completed step so progress is checkpointed.
    """

    async def run(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome: ...


def _branch_for(job_id: str) -> str:
    return f"openloop/job-{job_id}"


def _failed(
    job_id: str, state: WorkerState, status: str, exc: Exception
) -> ToolResult:
    """A failed outcome for a worker/PR step — never raised out of execute()."""
    return ToolResult(
        ok=False,
        summary=f"coding worker job {job_id} {status}: {exc}",
        data={
            "job_id": job_id,
            "status": status,
            "branch": state.branch,
            "completed_steps": state.completed_steps,
            "error": str(exc),
        },
    )


def _opened_result(cp: WorkerCheckpoint) -> ToolResult:
    """Reconstruct the success result from a checkpoint of an already-opened PR."""
    return ToolResult(
        ok=True,
        summary=(
            f"draft PR #{cp.pr_number} already open in {cp.repo} (job {cp.job_id})"
        ),
        data={
            "job_id": cp.job_id,
            "status": "opened",
            "branch": cp.branch,
            "pr_number": cp.pr_number,
            "pr_url": cp.pr_url,
            "completed_steps": cp.completed_steps,
            "resumed": True,
        },
    )


def _pr_body(body: str, job_id: str) -> str:
    """Stamp the job id into the PR body so the identity survives in GitHub."""
    body = (body or "").rstrip()
    footer = f"---\n🤖 Opened by the OpenLoop coding worker · job `{job_id}`"
    return f"{body}\n\n{footer}" if body else footer


class CodingWorkerConnector:
    """Maps ``coding_worker.pr:write`` onto a worker + a :class:`GitHubClient`."""

    name = "coding_worker"
    # When the gateway has a WorkflowEngine, this action runs as a durable
    # workflow (approval = wait node). Without one, execute() below is the Phase B
    # fallback path (checkpoint-based resume). Kept in sync with WORKFLOW_NAME in
    # openloop.workflows.coding_worker.
    workflow = "coding_worker"

    def __init__(
        self,
        worker: CodingWorker,
        github: GitHubClient,
        checkpoints: "CheckpointStore | None" = None,
    ) -> None:
        self.worker = worker
        self.github = github
        # Optional: when set, jobs are checkpointed per step and resume on crash.
        self.checkpoints = checkpoints

    def supported_permissions(self) -> set[str]:
        return {"pr:write"}

    def prepare_args(self, permission: str, args: dict) -> dict:
        """Mint ``job_id`` before approval so it threads the whole system.

        Called by the gateway prior to creating the approval request, so the id
        is persisted in the approval args and reused verbatim at execute time.
        """
        if permission == "pr:write" and not args.get("job_id"):
            args = {**args, "job_id": uuid.uuid4().hex[:12]}
        return args

    def describe(self, permission: str) -> ActionSpec:
        return ActionSpec(
            "Run the coding worker on an instruction and open a draft pull "
            "request with its changes. This starts the worker and opens a draft "
            "PR for review; it does not merge.",
            {
                "type": "object",
                "properties": {
                    "repo": _REPO,
                    "instruction": {
                        "type": "string",
                        "description": "what change the worker should make",
                    },
                    "base": {
                        "type": "string",
                        "description": "branch to open the PR against (default main)",
                    },
                },
                "required": ["repo", "instruction"],
            },
        )

    async def execute(self, permission: str, args: dict) -> ToolResult:
        if permission != "pr:write":
            return ToolResult(ok=False, summary=f"unsupported permission {permission}")

        job_id = args.get("job_id") or uuid.uuid4().hex[:12]
        base = args.get("base", "main")

        # Resume from a checkpoint if one exists (re-invocation after a crash, or
        # an approval re-resolved). Otherwise start fresh from the request args.
        cp = await self.checkpoints.get(job_id) if self.checkpoints else None
        if cp is not None and cp.status == "opened":
            # Idempotent: the draft PR already exists — never open a second one.
            return _opened_result(cp)

        if cp is not None:
            state = WorkerState.from_dict(cp.state_json)
        else:
            state = WorkerState(
                job_id=job_id,
                repo=args["repo"],
                instruction=args["instruction"],
                base=base,
                branch=_branch_for(job_id),
            )

        # The two side effects run after resolve() already marked the approval
        # approved, so neither may raise out of execute() — that would surface as
        # a generic error with no failed ToolResult. Record the failure instead.
        cost = (0.0, 0, 0)
        if "push" not in state.completed_steps:
            # The sandbox is ephemeral, so local steps (clone…commit) can't resume
            # from a crash — only the push survives. Re-run the worker from clean.
            state.completed_steps = []
            await self._save(state, "running")
            try:
                outcome = await self.worker.run(state, on_step=self._checkpointer())
            except Exception as exc:  # noqa: BLE001
                await self._save(state, "failed", error=str(exc))
                return _failed(job_id, state, "failed", exc)
            state.title, state.body = outcome.title, outcome.body
            cost = (outcome.cost_usd, outcome.prompt_tokens, outcome.completion_tokens)
            await self._save(state, "pushed")
        else:
            # Branch already pushed in an earlier run; just (re)open the PR.
            outcome = WorkerOutcome(
                branch=state.branch,
                title=state.title or "Automated change",
                body=state.body or "",
            )

        try:
            pull = await self._open_pr(state, outcome)
        except Exception as exc:  # noqa: BLE001
            await self._save(state, "open_pr_failed", error=str(exc))
            return _failed(job_id, state, "open_pr_failed", exc)

        await self._save(
            state, "opened", pr_number=pull.get("number"), pr_url=pull.get("html_url")
        )
        return ToolResult(
            ok=True,
            summary=(
                f"opened draft PR #{pull.get('number')} in {state.repo} "
                f"(job {job_id})"
            ),
            data={
                "job_id": job_id,
                "status": "opened",
                "branch": outcome.branch,
                "pr_number": pull.get("number"),
                "pr_url": pull.get("html_url"),
                "completed_steps": state.completed_steps,
                # Observable spend only — not yet enforced (see WorkerOutcome).
                "cost_usd": cost[0],
                "prompt_tokens": cost[1],
                "completion_tokens": cost[2],
                "idempotency_keys": {
                    "push": state.push_key(),
                    "open_pr": state.open_pr_key(),
                },
            },
        )

    async def _open_pr(self, state: WorkerState, outcome: WorkerOutcome) -> dict:
        """Open the draft PR, reusing an existing one for this head if present.

        The base comes from ``state.base`` (the checkpoint), not the request args:
        a resume that passes only ``job_id`` must still target the job's original
        base, never silently fall back to ``main``.
        """
        existing = await self.github.find_pull(state.repo, head=outcome.branch)
        if existing is not None:
            return existing
        return await self.github.create_pull(
            repo=state.repo,
            head=outcome.branch,
            base=state.base,
            title=outcome.title,
            body=_pr_body(outcome.body, state.job_id),
            draft=True,
        )

    # Statuses that are done (no resume) vs. interrupted (resume on startup).
    _TERMINAL = ("opened", "failed")

    async def resume_incomplete(self) -> list[str]:
        """Re-drive jobs left non-terminal by a crash. Call once at startup.

        The approval path (:meth:`ToolGateway.resolve`) marks the approval
        ``approved`` *before* :meth:`execute` runs and will not re-invoke it, so a
        crash mid-execute would otherwise strand the job — the resume logic in
        ``execute`` is unreachable through the normal path. This reconciler drives
        resume directly off the checkpoints instead: ``execute`` is idempotent
        (checkpoints + force-push + ``find_pull``), so finishing or restarting each
        job is safe.

        Across replicas, the app lifespan runs this under a ``startup-recovery``
        :class:`~openloop.coordination.DistributedLock` so only the leader resumes
        jobs; ``execute`` itself stays idempotent if two ever overlap. Phase C
        folds this into the workflow engine, where approval is an event and
        ``resolve`` is a thin adapter.
        """
        if self.checkpoints is None:
            return []
        resumed: list[str] = []
        for cp in await self.checkpoints.recent(limit=1000):
            if cp.status in self._TERMINAL:
                continue
            logger.info("resuming coding-worker job %s (was %s)", cp.job_id, cp.status)
            await self.execute(
                "pr:write",
                {
                    "job_id": cp.job_id,
                    "repo": cp.repo,
                    "instruction": cp.instruction,
                    "base": cp.base,
                },
            )
            resumed.append(cp.job_id)
        return resumed

    def _checkpointer(self) -> StepCallback | None:
        """A per-step callback that persists progress, or None when no store."""
        if self.checkpoints is None:
            return None

        async def on_step(state: WorkerState) -> None:
            await self._save(state, "running")

        return on_step

    async def _save(
        self,
        state: WorkerState,
        status: str,
        *,
        pr_number: int | None = None,
        pr_url: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.checkpoints is None:
            return
        await self.checkpoints.upsert(
            WorkerCheckpoint(
                job_id=state.job_id,
                repo=state.repo,
                instruction=state.instruction,
                base=state.base,
                branch=state.branch,
                status=status,
                completed_steps=list(state.completed_steps),
                state_json=state.to_dict(),
                title=state.title,
                body=state.body,
                pr_number=pr_number,
                pr_url=pr_url,
                error=error,
            )
        )


@runtime_checkable
class _Completer(Protocol):
    async def complete(self, model: str, messages: list[dict], **kwargs): ...


class GitCodingWorker:
    """Real worker: clone → model-edit → commit → push, in a temp sandbox.

    SECURITY: this runs model-generated edits, so it needs a least-privilege
    ``contents:write`` token and an isolated checkout. The clone happens in a
    throwaway temp dir that is removed after each run. Edits are applied as a
    unified diff via ``git apply`` — anything that doesn't apply cleanly fails
    the job rather than being force-written.

    Phase A only: no checkpointing. A crash mid-run loses the sandbox and leaves
    the approval stuck approved-but-incomplete — Phase B adds resume.
    """

    def __init__(
        self,
        token: str,
        model: str,
        gateway: _Completer | None = None,
        *,
        max_context_bytes: int = 60_000,
    ) -> None:
        self.token = token
        self.model = model
        self._gateway = gateway
        self.max_context_bytes = max_context_bytes

    def _completer(self) -> _Completer:
        if self._gateway is None:
            from openloop.models.gateway import ModelGateway

            self._gateway = ModelGateway()
        return self._gateway

    async def run(
        self, state: WorkerState, on_step: StepCallback | None = None
    ) -> WorkerOutcome:
        async def step(name: str) -> None:
            state.completed_steps.append(name)
            if on_step is not None:
                await on_step(state)

        sandbox = Path(tempfile.mkdtemp(prefix=f"openloop-{state.job_id}-"))
        try:
            url = (
                f"https://x-access-token:{self.token}@github.com/{state.repo}.git"
            )
            await self._git(
                "clone", "--depth", "1", "--branch", state.base, url, str(sandbox)
            )
            await step("clone")

            await self._git("checkout", "-b", state.branch, cwd=sandbox)
            await step("branch")

            diff, title, body, resp = await self._generate(state, sandbox)
            # Persist title/body so a post-push crash can still open the PR.
            state.title, state.body = title, body
            await self._git_input("apply", "--whitespace=nowarn", stdin=diff, cwd=sandbox)
            await step("edit")

            await self._git("add", "-A", cwd=sandbox)
            await self._git(
                "-c", "user.email=worker@openloop.ai",
                "-c", "user.name=OpenLoop coding worker",
                "commit", "-m", title, cwd=sandbox,
            )
            await step("commit")

            # Force-push to the job-exclusive branch so the push is idempotent.
            # The "push" checkpoint is only written by the step() below, so a crash
            # in the window between this push succeeding and that write leaves the
            # checkpoint saying "not pushed". Resume then re-runs the worker from
            # clean and pushes again — without --force that second push is rejected
            # as a non-fast-forward (the branch already exists). The branch is
            # owned solely by this job_id, so overwriting it is safe; the trade-off
            # is that a resumed run may carry a freshly regenerated diff.
            await self._git(
                "push", "--force", "origin", state.branch, cwd=sandbox
            )
            await step("push")

            return WorkerOutcome(
                branch=state.branch,
                title=title,
                body=body,
                cost_usd=resp.cost_usd,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    async def _generate(self, state: WorkerState, sandbox: Path):
        """Ask the model for a unified diff + PR title/body for the instruction.

        Returns ``(diff, title, body, response)`` — the response carries token
        counts and cost so the caller can surface the worker's model spend.
        """
        context = self._repo_context(sandbox)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding worker. Given a repository snapshot and an "
                    "instruction, produce changes as a single unified diff that "
                    "applies cleanly with `git apply` from the repo root. Respond "
                    "with exactly three sections, each on its own line and in this "
                    "order:\nTITLE: <one-line PR title>\nBODY: <short PR description>\n"
                    "DIFF:\n<the unified diff>\nDo not wrap the diff in markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{state.instruction}\n\n"
                    f"Repository {state.repo} (base {state.base}):\n{context}"
                ),
            },
        ]
        resp = await self._completer().complete(self.model, messages)
        diff, title, body = _parse_generation(resp.text)
        return diff, title, body, resp

    def _repo_context(self, sandbox: Path) -> str:
        """A best-effort, size-capped snapshot of tracked text files."""
        parts: list[str] = []
        budget = self.max_context_bytes
        for path in sorted(sandbox.rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            try:
                text = path.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(sandbox)
            block = f"\n=== {rel} ===\n{text}"
            if len(block) > budget:
                break
            parts.append(block)
            budget -= len(block)
        return "".join(parts)

    async def _git(self, *args: str, cwd: Path | None = None) -> str:
        return await self._run("git", *args, cwd=cwd)

    async def _git_input(self, *args: str, stdin: str, cwd: Path | None = None) -> str:
        return await self._run("git", *args, cwd=cwd, stdin=stdin)

    async def _run(
        self, *cmd: str, cwd: Path | None = None, stdin: str | None = None
    ) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(
            stdin.encode() if stdin is not None else None
        )
        if proc.returncode != 0:
            # Redact the token from BOTH the command and git's stderr — git prints
            # the full https://x-access-token:<token>@github.com/... URL on many
            # failures, and this text is returned in the failed ToolResult.
            cmd_str = self._redact(" ".join(cmd))
            stderr = self._redact(err.decode().strip())
            raise RuntimeError(f"`{cmd_str}` failed ({proc.returncode}): {stderr}")
        return out.decode()

    def _redact(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text


def _parse_generation(text: str) -> tuple[str, str, str]:
    """Split the model output into (diff, title, body)."""
    title, body, diff = "Automated change", "", ""
    if "DIFF:" in text:
        head, _, diff = text.partition("DIFF:")
        diff = diff.lstrip("\n")
    else:
        head = text
    for line in head.splitlines():
        if line.startswith("TITLE:"):
            title = line[len("TITLE:"):].strip() or title
        elif line.startswith("BODY:"):
            body = line[len("BODY:"):].strip()
    if not diff.strip():
        raise RuntimeError("model returned no diff")
    if not diff.endswith("\n"):
        diff += "\n"
    return diff, title, body
