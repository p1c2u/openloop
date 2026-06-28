"""Runtime configuration loaded from environment / `.env`.

Mirrors the keys documented in `.env.example`. Only what the first vertical
slice needs is wired up here; more lands as the runtime grows.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model providers — LiteLLM reads these from the environment directly, but we
    # surface them here so the runtime can report which providers are configured.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Slack surface
    slack_bot_token: str | None = None
    slack_signing_secret: str | None = None
    slack_app_token: str | None = None

    # GitHub connector
    github_token: str | None = None

    # Coding worker — model the worker uses to generate edits. Matches the
    # `task: code` route in the example agent. Codegen is multi-step and
    # token-heavy; revisit `per_task_usd` for `task: code` accordingly.
    coding_worker_model: str = "anthropic/claude-sonnet-4-6"
    # Enable the real git-backed worker (needs a contents:write token + a
    # sandboxed checkout). Off by default — the connector stays unregistered.
    coding_worker_enabled: bool = False

    # Storage / queue
    database_url: str = (
        "postgresql://openloop:change-me@localhost:5432/openloop_agents"
    )
    redis_url: str = "redis://localhost:6379/0"

    # Cross-process coordination for multi-replica deploys. "memory" (process-local
    # lock, default — correct for a single replica) or "redis" (shared lock so
    # exactly one replica leads startup recovery). Needs `redis_url` + the `redis`
    # extra; falls back to in-process if Redis is unreachable.
    lock_backend: str = "memory"
    # How often (seconds) to re-run the crash-recovery sweep under the lock, the
    # backstop that heals a recovery leader that died mid-sweep. 0 disables the
    # periodic retry (startup-only). Runs once at startup regardless.
    recovery_interval_seconds: int = 300

    # Runtime
    log_level: str = "info"

    # Where agent config-as-code lives
    agents_dir: str = "agents"

    # Memory
    # Backend: "memory" (process-local, default — runs without a DB) or
    # "postgres" (pgvector-backed, persistent).
    memory_backend: str = "memory"
    # Set to false to disable semantic recall (recency-only memory).
    embeddings_enabled: bool = True
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    @property
    def embedding_provider(self) -> str:
        """LiteLLM-style provider prefix of the embedding model."""
        return self.embedding_model.split("/", 1)[0]

    @property
    def configured_providers(self) -> list[str]:
        """Provider prefixes (LiteLLM-style) that have a key set."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.openrouter_api_key:
            providers.append("openrouter")
        return providers


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
