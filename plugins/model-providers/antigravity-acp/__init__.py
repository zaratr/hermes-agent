"""Antigravity ACP provider profile.

antigravity-acp uses an external ACP subprocess (`agy-acp-windows-x64.exe`)
handled separately in run_agent.py.
"""

from providers import register_provider
from providers.base import ProviderProfile


class AntigravityACPProfile(ProviderProfile):
    """Antigravity ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the ACP subprocess."""
        return ["gemini-3.6-flash-high", "gemini-3.1-pro-high", "antigravity-acp"]


antigravity_acp = AntigravityACPProfile(
    name="antigravity-acp",
    aliases=("antigravity-acp-agent", "agy-acp", "antigravity-acp-model"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://antigravity",
    auth_type="external_process",
)

register_provider(antigravity_acp)
