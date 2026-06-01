"""Import-only / config smoke tests - no network, no DB."""

from agentkit.config import Settings


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.anthropic_model
    assert s.database_url.startswith("postgresql")


def test_app_imports():
    # Importing the API module must not require a live DB or API key.
    from agentkit.api.app import app

    assert app.title == "agentkit"
