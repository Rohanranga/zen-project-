"""Provider wiring tests — OmniRoute (and LLM factory) config is validated
without network I/O.

The OmniRoute gateway is an external service (it may be down, or require a
real key), so these tests only assert configuration parsing and factory
wiring: a valid config must construct cleanly, and missing credentials must
raise at construction time rather than fail mid-request.
"""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def apply_env(monkeypatch):
    """Rebind ``app.config.settings`` (and ``app.llm.settings``) from env vars.

    Modules do ``from app.config import settings``, which keeps its own
    reference to the old object — so we reassign the module attribute to the
    freshly-built ``Settings`` after setting the env vars.
    """

    def _apply(**env) -> Settings:
        defaults = {
            "LLM_PROVIDER": "extractive",
            "OMNIROUTE_API_KEY": "",
            "OMNIROUTE_MODEL": "",
            "OMNIROUTE_BASE_URL": "http://localhost:20128/v1",
        }
        defaults.update(env)
        for key, value in defaults.items():
            monkeypatch.setenv(key, value)

        import app.config

        app.config.settings = Settings()

        import app.llm

        app.llm.settings = app.config.settings
        return app.config.settings

    return _apply


class TestOmniRouteCredentials:
    def test_missing_key_raises(self, apply_env) -> None:
        apply_env(LLM_PROVIDER="omniroute", OMNIROUTE_MODEL="auto/best-coding", OMNIROUTE_API_KEY="")
        import app.llm

        with pytest.raises(ValueError, match="OMNIROUTE_API_KEY"):
            app.llm.OmniRouteLLM()

    def test_missing_model_raises(self, apply_env) -> None:
        apply_env(LLM_PROVIDER="omniroute", OMNIROUTE_API_KEY="sk-test", OMNIROUTE_MODEL="")
        import app.llm

        with pytest.raises(ValueError, match="OMNIROUTE_MODEL"):
            app.llm.OmniRouteLLM()

    def test_valid_config_constructs(self, apply_env) -> None:
        apply_env(
            LLM_PROVIDER="omniroute",
            OMNIROUTE_API_KEY="sk-test",
            OMNIROUTE_MODEL="auto/best-coding",
            OMNIROUTE_BASE_URL="http://localhost:20128/v1",
        )
        import app.llm

        provider = app.llm.OmniRouteLLM()
        assert provider.provider_name == "omniroute:auto/best-coding"


class TestLLMFactory:
    def test_omniroute_config_returns_omniroute(self, apply_env) -> None:
        apply_env(
            LLM_PROVIDER="omniroute",
            OMNIROUTE_API_KEY="sk-test",
            OMNIROUTE_MODEL="auto/best-coding",
        )
        import app.llm

        assert isinstance(app.llm.get_llm(), app.llm.OmniRouteLLM)

    def test_default_returns_extractive(self, apply_env) -> None:
        apply_env(LLM_PROVIDER="extractive")
        import app.llm

        assert isinstance(app.llm.get_llm(), app.llm.ExtractiveLLM)