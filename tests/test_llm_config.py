"""
Unit tests for _build_llm_config() across all LLM providers.

These tests monkeypatch the module-level LLM_PROVIDER variable and individual
env vars so _build_llm_config() can be called with different provider configs
without restarting the process or reloading the module.
"""

import pytest

import backend.agent.llm_agent as m


class TestAzureV1:
    def test_model_string(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "v1")

        cfg = m._build_llm_config()

        assert cfg.model == "openai/gpt-4o"

    def test_api_base_includes_openai_v1_path(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "v1")

        cfg = m._build_llm_config()

        assert cfg.api_base == "https://myaccount.openai.azure.com/openai/v1"

    def test_no_api_version_for_v1(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "v1")

        cfg = m._build_llm_config()

        assert cfg.api_version is None

    def test_trailing_slash_stripped_from_endpoint(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "v1")

        cfg = m._build_llm_config()

        assert not cfg.api_base.endswith("//")


class TestAzureLegacy:
    def test_model_uses_azure_prefix(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "legacy")

        cfg = m._build_llm_config()

        assert cfg.model == "azure/gpt-4o"

    def test_api_version_set_for_legacy(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "testkey")
        monkeypatch.setenv("AZURE_OPENAI_API_STYLE", "legacy")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-11-20")

        cfg = m._build_llm_config()

        assert cfg.api_version == "2024-11-20"

    def test_missing_key_raises(self, monkeypatch):
        # When api_key is absent the code falls through to the AWS Secrets Manager
        # path. In CI (no boto3 / no SM creds) it must still raise a RuntimeError —
        # the exact message depends on whether boto3 is installed.
        monkeypatch.setattr(m, "LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FES_AZURE_OPENAI_SECRET_ID", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)

        with pytest.raises(RuntimeError):
            m._build_llm_config()


class TestDatabricks:
    def test_model_string(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "databricks")
        monkeypatch.setenv("DATABRICKS_HOST", "https://myworkspace.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dbrx_token")
        monkeypatch.setenv("LLM_ENDPOINT", "fes-gpt4o")

        cfg = m._build_llm_config()

        assert cfg.model == "databricks/fes-gpt4o"

    def test_api_base_is_host_only(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "databricks")
        monkeypatch.setenv("DATABRICKS_HOST", "https://myworkspace.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dbrx_token")
        monkeypatch.setenv("LLM_ENDPOINT", "fes-gpt4o")

        cfg = m._build_llm_config()

        # LiteLLM constructs the full /serving-endpoints/{endpoint}/invocations path.
        assert cfg.api_base == "https://myworkspace.cloud.databricks.com"
        assert "/serving-endpoints" not in cfg.api_base

    def test_missing_host_raises(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "databricks")
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.setenv("DATABRICKS_TOKEN", "dbrx_token")
        monkeypatch.setenv("LLM_ENDPOINT", "fes-gpt4o")

        with pytest.raises(RuntimeError, match="DATABRICKS_HOST"):
            m._build_llm_config()


class TestHuggingFace:
    def test_model_string(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "huggingface")
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_testkey")
        monkeypatch.setenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")

        cfg = m._build_llm_config()

        assert cfg.model == "huggingface/HuggingFaceH4/zephyr-7b-beta"

    def test_api_base_is_none(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "huggingface")
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_testkey")
        monkeypatch.setenv("HUGGINGFACE_MODEL", "HuggingFaceH4/zephyr-7b-beta")

        cfg = m._build_llm_config()

        assert cfg.api_base is None


class TestUnsupportedProvider:
    def test_raises_for_unknown_provider(self, monkeypatch):
        monkeypatch.setattr(m, "LLM_PROVIDER", "openai")

        with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
            m._build_llm_config()
