from utils.codex_interface import _active_model_provider, _retry_config_args


def test_active_provider_from_config():
    text = 'model = "x"\nmodel_provider = "hr_api"\n\n[model_providers.hr_api]\nbase_url = "http://x"\n'
    assert _active_model_provider(text) == "hr_api"


def test_active_provider_defaults_to_openai():
    assert _active_model_provider("") == "openai"
    assert _active_model_provider("[model_providers.foo]\nname = \"foo\"\n") == "openai"


def test_retry_config_args_target_named_provider(monkeypatch):
    monkeypatch.setattr("utils.codex_interface.IDLE_TIMEOUT_MS", 120000)
    monkeypatch.setattr("utils.codex_interface.REQUEST_RETRIES", 4)
    monkeypatch.setattr("utils.codex_interface.STREAM_RETRIES", 5)
    assert _retry_config_args("hr_api") == [
        "-c", "model_providers.hr_api.stream_idle_timeout_ms=120000",
        "-c", "model_providers.hr_api.request_max_retries=4",
        "-c", "model_providers.hr_api.stream_max_retries=5",
    ]


def test_retry_config_args_disabled_when_idle_and_retries_zero(monkeypatch):
    monkeypatch.setattr("utils.codex_interface.IDLE_TIMEOUT_MS", 0)
    monkeypatch.setattr("utils.codex_interface.REQUEST_RETRIES", 0)
    monkeypatch.setattr("utils.codex_interface.STREAM_RETRIES", 0)
    assert _retry_config_args("hr_api") == []
