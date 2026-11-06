import json

import pytest

from sciworld import settings


def test_defaults_when_nothing_is_saved(isolated):
    assert not settings.is_initialized()
    cfg = settings.load_config()
    assert cfg == settings.DEFAULT_CONFIG
    assert settings.config_dir() == isolated / "sciworld_home"


def test_config_dir_defaults_to_home(monkeypatch, isolated):
    monkeypatch.delenv("SCIWORLD_HOME")
    assert settings.config_dir() == isolated / "home" / ".sciworld"


def test_save_and_load_round_trip():
    cfg = dict(settings.DEFAULT_CONFIG, data_dir="/data/sw", container="docker")
    settings.save_config(cfg)
    assert settings.is_initialized()
    assert settings.load_config() == cfg
    assert json.loads(settings.config_file().read_text())["container"] == "docker"


def test_missing_keys_are_filled_from_defaults():
    settings.config_dir().mkdir(parents=True)
    settings.config_file().write_text(json.dumps({"data_dir": "/x"}))
    cfg = settings.load_config()
    assert cfg["data_dir"] == "/x" and cfg["org"] == "ScientistWorld" and cfg["bind_paths"] == []


def test_environment_overrides_are_not_written_back(monkeypatch):
    settings.save_config(dict(settings.DEFAULT_CONFIG, data_dir="/saved", org="Saved"))
    monkeypatch.setenv("SCIWORLD_DATA_DIR", "/from/env")
    monkeypatch.setenv("SCIWORLD_ORG", "EnvOrg")
    assert settings.load_config()["data_dir"] == "/from/env"
    assert settings.load_config()["org"] == "EnvOrg"
    stored = settings.load_config(apply_env=False)
    assert stored["data_dir"] == "/saved" and stored["org"] == "Saved"


def test_data_dir_tilde_is_expanded(isolated):
    settings.save_config(dict(settings.DEFAULT_CONFIG, data_dir="~/sw"))
    assert settings.load_config()["data_dir"] == str(isolated / "home" / "sw")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_unreadable_config_is_reported(content):
    settings.config_dir().mkdir(parents=True)
    settings.config_file().write_text(content)
    with pytest.raises(settings.ConfigError):
        settings.load_config()


def test_containers_dir_follows_data_dir(isolated):
    assert settings.containers_dir({"data_dir": "/d"}).as_posix() == "/d/containers"
    assert settings.containers_dir({"data_dir": ""}) == isolated / "sciworld_home" / "containers"


def test_instructions_round_trip_and_removal():
    assert settings.load_instruction("compute") == ""
    settings.save_instruction("compute", "  use srun  \n")
    assert settings.load_instruction("compute") == "use srun"
    assert settings.list_instructions() == ["compute"]
    settings.remove_instruction("compute")
    settings.remove_instruction("compute")  # removing twice is fine
    assert settings.load_instruction("compute") == ""


def test_overrides_win_over_files_and_can_blank_them():
    settings.save_instruction("system_prompt", "from disk")
    token = settings._instruction_overrides.set({"system_prompt": "from call", "compute": None})
    try:
        assert settings.load_instruction("system_prompt") == "from call"
        assert settings.load_instruction("compute") == ""
    finally:
        settings._instruction_overrides.reset(token)
    assert settings.load_instruction("system_prompt") == "from disk"
