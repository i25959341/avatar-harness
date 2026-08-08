import os

from interactive_avatar.environment import load_env_file


def test_load_env_file_parses_values_and_preserves_existing(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "# comment\nAVATAR_HARNESS_NEW='new value'\nAVATAR_HARNESS_EXISTING=replaced\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AVATAR_HARNESS_EXISTING", "original")
    monkeypatch.delenv("AVATAR_HARNESS_NEW", raising=False)

    load_env_file(env_file)

    assert os.environ["AVATAR_HARNESS_NEW"] == "new value"
    assert os.environ["AVATAR_HARNESS_EXISTING"] == "original"


def test_load_env_file_accepts_a_missing_file(tmp_path) -> None:
    load_env_file(tmp_path / "missing.env")
