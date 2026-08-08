from interactive_avatar.flashhead.config import FlashHeadConfig


def test_flashhead_interruption_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FLASHHEAD_INTERRUPTION_TRANSITION", "VAE")
    monkeypatch.setenv("FLASHHEAD_INTERRUPTION_BRIDGE_FRAMES", "6")

    config = FlashHeadConfig.from_env()

    assert config.interruption_transition == "vae"
    assert config.interruption_bridge_frames == 6
