from __future__ import annotations

from local_whisper_transcriber.models import MODEL_PROFILES, MODEL_SIZES, format_model_profile


def test_model_profiles_cover_all_gui_models() -> None:
    assert set(MODEL_PROFILES) == set(MODEL_SIZES)

    for model_size in MODEL_SIZES:
        profile = MODEL_PROFILES[model_size]
        assert profile.speed
        assert profile.accuracy
        assert profile.resource
        assert profile.recommended_for
        assert profile.note
        assert model_size in format_model_profile(model_size)


def test_large_v3_and_tiny_have_clear_tradeoffs() -> None:
    assert MODEL_PROFILES["large-v3"].accuracy == "最高"
    assert MODEL_PROFILES["tiny"].speed == "最快"
    assert "正式" in MODEL_PROFILES["large-v3"].recommended_for
    assert "测试" in MODEL_PROFILES["tiny"].recommended_for
