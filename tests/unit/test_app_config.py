#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/config.py 单元测试 — AppConfig 方法与预设完整性"""

from core.config import (
    AppConfig, VAD_PRESETS, TRANSLATION_PROMPTS, LLM_PROVIDERS,
    get_content_type_display_name, get_content_type_description,
)
from core.models import (
    ContentType, ProviderConfig, WhisperConfig, PromptTemplate,
)


# ============================================================================
# VAD_PRESETS 完整性
# ============================================================================

class TestVADPresets:
    def test_all_content_types_covered(self):
        for ct in ContentType:
            assert ct in VAD_PRESETS, f"VAD_PRESETS 缺少 {ct}"

    def test_preset_values_valid(self):
        for ct, vad in VAD_PRESETS.items():
            assert 0 < vad.threshold <= 1.0
            assert vad.min_speech_duration_ms > 0
            assert vad.min_silence_duration_ms > 0
            assert vad.speech_pad_ms >= 0


# ============================================================================
# TRANSLATION_PROMPTS 完整性
# ============================================================================

class TestTranslationPrompts:
    def test_all_content_types_covered(self):
        for ct in ContentType:
            assert ct in TRANSLATION_PROMPTS, f"TRANSLATION_PROMPTS 缺少 {ct}"

    def test_prompts_have_content(self):
        for ct, tpl in TRANSLATION_PROMPTS.items():
            assert tpl.role, f"{ct} role 为空"
            assert tpl.rules, f"{ct} rules 为空"
            assert tpl.style_guide, f"{ct} style_guide 为空"


# ============================================================================
# LLM_PROVIDERS 完整性
# ============================================================================

class TestLLMProviders:
    def test_has_expected_providers(self):
        expected = ["Ollama (本地模型)", "DeepSeek (深度求索)", "Google Gemini", "OpenAI (官方)"]
        for name in expected:
            assert name in LLM_PROVIDERS

    def test_provider_configs_have_base_url(self):
        for name, cfg in LLM_PROVIDERS.items():
            if name != "自定义 (Custom)":
                assert cfg.get('base_url'), f"{name} 缺少 base_url"
                assert cfg.get('model'), f"{name} 缺少 model"


# ============================================================================
# AppConfig.get_vad_parameters
# ============================================================================

class TestAppConfigGetVAD:
    def test_returns_preset_for_content_type(self):
        config = AppConfig(content_type=ContentType.MOVIE)
        vad = config.get_vad_parameters()
        assert vad == VAD_PRESETS[ContentType.MOVIE]

    def test_different_content_types_return_different_presets(self):
        movie = AppConfig(content_type=ContentType.MOVIE).get_vad_parameters()
        music = AppConfig(content_type=ContentType.MUSIC).get_vad_parameters()
        assert movie.threshold != music.threshold


# ============================================================================
# AppConfig.get_prompt_template
# ============================================================================

class TestAppConfigGetPromptTemplate:
    def test_default_template(self):
        config = AppConfig(content_type=ContentType.MOVIE)
        tpl = config.get_prompt_template()
        assert tpl.role == TRANSLATION_PROMPTS[ContentType.MOVIE].role

    def test_user_custom_template_overrides_default(self):
        custom = PromptTemplate(role="custom role", rules="custom rules", style_guide="custom style")
        config = AppConfig(
            content_type=ContentType.MOVIE,
            prompt_templates={ContentType.MOVIE: custom}
        )
        tpl = config.get_prompt_template(ContentType.MOVIE)
        assert tpl.role == "custom role"

    def test_explicit_content_type(self):
        config = AppConfig(content_type=ContentType.MOVIE)
        tpl = config.get_prompt_template(ContentType.ANIMATION)
        assert tpl.role == TRANSLATION_PROMPTS[ContentType.ANIMATION].role

    def test_unknown_content_type_falls_back_to_custom(self):
        config = AppConfig()
        tpl = config.get_prompt_template(ContentType.CUSTOM)
        assert tpl.role == TRANSLATION_PROMPTS[ContentType.CUSTOM].role


# ============================================================================
# AppConfig.get_current_provider_config
# ============================================================================

class TestAppConfigGetProviderConfig:
    def test_returns_saved_config(self):
        saved = ProviderConfig(api_key="sk-123", base_url="https://api.test.com", model_name="test")
        config = AppConfig(
            current_provider="Custom",
            provider_configs={"Custom": saved}
        )
        cfg = config.get_current_provider_config()
        assert cfg.api_key == "sk-123"

    def test_returns_default_when_not_saved(self):
        config = AppConfig(current_provider="Ollama (本地模型)")
        cfg = config.get_current_provider_config()
        assert cfg.base_url == LLM_PROVIDERS["Ollama (本地模型)"]["base_url"]

    def test_unknown_provider_returns_empty(self):
        config = AppConfig(current_provider="Nonexistent")
        cfg = config.get_current_provider_config()
        assert cfg.api_key == ''
        assert cfg.base_url == ''


# ============================================================================
# AppConfig.update_provider_config
# ============================================================================

class TestAppConfigUpdateProviderConfig:
    def test_update_creates_config(self):
        config = AppConfig()
        config.update_provider_config("Custom", "sk-key", "https://api.custom.com", "model-1")
        assert config.current_provider == "Custom"
        assert config.provider_configs["Custom"].api_key == "sk-key"


# ============================================================================
# AppConfig serialization round-trip
# ============================================================================

class TestAppConfigSerialization:
    def test_round_trip(self):
        config = AppConfig(
            whisper=WhisperConfig(model_size="large-v3", device="cuda"),
            content_type=ContentType.ANIMATION,
            current_provider="DeepSeek (深度求索)",
        )
        config.update_provider_config("DeepSeek (深度求索)", "sk-deep", "https://api.deepseek.com", "deepseek-chat")
        d = config.to_dict()
        restored = AppConfig.from_dict(d)
        assert restored.whisper.model_size == "large-v3"
        assert restored.whisper.device == "cuda"
        assert restored.content_type == ContentType.ANIMATION
        assert restored.current_provider == "DeepSeek (深度求索)"
        assert restored.provider_configs["DeepSeek (深度求索)"].api_key == "sk-deep"

    def test_round_trip_with_prompt_templates(self):
        custom = PromptTemplate(role="r", rules="ru", style_guide="s")
        config = AppConfig(prompt_templates={ContentType.MOVIE: custom})
        d = config.to_dict()
        restored = AppConfig.from_dict(d)
        assert restored.prompt_templates[ContentType.MOVIE].role == "r"

    def test_from_dict_invalid_content_type_defaults_to_movie(self):
        data = {'content_type': 'invalid_type'}
        config = AppConfig.from_dict(data)
        assert config.content_type == ContentType.MOVIE

    def test_auto_scan_defaults(self):
        config = AppConfig()
        assert config.auto_scan_enabled is False
        assert config.auto_scan_interval_minutes == 30

    def test_auto_scan_round_trip(self):
        config = AppConfig(auto_scan_enabled=True, auto_scan_interval_minutes=15)
        d = config.to_dict()
        restored = AppConfig.from_dict(d)
        assert restored.auto_scan_enabled is True
        assert restored.auto_scan_interval_minutes == 15

    def test_auto_scan_from_dict_missing_keys_use_defaults(self):
        data = {}
        config = AppConfig.from_dict(data)
        assert config.auto_scan_enabled is False
        assert config.auto_scan_interval_minutes == 30


# ============================================================================
# get_content_type_display_name / description
# ============================================================================

class TestContentTypeHelpers:
    def test_display_name_all_types(self):
        for ct in ContentType:
            name = get_content_type_display_name(ct)
            assert name  # not empty

    def test_description_all_types(self):
        for ct in ContentType:
            desc = get_content_type_description(ct)
            assert desc  # not empty
