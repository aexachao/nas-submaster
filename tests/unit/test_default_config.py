#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for v1.8.2 default config values.

- TranslationConfig.enabled 默认 True (v1.8.2 改)
- WhisperConfig.model_size 默认 'medium' (v1.8.2 改)

同时覆盖 ConfigManager.load() 的字符串默认值（'true' / 'medium'），
确保 dataclass 默认和 DB 字符串默认保持一致。

⚠️ 重要：升级到 v1.8.2 之前已经存了 enable_translation=false 的用户
不应受新默认值影响 —— DB 里有值就用 DB 的，没值才用新默认。
这个测试也覆盖"老 DB 行为保持不变"。
"""

import sqlite3
import pytest
from unittest.mock import patch

from core.models import WhisperConfig, TranslationConfig
from core.config import AppConfig


# ============================================================================
# dataclass 默认值
# ============================================================================

class TestWhisperConfigDefault:
    def test_default_model_size_is_medium(self):
        """v1.8.2+: 默认 Whisper 模型 = medium（不再是 base）"""
        cfg = WhisperConfig()
        assert cfg.model_size == "medium", (
            f"WhisperConfig.model_size 默认应该是 'medium'，"
            f"实际: {cfg.model_size!r}"
        )


class TestTranslationConfigDefault:
    def test_default_enabled_is_true(self):
        """v1.8.2+: 默认开启翻译（不再是 False）"""
        cfg = TranslationConfig()
        assert cfg.enabled is True, (
            f"TranslationConfig.enabled 默认应该是 True，"
            f"实际: {cfg.enabled!r}"
        )


class TestAppConfigDefault:
    def test_appconfig_default_includes_new_defaults(self):
        """AppConfig 默认实例的翻译和 Whisper 都用新默认"""
        cfg = AppConfig()
        assert cfg.translation.enabled is True
        assert cfg.whisper.model_size == "medium"


# ============================================================================
# ConfigManager.load() 字符串默认值（DB 里没值时的回退）
# ============================================================================

@pytest.fixture
def empty_db(tmp_path):
    """空 config DB（没有任何 key）"""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    return _new_conn, db_path


class TestConfigManagerLoadDefaults:
    def test_empty_db_uses_medium_default(self, empty_db):
        """空 DB（全新装）应返回 medium 模型"""
        new_conn, _ = empty_db
        from core.config import ConfigManager
        mgr = ConfigManager(new_conn)
        cfg = mgr.load()
        assert cfg.whisper.model_size == "medium"

    def test_empty_db_uses_translation_enabled_default(self, empty_db):
        """空 DB（全新装）应默认开启翻译"""
        new_conn, _ = empty_db
        from core.config import ConfigManager
        mgr = ConfigManager(new_conn)
        cfg = mgr.load()
        assert cfg.translation.enabled is True


# ============================================================================
# 升级兼容性：老 DB 里存的值不应被新默认值覆盖
# ============================================================================

@pytest.fixture
def legacy_db_with_disabled_translation(tmp_path):
    """模拟 v1.8.1 用户的 DB：显式存了 enable_translation=false"""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    # v1.8.1 用户显式关了翻译
    conn.execute(
        "INSERT INTO config VALUES ('enable_translation', 'false')"
    )
    conn.commit()
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    return _new_conn


@pytest.fixture
def legacy_db_with_base_model(tmp_path):
    """模拟 v1.8.1 用户的 DB：显式存了 whisper_model=base"""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO config VALUES ('whisper_model', 'base')"
    )
    conn.commit()
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    return _new_conn


class TestUpgradeCompatibility:
    def test_legacy_disabled_translation_preserved(self, legacy_db_with_disabled_translation):
        """
        升级 v1.8.2 后，老用户显式关掉的翻译不应被新默认 True 覆盖。

        这是 v1.8.2 的关键安全保证 —— 否则升级瞬间所有老用户翻译会自动开启。
        """
        new_conn = legacy_db_with_disabled_translation
        from core.config import ConfigManager
        mgr = ConfigManager(new_conn)
        cfg = mgr.load()
        # DB 里是 'false'，必须读出来还是 False
        assert cfg.translation.enabled is False, (
            "v1.8.2 默认值不应覆盖 DB 里的 enable_translation=false"
        )

    def test_legacy_base_model_preserved(self, legacy_db_with_base_model):
        """
        升级 v1.8.2 后，老用户选 base 模型不应被新默认 medium 覆盖。
        否则所有老用户会自动下载 1.5GB medium。
        """
        new_conn = legacy_db_with_base_model
        from core.config import ConfigManager
        mgr = ConfigManager(new_conn)
        cfg = mgr.load()
        assert cfg.whisper.model_size == "base", (
            "v1.8.2 默认值不应覆盖 DB 里的 whisper_model=base"
        )


# ============================================================================
# from_dict / to_dict 序列化
# ============================================================================

class TestAppConfigSerialization:
    def test_from_dict_uses_new_defaults_for_missing_keys(self):
        """from_dict 缺字段时使用 dataclass 默认值（True / medium）"""
        cfg = AppConfig.from_dict({})  # 完全空
        assert cfg.translation.enabled is True
        assert cfg.whisper.model_size == "medium"

    def test_to_dict_roundtrip_preserves_values(self):
        """to_dict → from_dict 来回不丢值"""
        original = AppConfig()
        serialized = original.to_dict()
        restored = AppConfig.from_dict(serialized)
        assert restored.translation.enabled == original.translation.enabled
        assert restored.whisper.model_size == original.whisper.model_size


# ============================================================================
# 守卫测试：未来不能再改回 False / base（除非有意识的 v2.x）
# ============================================================================

class TestDefaultsAreDeliberate:
    """防止有人"好心"把默认值改回 False / base"""

    def test_translation_default_is_true_not_false(self):
        # 反向断言：明确禁止改回 False
        cfg = TranslationConfig()
        assert cfg.enabled is True, (
            "🚨 TranslationConfig.enabled 默认被改回 False！"
            "v1.8.2+ 应该有意识地保持 True，"
            "如果确实要改回 False，请先讨论用户影响（升级用户行为变化）"
        )

    def test_whisper_default_is_medium_not_base(self):
        cfg = WhisperConfig()
        assert cfg.model_size == "medium", (
            "🚨 WhisperConfig.model_size 默认被改回 base！"
            "v1.8.2+ 应该有意识地保持 medium（1.5GB），"
            "如果确实要改回 base，请先讨论（NAS 用户会突然少下载 1.4GB 模型）"
        )
