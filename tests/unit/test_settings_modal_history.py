#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/settings_modal.py 静态 + 烟雾测试

覆盖：
- "历史版本" UI 区块已删除（防止误恢复）
- 相关 import 也清理（get_all_releases / ReleaseInfo）
- 模块仍可正常 import
- 后续加新功能时不要悄悄回滚
"""

import ast
import sys
import pytest


SETTINGS_MODAL_PATH = "ui/settings_modal.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source_text() -> str:
    with open(SETTINGS_MODAL_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def source_tree(source_text: str):
    return ast.parse(source_text)


@pytest.fixture
def updater_imports(source_tree: ast.Module):
    """提取 from services.updater import ... 的名字列表"""
    for node in ast.walk(source_tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "services.updater"
        ):
            return [n.name for n in node.names]
    return []


# ---------------------------------------------------------------------------
# "历史版本" UI 已删除
# ---------------------------------------------------------------------------

class TestHistoryBlockRemoved:
    """"历史版本" UI 区块必须不存在"""

    def test_no_history_subheader(self, source_text: str):
        """不该再渲染 '历史版本' subheader"""
        assert "st.subheader(\"历史版本\")" not in source_text
        assert "st.subheader('历史版本')" not in source_text

    def test_no_history_info_fallback(self, source_text: str):
        """不该再有 '无法获取版本历史' 占位文案"""
        assert "无法获取版本历史" not in source_text

    def test_no_get_all_releases_call(self, source_text: str):
        """不该再调用 get_all_releases()"""
        assert "get_all_releases(" not in source_text

    def test_no_releases_loop(self, source_text: str):
        """不该再有 for r in releases 循环"""
        assert "for r in releases" not in source_text


# ---------------------------------------------------------------------------
# 配套 import 已清理
# ---------------------------------------------------------------------------

class TestUselessImportsRemoved:
    """get_all_releases / ReleaseInfo 不该再 import"""

    def test_get_all_releases_not_imported(self, updater_imports: list):
        assert "get_all_releases" not in updater_imports

    def test_release_info_not_imported(self, updater_imports: list):
        assert "ReleaseInfo" not in updater_imports


# ---------------------------------------------------------------------------
# 项目地址链接必须保留（不能误删）
# ---------------------------------------------------------------------------

class TestGitHubLinkPreserved:
    """相邻的 GitHub 链接不能因为重构被误删"""

    def test_github_link_present(self, source_text: str):
        assert "github.com/aexachao/nas-submaster" in source_text


# ---------------------------------------------------------------------------
# 模块仍能正常 import
# ---------------------------------------------------------------------------

class TestModuleImportable:
    """ui.settings_modal 必须可导入（语法、依赖都 OK）

    需要 streamlit 才能 import,本地没装时优雅跳过。
    容器/CI 环境会执行。
    """

    def test_settings_modal_imports(self):
        pytest.importorskip("streamlit", reason="streamlit not in local env")
        try:
            import ui.settings_modal  # noqa: F401
        except Exception as e:
            pytest.fail(f"ui.settings_modal import failed: {e}")
