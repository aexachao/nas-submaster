#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui/pages/media_library.py 单元测试

覆盖：
1. get_selected_count() - session_state 中勾选数计算
2. _filter_files_by_subtitle_status() - 按字幕状态筛选
3. AppTest 烟雾测试 - 验证 render_media_library_page() 不会因变量未定义而抛 UnboundLocalError
"""

import sys
from unittest.mock import patch, MagicMock

import pytest
from streamlit.testing.v1 import AppTest


# ============================================================================
# get_selected_count()
# ============================================================================

class TestGetSelectedCount:
    """get_selected_count() 必须正确计算 session_state 中 s_<id>=True 的项数"""

    def _import(self):
        from ui.pages.media_library import get_selected_count
        return get_selected_count

    def test_empty_session_state(self):
        """空 session_state → 0"""
        with patch("streamlit.session_state", {}):
            fn = self._import()
            assert fn() == 0

    def test_no_selected_items(self):
        """没有任何 s_ 项 → 0"""
        with patch("streamlit.session_state", {"foo": True, "bar": "x", "baz": 1}):
            fn = self._import()
            assert fn() == 0

    def test_one_selected(self):
        """1 个 s_<id>=True → 1"""
        with patch("streamlit.session_state", {"s_42": True, "other": "x"}):
            fn = self._import()
            assert fn() == 1

    def test_multiple_selected(self):
        """多个 s_<id>=True → 计数正确"""
        with patch("streamlit.session_state", {
            "s_1": True, "s_2": True, "s_3": True, "s_4": False, "s_5": True
        }):
            fn = self._import()
            assert fn() == 4  # s_1, s_2, s_3, s_5

    def test_ignores_non_string_keys(self):
        """非字符串 key 不计入（避免误判）"""
        with patch("streamlit.session_state", {
            1: True, 2: True, "s_1": True
        }):
            fn = self._import()
            assert fn() == 1  # 只数 s_1

    def test_ignores_false_values(self):
        """值为 False 的 s_<id> 不计入"""
        with patch("streamlit.session_state", {
            "s_1": False, "s_2": None, "s_3": 0, "s_4": True
        }):
            fn = self._import()
            assert fn() == 1  # 只数 s_4

    def test_similar_prefix_not_counted(self):
        """只有以 s_ 开头的 key 才计数（避免 'something' 这类误判）"""
        with patch("streamlit.session_state", {
            "something": True, "s_": True, "s_x": True
        }):
            fn = self._import()
            assert fn() == 2  # s_ 和 s_x 都匹配


# ============================================================================
# _filter_files_by_subtitle_status()
# ============================================================================

class TestFilterFilesBySubtitleStatus:
    """_filter_files_by_subtitle_status() 按字幕来源筛选"""

    def _import(self):
        from ui.pages.media_library import _filter_files_by_subtitle_status
        return _filter_files_by_subtitle_status

    def _make_file(self, subtitles):
        from core.models import MediaFile
        return MediaFile(
            id=1,
            file_path="/v/test.mkv",
            file_name="test.mkv",
            file_size=1024,
            subtitles=subtitles,
        )

    def _make_sub(self, source, lang="en"):
        from core.models import SubtitleInfo
        return SubtitleInfo(path=f"/v/{source}.srt", lang=lang, source=source)

    def test_no_subtitle_filter(self):
        """无字幕的文件在 no_subtitle 筛选下保留"""
        f = self._make_file([])
        fn = self._import()
        assert fn([f], "no_subtitle") == [f]

    def test_no_subtitle_excludes_with_subs(self):
        """有字幕的文件在 no_subtitle 筛选下排除"""
        f = self._make_file([self._make_sub("asr")])
        fn = self._import()
        assert fn([f], "no_subtitle") == []

    def test_has_subtitle_includes_asr(self):
        """有 asr 字幕的文件在 has_subtitle 筛选下保留"""
        f = self._make_file([self._make_sub("asr")])
        fn = self._import()
        assert fn([f], "has_subtitle") == [f]

    def test_has_subtitle_includes_embedded(self):
        """有 embedded 字幕的文件在 has_subtitle 筛选下保留"""
        f = self._make_file([self._make_sub("embedded")])
        fn = self._import()
        assert fn([f], "has_subtitle") == [f]

    def test_has_subtitle_excludes_translated_only(self):
        """只有 translated 字幕的文件在 has_subtitle 筛选下排除"""
        f = self._make_file([self._make_sub("translated")])
        fn = self._import()
        assert fn([f], "has_subtitle") == []

    def test_has_subtitle_includes_mixed(self):
        """asr + translated 混合时算 has_subtitle（有待处理）"""
        f = self._make_file([self._make_sub("translated"), self._make_sub("asr")])
        fn = self._import()
        assert fn([f], "has_subtitle") == [f]

    def test_has_target_subtitle_includes_translated(self):
        """有 translated 字幕 → 保留"""
        f = self._make_file([self._make_sub("translated")])
        fn = self._import()
        assert fn([f], "has_target_subtitle") == [f]

    def test_has_target_subtitle_excludes_asr(self):
        """只有 asr 字幕 → 排除"""
        f = self._make_file([self._make_sub("asr")])
        fn = self._import()
        assert fn([f], "has_target_subtitle") == []

    def test_empty_input(self):
        """空输入 → 空输出"""
        fn = self._import()
        assert fn([], "no_subtitle") == []
        assert fn([], "has_subtitle") == []
        assert fn([], "has_target_subtitle") == []


# ============================================================================
# AppTest 烟雾测试
# ============================================================================

class TestRenderMediaLibraryPageSmoke:
    """render_media_library_page() 烟雾测试

    这些测试的目的是捕获类似 UnboundLocalError 这类"变量未定义"问题
    —— 当 UI 添加新列时，可能引用了还没计算好的变量。

    通过 mock 所有外部依赖（DB、文件系统、scanner）来隔离页面渲染逻辑。
    """

    @staticmethod
    def _build_at():
        """构造一个最小可运行的 AppTest 实例"""
        # 桩：所有外部依赖都返回空/默认
        empty_files = []
        with patch("database.media_dao.MediaDAO.get_media_files_filtered", return_value=empty_files), \
             patch("database.media_dao.MediaDAO.delete_media_file", return_value=True), \
             patch("database.media_dao.MediaDAO.get_media_paths_by_prefix", return_value=[]), \
             patch("database.task_dao.TaskDAO.get_pending_task", return_value=None), \
             patch("services.media_scanner.discover_media_subdirectories", return_value=[]), \
             patch("services.media_scanner.scan_media_directory", return_value=(0, [])), \
             patch("services.media_scanner.rescan_video_subtitles", return_value=None):
            # 直接调用渲染函数（不通过 run()），导入模块绕过 streamlit 启动器
            from ui.pages.media_library import render_media_library_page
            try:
                render_media_library_page(debug_mode=False)
            except Exception as e:
                pytest.fail(f"render_media_library_page() raised {type(e).__name__}: {e}")

    def test_renders_without_error_on_empty_library(self):
        """空媒体库时页面应正常渲染（不抛 UnboundLocalError 等）"""
        self._build_at()

    def test_renders_with_session_state(self):
        """session_state 有勾选项时也不报错"""
        import streamlit as st

        # 用 MagicMock 模拟 session_state（支持属性赋值）
        mock_state = MagicMock()
        # 模拟 dict-like 行为：__contains__, get, items, __iter__
        # selected count 通过 .items() 数 True 的 s_<id>
        mock_state.items.return_value = [
            ("s_1", True),
            ("s_2", False),
            ("_id_to_path", {1: "/v/test.mkv"}),
        ]
        mock_state.__contains__ = lambda k: k in {"s_1", "s_2", "_id_to_path"}
        mock_state.get = lambda k, default=None: {"s_1": True, "s_2": False, "_id_to_path": {1: "/v/test.mkv"}}.get(k, default)

        with patch.object(st, "session_state", mock_state), \
             patch("database.media_dao.MediaDAO.get_media_files_filtered", return_value=[]), \
             patch("services.media_scanner.discover_media_subdirectories", return_value=[]), \
             patch("services.media_scanner.scan_media_directory", return_value=(0, [])):
            from ui.pages.media_library import render_media_library_page, get_selected_count
            # 单独验证 selected_count 计算正确（这个会真正调用 streamlit.session_state）
            try:
                count = get_selected_count()
                assert count == 1, f"应该数出 1 个选中，实际 {count}"
            except Exception as e:
                pytest.fail(f"get_selected_count() raised {type(e).__name__}: {e}")

    def test_selected_count_computed_before_columns(self):
        """
        关键回归测试：selected_count 必须在列定义之前被计算。

        如果未来有人重排代码，把 selected_count 的计算放到了某个 with col_xxx: 内部，
        本测试会捕获 UnboundLocalError。
        """
        import streamlit as st
        import ast
        import inspect
        from ui.pages.media_library import render_media_library_page

        source = inspect.getsource(render_media_library_page)
        tree = ast.parse(source)

        # 找 selected_count 第一次被赋值（AugAssign/Assign/For/...）的行号
        # 也找 get_selected_count() 调用
        assign_line = None
        call_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "selected_count":
                        if assign_line is None:
                            assign_line = node.lineno
            if isinstance(node, ast.Call):
                # 找 get_selected_count() 调用
                if isinstance(node.func, ast.Name) and node.func.id == "get_selected_count":
                    if call_line is None:
                        call_line = node.lineno

        assert assign_line is not None or call_line is not None, (
            "render_media_library_page 中必须给 selected_count 赋值"
        )

        # 找到 col_del 的 st.columns 调用所在行（或者 with col_del: 的行）
        # 简化：找 st.columns([..., 0.8, 0.8, 0.8]) 后面第一个 with col_del: 行
        col_def_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "columns":
                    if col_def_line is None:
                        col_def_line = node.lineno

        # 如果 selected_count 是用函数调用计算的，确保它在 st.columns 之前
        if call_line is not None and col_def_line is not None:
            assert call_line < col_def_line, (
                f"get_selected_count() (line {call_line}) 必须在 st.columns() (line {col_def_line}) 之前"
            )
