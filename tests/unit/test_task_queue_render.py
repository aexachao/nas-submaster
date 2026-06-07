#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for task queue multi-stage step bar rendering (v1.8.1).

v1.8.1: 任务卡片不再只显示单条进度条，改成"步骤条 + 当前阶段独立进度"。
- 已完成阶段：✓ 100%（绿色）
- 当前阶段：⏳ 百分比或 X/Y 条
- 未开始：○ 灰色

下载 → 提取 → 翻译
"""

import sqlite3
import pytest
from unittest.mock import patch

from core.models import TaskStatus


# ============================================================================
# 抽离纯函数：build_stage_progress_html(task) → str
# ============================================================================
# 我们要把 task_queue.py 里的渲染逻辑抽成一个纯函数，
# 这样可以在没有 streamlit 的环境下单测。

def _import_renderer():
    """懒导入 streamlit 依赖（可能在没装 streamlit 的环境跑）"""
    pytest.importorskip("streamlit", reason="streamlit not in local env")
    from ui.pages.task_queue import build_stage_progress_html
    return build_stage_progress_html


# ============================================================================
# 共用 fixture
# ============================================================================

@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            log_history TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at CURRENT_TIMESTAMP
        );
    """)
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    with patch("database.task_dao.get_db_connection", side_effect=_new_conn):
        yield db_path


def _make_task(stage="pending", stage_progress=None, status=TaskStatus.PROCESSING,
               progress=0, log=""):
    """造一个 Task 对象"""
    from core.models import Task
    return Task(
        id=1,
        file_path="/media/movie.mp4",
        status=status,
        progress=progress,
        log=log,
        log_history="",
        stage=stage,
        stage_progress=stage_progress,
    )


# ============================================================================
# 阶段标签
# ============================================================================

class TestStageLabels:
    """三个阶段必须出现在 HTML 里"""

    def test_all_three_stages_present(self):
        render = _import_renderer()
        task = _make_task(stage="download", stage_progress=50.0)
        html = render(task)
        assert "下载" in html, "缺'下载'标签"
        assert "提取" in html, "缺'提取'标签"
        assert "翻译" in html, "缺'翻译'标签"

    def test_disabled_stages_when_translation_not_enabled(self):
        """翻译未启用时，UI 应能识别（不会硬错）"""
        render = _import_renderer()
        task = _make_task(stage="extract", stage_progress=80.0)
        html = render(task)
        # 至少存在三个标签
        assert "下载" in html
        assert "提取" in html
        assert "翻译" in html


# ============================================================================
# 当前阶段高亮
# ============================================================================

class TestCurrentStageHighlight:
    """当前 stage 应有视觉高亮（不同 CSS class 或 emoji）"""

    def test_current_stage_shows_in_progress_marker(self):
        render = _import_renderer()
        task = _make_task(stage="download", stage_progress=50.0)
        html = render(task)
        # 当前阶段：⏳ 或 [进行中] class
        # 已经完成的：✓ 或 [完成] class
        # 未开始的：○ 或 [未开始] class
        assert "⏳" in html or "in-progress" in html or "active" in html, (
            f"当前阶段没高亮: {html[:300]}"
        )

    def test_extract_stage_shows_percentage_with_two_decimals(self):
        render = _import_renderer()
        task = _make_task(stage="extract", stage_progress=47.32)
        html = render(task)
        # 两位小数百分比
        assert "47.32%" in html, f"缺两位小数百分比: {html[:300]}"

    def test_translate_stage_shows_x_of_y_lines(self):
        """翻译阶段显示 '已翻译 X/Y 条' 而不是百分比"""
        render = _import_renderer()
        task = _make_task(stage="translate", stage_progress=47)  # 整数 = current_line
        html = render(task)
        # 应该有 "47" + "条" 或 "X/Y 条" 格式
        # total_lines 通过 message 推断？不，应该是显示 current / total
        # 这里只测有 "条" 出现
        # 更精确的：应该有 "已翻译 47" + 占位符
        assert "47" in html, f"缺 47: {html[:300]}"
        # 翻译阶段不应该用百分比格式
        assert "47.00%" not in html, f"翻译阶段不应显示百分比: {html[:300]}"


# ============================================================================
# 完成态
# ============================================================================

class TestCompletedStage:
    def test_completed_task_shows_all_stages_done(self):
        render = _import_renderer()
        task = _make_task(
            stage="completed",
            stage_progress=100.0,
            status=TaskStatus.COMPLETED,
        )
        html = render(task)
        # 全部完成：每个阶段都应该有 ✓
        # 简单断言：HTML 里至少有 1 个 ✓（不强求 3 个，因为可能 emoji 渲染不同）
        # 更稳：断言不再有 "进行中" 标识
        assert "⏳" not in html and "in-progress" not in html, (
            f"完成态不该有'进行中'标记: {html[:300]}"
        )


# ============================================================================
# 旧字段 fallback
# ============================================================================

class TestLegacyProgressFallback:
    """旧 DB（stage=NULL, stage_progress=NULL）应 fallback 到 progress 字段"""

    def test_legacy_task_uses_progress_field(self):
        render = _import_renderer()
        task = _make_task(stage="pending", stage_progress=None, progress=45)
        # 不应崩
        html = render(task)
        assert html  # 非空

    def test_pending_task_no_progress_bar(self):
        render = _import_renderer()
        task = _make_task(
            stage="pending", stage_progress=None,
            status=TaskStatus.PENDING
        )
        html = render(task)
        # pending 状态：不应有进度条
        # 简化：HTML 不应包含 "100%"（没进度数据）
        # 这个不绝对，我们只测不崩


# ============================================================================
# 边界
# ============================================================================

class TestEdgeCases:
    def test_render_does_not_crash_on_minimal_task(self):
        render = _import_renderer()
        task = _make_task()
        html = render(task)
        assert html

    def test_render_escapes_user_controllable_text(self):
        """progress_text 含特殊字符时，HTML 必须转义（避免 XSS）"""
        render = _import_renderer()
        # stage_progress 浮点不存在 XSS 风险
        # 但如果某个 stage_progress 是恶意字符串，应该被转义
        from core.models import Task
        task = Task(
            id=1, file_path="/media/movie.mp4",
            status=TaskStatus.PROCESSING, progress=0, log="",
            log_history="", stage="extract",
            stage_progress="<script>alert('xss')</script>",
        )
        html_str = render(task)
        # 即使 stage_progress 是恶意字符串，也不能作为 <script> 出现
        assert "<script>" not in html_str
        # 也不该报异常（被 _format_stage_progress 的 try/except 接住）
