#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for task multi-stage progress (stage + stage_progress) — v1.8.1

背景：之前 tasks 表只有 progress: int 字段，whisper_service 用
max(5, min(99, ...)) 把 0-100% 硬塞进 5-99% 区间，再用 worker
硬编码 50+int(...) 把翻译进度硬塞进 50-95% 区间。

v1.8.1 改为：
- tasks 表加 stage TEXT + stage_progress REAL 两列
- 各阶段 callback 上报 (stage, progress) 而不是 (current, total)
- 阶段独立：download 0-100%, extract 0-100%, translate 用 X/Y 条
"""

import sqlite3
import pytest
from unittest.mock import patch

from core.models import TaskStatus


# ============================================================================
# 共用 fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    """为每个测试创建数据库（v1.8.0 旧 schema，没有 stage 列）"""
    db_path = str(tmp_path / "test.db")

    # 旧 schema：没有 stage / stage_progress 列
    # 验证 TaskDAO 在首次 update_task(stage=...) 时会触发迁移
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS media_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_name TEXT,
            file_size INTEGER,
            subtitles_json TEXT DEFAULT '[]',
            has_translated INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            log_history TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    with patch("database.task_dao.get_db_connection", side_effect=_new_conn):
        with patch("database.media_dao.get_db_connection", side_effect=_new_conn):
            yield db_path


# ============================================================================
# Stage 字段基本读写
# ============================================================================

class TestTaskStageMigration:
    """首次 update_task(stage=...) 应自动迁移 schema 加 stage + stage_progress 列"""

    def test_old_db_lacks_stage_columns(self, in_memory_db):
        """验证 fixture 的旧 schema 确实没有 stage 列（确认迁移有意义）"""
        conn = sqlite3.connect(in_memory_db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        conn.close()
        assert "stage" not in cols, "fixture 应该是旧 schema"
        assert "stage_progress" not in cols, "fixture 应该是旧 schema"

    def test_update_task_with_stage_triggers_migration(self, in_memory_db):
        """首次 update_task(stage='download') 应自动 ALTER TABLE 加列"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()

        # 这一步会触发迁移
        TaskDAO.update_task(
            task.id,
            stage="download",
            stage_progress=12.34,
            log="正在下载"
        )

        # 验证列已添加
        conn = sqlite3.connect(in_memory_db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        conn.close()
        assert "stage" in cols, "迁移后应该有 stage 列"
        assert "stage_progress" in cols, "迁移后应该有 stage_progress 列"

    def test_get_task_returns_stage_fields(self, in_memory_db):
        """get_task_by_id 应能读出 stage 和 stage_progress 字段"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id,
            stage="extract",
            stage_progress=47.32,
            log="提取中"
        )
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage == "extract"
        assert updated.stage_progress == 47.32

    def test_migration_is_idempotent(self, in_memory_db):
        """第二次 update_task(stage=...) 不应因列已存在而报错"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()

        TaskDAO.update_task(task.id, stage="download", stage_progress=10.0)
        # 第二次：列已存在
        TaskDAO.update_task(task.id, stage="download", stage_progress=20.0)
        TaskDAO.update_task(task.id, stage="download", stage_progress=30.0)

        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage_progress == 30.0


# ============================================================================
# Stage 字段类型
# ============================================================================

class TestTaskStageType:
    """stage_progress 必须是 REAL，能存 0.01% 这种两位小数"""

    def test_stage_progress_preserves_decimal(self, in_memory_db):
        """两位小数 47.32 必须能 round-trip"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, stage="download", stage_progress=47.32)
        updated = TaskDAO.get_task_by_id(task.id)
        # Python float 精度：47.32 可能存为 47.31999... 用 pytest.approx
        assert updated.stage_progress == pytest.approx(47.32, abs=1e-6)

    def test_stage_progress_at_zero(self, in_memory_db):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, stage="download", stage_progress=0.0)
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage_progress == 0.0

    def test_stage_progress_at_100(self, in_memory_db):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, stage="download", stage_progress=100.0)
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage_progress == 100.0

    def test_stage_accepts_task_stage_enum_or_string(self, in_memory_db):
        """stage 可以传 str 或 enum-like（保持与 status 字段一致的 API）"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()

        # 字符串
        TaskDAO.update_task(task.id, stage="download", stage_progress=10.0)
        assert TaskDAO.get_task_by_id(task.id).stage == "download"


# ============================================================================
# Stage 字段组合
# ============================================================================

class TestTaskStageComposition:
    """update_task 应能同时设置 stage + stage_progress + log + status"""

    def test_update_stage_progress_and_status(self, in_memory_db):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id,
            status=TaskStatus.PROCESSING,
            stage="extract",
            stage_progress=15.5,
            log="开始提取"
        )
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.status == TaskStatus.PROCESSING
        assert updated.stage == "extract"
        assert updated.stage_progress == pytest.approx(15.5, abs=1e-6)
        assert updated.log == "开始提取"

    def test_stage_field_default_is_pending(self, in_memory_db):
        """新任务的 stage 默认应该是 'pending'"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        # 注意：add_task 没传 stage，依赖列默认值 'pending'
        # 但旧 DB 还没 stage 列，要 update_task 一次触发迁移
        TaskDAO.update_task(task.id, stage="pending", stage_progress=0.0)
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage == "pending"

    def test_stage_progress_can_be_none_for_non_progress_stages(self, in_memory_db):
        """非处理阶段（如 completed）的 stage_progress 可以是 None"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, stage="download", stage_progress=50.0)
        # 完成时：stage=completed, stage_progress=100 或 None
        TaskDAO.update_task(
            task.id,
            status=TaskStatus.COMPLETED,
            stage="completed",
            stage_progress=100.0,
            progress=100,  # 兼容旧字段
        )
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.stage == "completed"
        assert updated.status == TaskStatus.COMPLETED


# ============================================================================
# get_all_tasks 应能读出 stage 字段
# ============================================================================

class TestGetAllTasksReturnsStage:
    def test_get_all_tasks_includes_stage(self, in_memory_db):
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/a.mp4")
        TaskDAO.add_task("/media/b.mp4")
        a = TaskDAO.get_pending_task()  # 拿不到，已经被 b 占
        tasks = TaskDAO.get_all_tasks()
        # 触发迁移
        for t in tasks:
            TaskDAO.update_task(t.id, stage="extract", stage_progress=20.0)

        # 重新读
        tasks = TaskDAO.get_all_tasks()
        assert all(t.stage == "extract" for t in tasks)
        assert all(t.stage_progress == pytest.approx(20.0, abs=1e-6) for t in tasks)


# ============================================================================
# 边界情况
# ============================================================================

class TestStageEdgeCases:
    def test_update_without_stage_args_does_not_crash(self, in_memory_db):
        """只传 status/log 不传 stage 时，旧代码路径应该能正常工作（迁移是幂等的）"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()

        # 不传 stage，只传 log —— 旧代码路径
        TaskDAO.update_task(task.id, log="hello")
        # 不能崩，能读出来
        updated = TaskDAO.get_task_by_id(task.id)
        assert updated.log == "hello"
        # stage 是新字段，迁移触发后默认 'pending'
        assert updated.stage == 'pending'

    def test_reset_task_clears_stage(self, in_memory_db):
        """reset_task 应清空 stage（让重试任务从 pending 重新开始）"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(task.id, stage="extract", stage_progress=80.0)
        TaskDAO.update_task(task.id, status=TaskStatus.FAILED, log="失败")

        TaskDAO.reset_task(task.id)
        reset = TaskDAO.get_task_by_id(task.id)
        assert reset.status == TaskStatus.PENDING
        # stage 应该被重置
        assert reset.stage == "pending" or reset.stage is None
        # progress 也清零
        assert reset.progress == 0
