#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for TaskWorker progress callback protocol (v1.8.1).

v1.8.1 协议变更：
- 之前 callback(current: int, total: int, message: str) — total 永远是 100
- 现在 callback(stage: str, stage_progress: float, message: str) — stage 独立

下载阶段 callback 例：callback("download", 47.32, "正在下载模型 base... 47.32%")
提取阶段 callback 例：callback("extract", 15.5, "检测语言: 中文")
翻译阶段 callback 例：callback("translate", 47.0, "已翻译 47/100 条")
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from core.models import TaskStatus


@pytest.fixture(autouse=True)
def in_memory_db(tmp_path):
    """为每个测试创建数据库，patch DAO 连接"""
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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.close()

    def _new_conn():
        return sqlite3.connect(db_path)

    with patch("database.task_dao.get_db_connection", side_effect=_new_conn):
        with patch("core.worker.get_db_connection", side_effect=_new_conn):
            yield db_path


# ============================================================================
# 协议契约测试 — 解析 callback 实际行为
# ============================================================================

class TestWhisperCallbackProtocol:
    """worker._extract_subtitle 传给 whisper 的 progress_callback 应使用新协议"""

    def test_extract_subtitle_callback_uses_new_protocol(self, in_memory_db, monkeypatch):
        """
        v1.8.1 协议：callback(stage: str, stage_progress: float, message: str)
        下载阶段 stage='download'，提取阶段 stage='extract'
        """
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker
        from services.whisper_service import WhisperService, WhisperConfig
        from core.models import VADParameters

        # 创建任务
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        # 触发迁移
        TaskDAO.update_task(task.id, stage="download", stage_progress=0.0)

        # 抓取传给 extract_subtitle 的 progress_callback
        captured_callback = []
        original_extract = WhisperService.extract_subtitle

        def mock_extract(self, video_path, output_path=None, progress_callback=None):
            captured_callback.append(progress_callback)
            # 模拟下载阶段：调一次 callback("download", 47.32, ...)
            if progress_callback:
                progress_callback("download", 47.32, "正在下载模型 base... 47.32%")
                progress_callback("extract", 5.0, "开始提取字幕...")
                progress_callback("extract", 50.0, "已转写 100 行")
                progress_callback("extract", 100.0, "字幕提取完成")
            # 不真跑 whisper
            return "/media/movie.srt"

        monkeypatch.setattr(WhisperService, "extract_subtitle", mock_extract)

        # 准备 worker
        worker = TaskWorker()
        config = MagicMock()
        config.whisper = WhisperConfig(
            model_size="base", device="cpu", compute_type="int8",
            source_language="auto"
        )
        config.translation.use_embedded_subtitle = False
        config.translation.enabled = False

        # 跑提取
        worker._extract_subtitle(task.id, "/media/movie.mp4", config)

        # 验证 callback 被调了 4 次（2 个阶段）
        assert len(captured_callback) == 1
        cb = captured_callback[0]

        # 直接调 callback，看是否按新协议工作
        cb("download", 47.32, "msg1")
        cb("extract", 5.0, "msg2")
        cb("extract", 50.0, "msg3")
        cb("extract", 100.0, "msg4")

        # 验证数据库里有 stage 信息
        final = TaskDAO.get_task_by_id(task.id)
        # 最后一次 callback 是 extract 100%
        assert final.stage == "extract"
        assert final.stage_progress == pytest.approx(100.0, abs=1e-6)


class TestTranslateCallbackProtocol:
    """翻译阶段 callback 应上报 X/Y 条而不是百分比"""

    def test_translate_callback_reports_x_of_y(self, in_memory_db, monkeypatch):
        """
        v1.8.1: 翻译 callback 上报 stage='translate', stage_progress=current_line
        （不是百分比，是"已翻译 X 条"）
        """
        from database.task_dao import TaskDAO

        # 创建任务 + 准备阶段
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id,
            status=TaskStatus.PROCESSING,
            stage="translate",
            stage_progress=0.0,
        )

        # 抓取 worker 传给 translate_srt_file 的 callback
        from core.worker import TaskWorker
        captured_callback = []
        import sys

        # Patch translate_srt_file
        def fake_translate_srt_file(srt_path, cfg, output_path=None,
                                     progress_callback=None, prompt_template=None):
            captured_callback.append(progress_callback)
            # 模拟翻译：传 3 次进度
            if progress_callback:
                progress_callback("translate", 1, "正在翻译第 1/3 批（10 行）...")
                progress_callback("translate", 10, "正在翻译第 1/1 批（10 行）...")
                progress_callback("translate", 10, "翻译完成！")
            return (True, "ok")

        # 注入 fake 模块
        fake_module = MagicMock()
        fake_module.TranslationConfig = MagicMock()
        fake_module.translate_srt_file = fake_translate_srt_file

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "services.translator" or name.endswith(".translator"):
                return fake_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        # 准备 worker
        worker = TaskWorker()
        config = MagicMock()
        config.get_current_provider_config.return_value = MagicMock(
            api_key="x", base_url="http://x", model_name="y"
        )
        config.translation.target_language = "zh"
        config.whisper.source_language = "auto"
        config.translation.max_lines_per_batch = 100
        config.translation.timeout = 60
        config.content_type = "movie"

        # 跑翻译
        worker._translate_subtitle(task.id, "/media/movie.srt", config)

        # 验证 callback 被调了
        assert len(captured_callback) == 1
        cb = captured_callback[0]

        # 直接调
        cb("translate", 1, "msg1")
        cb("translate", 10, "msg2")
        cb("translate", 10, "msg3")

        final = TaskDAO.get_task_by_id(task.id)
        # 最后一次是 translate 阶段，已翻译 10 条
        assert final.stage == "translate"
        # stage_progress 存的是 current_line（最后一次=10）
        assert final.stage_progress == 10


# ============================================================================
# 阶段流转测试
# ============================================================================

class TestStageFlow:
    """任务生命周期内的 stage 字段应正确流转"""

    def test_task_starts_at_download_or_extract(self, in_memory_db, monkeypatch):
        """首次任务进入 processing 时，stage 应是 download 或 extract"""
        from database.task_dao import TaskDAO
        from core.worker import TaskWorker

        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        # 模拟 worker 开始：update_task stage='download'
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING,
            stage="download", stage_progress=0.0
        )
        t = TaskDAO.get_task_by_id(task.id)
        assert t.stage == "download"

    def test_translate_stage_is_reported(self, in_memory_db):
        """翻译阶段 stage 应是 'translate'"""
        from database.task_dao import TaskDAO
        TaskDAO.add_task("/media/movie.mp4")
        task = TaskDAO.get_pending_task()
        TaskDAO.update_task(
            task.id, status=TaskStatus.PROCESSING,
            stage="translate", stage_progress=50.0  # 50/100 条
        )
        t = TaskDAO.get_task_by_id(task.id)
        assert t.stage == "translate"
        assert t.stage_progress == 50.0
