#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后台任务处理器
负责处理任务队列中的视频字幕提取和翻译
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional

from core.models import TaskStatus
from core.config import AppConfig, ConfigManager
from database.connection import wait_for_database, get_db_connection
from database.task_dao import TaskDAO
from services.media_scanner import rescan_video_subtitles
from services.whisper_service import WhisperService


class TaskWorker:
    """任务处理器"""

    def __init__(self):
        """初始化任务处理器"""
        self.running = False
        self.config_manager = ConfigManager(get_db_connection)
        # 缓存 Whisper 服务实例，避免每个任务重复加载模型
        self._whisper_service: Optional[WhisperService] = None
        self._whisper_config_key: Optional[str] = None

    def start(self):
        """启动处理器（在独立线程中运行）"""
        if self.running:
            print("[TaskWorker] Already running")
            return

        # 等待数据库就绪
        if not wait_for_database():
            print("[TaskWorker] Database not ready, worker stopped")
            return

        # 将上次崩溃遗留的 PROCESSING 任务重置为 PENDING，避免死锁
        TaskDAO.reset_stale_processing_tasks()

        print("[TaskWorker] Starting...")
        self.running = True

        # 启动处理循环
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def stop(self):
        """停止处理器"""
        print("[TaskWorker] Stopping...")
        self.running = False

    def _worker_loop(self):
        """工作循环（持续处理任务）"""
        while self.running:
            try:
                # 加载最新配置
                config = self.config_manager.load()

                # 获取待处理任务
                task = TaskDAO.get_pending_task()

                if task:
                    print(f"[TaskWorker] Processing task {task.id}: {task.file_path}")
                    self._process_task(task.id, task.file_path, config)
                else:
                    # 无任务时休眠
                    time.sleep(5)

            except Exception as e:
                print(f"[TaskWorker] Error in worker loop: {e}")
                time.sleep(10)

    def _get_whisper_service(self, config: AppConfig) -> WhisperService:
        """
        获取（或复用）Whisper 服务实例。
        仅在模型/设备/精度配置变更时才重新加载模型。
        """
        config_key = (
            f"{config.whisper.model_size}|"
            f"{config.whisper.device}|"
            f"{config.whisper.compute_type}"
        )
        if self._whisper_service is None or self._whisper_config_key != config_key:
            if self._whisper_service is not None:
                self._whisper_service.unload_model()
            vad_params = config.get_vad_parameters()
            self._whisper_service = WhisperService(config.whisper, vad_params)
            self._whisper_config_key = config_key
        else:
            # 即使不重建服务，也要更新 VAD 参数（内容类型可能改变）
            self._whisper_service.vad_params = config.get_vad_parameters()

        return self._whisper_service

    def _process_task(self, task_id: int, file_path: str, config: AppConfig):
        """
        处理单个任务

        流程：提取字幕 → 翻译（可选）→ 导出格式 → 更新媒体库 → 标记完成
        """
        try:
            # 更新任务状态
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=0,
                log="任务启动"
            )

            # 检查文件是否存在
            if not os.path.exists(file_path):
                TaskDAO.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    log="文件丢失"
                )
                return

            # 步骤 1: Whisper 提取字幕
            srt_path = self._extract_subtitle(task_id, file_path, config)
            if not srt_path:
                return  # 提取失败，状态已在内部设置

            # 步骤 2: 翻译字幕（如果启用）
            if config.translation.enabled:
                success = self._translate_subtitle(task_id, srt_path, config)
                if not success:
                    return  # 翻译失败，状态已在内部设置

            # 步骤 3: 导出其他格式（在标记完成之前）
            self._export_formats(task_id, file_path, config)

            # 步骤 4: 更新媒体库（在标记完成之前）
            rescan_video_subtitles(file_path)

            # 步骤 5: 标记任务完成
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                log="完成"
            )

            print(f"[TaskWorker] Task {task_id} completed")

        except Exception as e:
            print(f"[TaskWorker] Task {task_id} failed: {e}")
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                log=f"异常: {str(e)[:100]}"
            )

    def _extract_subtitle(
        self,
        task_id: int,
        file_path: str,
        config: AppConfig
    ) -> Optional[str]:
        """
        提取字幕（步骤 1）

        Returns:
            SRT 文件路径，失败则返回 None
        """
        srt_path = Path(file_path).with_suffix('.srt')

        # 如果字幕已存在，跳过提取
        if srt_path.exists():
            TaskDAO.update_task(task_id, progress=50, log="基础字幕已存在")
            return str(srt_path)

        try:
            TaskDAO.update_task(
                task_id,
                progress=5,
                log=f"加载 Whisper ({config.whisper.model_size})..."
            )

            whisper = self._get_whisper_service(config)

            def progress_callback(current, total, message):
                TaskDAO.update_task(task_id, progress=current, log=message)

            whisper.extract_subtitle(
                file_path,
                str(srt_path),
                progress_callback
            )

            return str(srt_path)

        except Exception as e:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                log=f"提取失败: {str(e)[:100]}"
            )
            return None

    def _translate_subtitle(
        self,
        task_id: int,
        srt_path: str,
        config: AppConfig
    ) -> bool:
        """
        翻译字幕（步骤 2）

        Returns:
            True 表示成功，False 表示失败（状态已在内部更新）
        """
        TaskDAO.update_task(task_id, progress=50, log="准备翻译...")

        try:
            from services.translator import (
                TranslationConfig,
                translate_srt_file
            )

            # 构建翻译配置
            provider_cfg = config.get_current_provider_config()
            trans_config = TranslationConfig(
                api_key=provider_cfg.api_key,
                base_url=provider_cfg.base_url,
                model_name=provider_cfg.model_name,
                target_language=config.translation.target_language,
                source_language=config.whisper.source_language,
                max_lines_per_batch=config.translation.max_lines_per_batch
            )

            def progress_callback(current, total, message):
                progress = 50 + int((current / total) * 45)
                TaskDAO.update_task(task_id, progress=progress, log=message)

            success, msg = translate_srt_file(
                srt_path,
                trans_config,
                progress_callback=progress_callback
            )

            if not success:
                TaskDAO.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    progress=100,
                    log=f"翻译失败: {msg}"
                )
                return False

            return True

        except ImportError:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=100,
                log="翻译模块未安装"
            )
            return False
        except Exception as e:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=100,
                log=f"翻译异常: {str(e)[:100]}"
            )
            return False

    def _export_formats(
        self,
        task_id: int,
        file_path: str,
        config: AppConfig
    ):
        """
        导出其他格式（步骤 3）
        """
        try:
            from services.subtitle_converter import SubtitleConverter  # 修复：正确的导入路径

            srt_path = Path(file_path).with_suffix('.srt')
            exported_formats = []

            for fmt in config.export.formats:
                if fmt == 'srt':
                    continue  # SRT 已生成

                try:
                    SubtitleConverter.convert_file(str(srt_path), fmt)
                    exported_formats.append(fmt.upper())

                    # 如果有翻译版本，也转换
                    if config.translation.enabled:
                        trans_srt = Path(file_path).parent / \
                                   f"{Path(file_path).stem}.{config.translation.target_language}.srt"
                        if trans_srt.exists():
                            SubtitleConverter.convert_file(str(trans_srt), fmt)

                except Exception as e:
                    print(f"[TaskWorker] Failed to export {fmt}: {e}")

            if exported_formats:
                current_task = TaskDAO.get_task_by_id(task_id)
                if current_task:
                    TaskDAO.update_task(
                        task_id,
                        log=f"{current_task.log}（已导出: {', '.join(exported_formats)}）"
                    )

        except ImportError:
            pass  # 转换器模块未安装


# ============================================================================
# 全局工作器实例
# ============================================================================

_worker_instance: Optional[TaskWorker] = None


def start_worker():
    """启动全局工作器"""
    global _worker_instance

    if _worker_instance is None:
        _worker_instance = TaskWorker()

    _worker_instance.start()


def stop_worker():
    """停止全局工作器"""
    global _worker_instance

    if _worker_instance:
        _worker_instance.stop()


def get_worker() -> Optional[TaskWorker]:
    """获取全局工作器实例"""
    return _worker_instance
