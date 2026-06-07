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
from typing import Optional, Callable

from core.models import TaskStatus
from core.config import AppConfig, ConfigManager
from database.connection import wait_for_database, get_db_connection
from database.task_dao import TaskDAO
from services.media_scanner import rescan_video_subtitles, scan_media_directory
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
        # 取消标志：Worker 轮询此事件，设置后当前任务尽快退出
        self._cancel_event = threading.Event()
        # 自动扫描：上次扫描时间戳
        self._last_scan_time: float = 0

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

    def request_cancel(self):
        """请求取消当前正在处理的任务"""
        self._cancel_event.set()

    def _check_cancelled(self, task_id: int) -> bool:
        """
        检查当前任务是否已被请求取消（通过事件标志或数据库状态）

        Returns:
            True 表示应取消，False 表示继续
        """
        if self._cancel_event.is_set():
            return True
        # 也检查数据库中的状态，支持多进程场景下的取消
        task = TaskDAO.get_task_by_id(task_id)
        return task is not None and task.status == TaskStatus.CANCELLED

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
                    # 无任务时：检查是否需要自动扫描
                    if config.auto_scan_enabled:
                        interval_sec = config.auto_scan_interval_minutes * 60
                        if time.time() - self._last_scan_time >= interval_sec:
                            print("[TaskWorker] Auto-scanning media library...")
                            scan_media_directory()
                            self._last_scan_time = time.time()
                    # 休眠
                    time.sleep(5)

            except Exception as e:
                print(f"[TaskWorker] Error in worker loop: {e}")
                time.sleep(10)

    def _get_whisper_service(self, config: AppConfig) -> WhisperService:
        """
        获取（或复用）Whisper 服务实例。
        仅在模型/设备/精度配置变更时才重新加载模型。

        2026-06-06 改进：在重建 service 前先强制验证缓存完整性。
        之前用户报告"取消下载后重试新任务卡在准备中" —— 根因是：
        1. 用户取消时 WhisperModel 内部 hf_hub 还在下载
        2. 取消信号被吞,hf_hub 完成后 WhisperModel 加载半下载文件失败抛异常
        3. 异常被 worker 外层 except 接住,任务标 FAILED
        4. 但半下载文件残留在磁盘上,_is_model_cached 假阳性
        5. 新任务进来 → 复用旧 service(model 已损坏)/或重新加载半下载文件 → 死循环
        修复:重建 service 前先验证缓存完整性,失败就清理半下载文件,
        强制 hf_hub 重新下载(完整且可中断的下载流程)。
        """
        config_key = (
            f"{config.whisper.model_size}|"
            f"{config.whisper.device}|"
            f"{config.whisper.compute_type}"
        )
        if self._whisper_service is None or self._whisper_config_key != config_key:
            if self._whisper_service is not None:
                # 2026-06-06: 卸载旧 service 前先验证缓存完整性
                # 如果半下载,把残留文件清掉,避免下次加载卡死
                self._whisper_service.unload_model()
                if not self._whisper_service._verify_model_files():
                    print(
                        f"[TaskWorker] 检测到半下载的 {config.whisper.model_size} 模型,清理中..."
                    )
                    self._cleanup_partial_model(config.whisper.model_size)
            vad_params = config.get_vad_parameters()
            self._whisper_service = WhisperService(config.whisper, vad_params)
            self._whisper_config_key = config_key
        else:
            # 即使不重建服务，也要更新 VAD 参数（内容类型可能改变）
            self._whisper_service.vad_params = config.get_vad_parameters()

        return self._whisper_service

    def _cleanup_partial_model(self, model_size: str):
        """
        清理半下载的模型文件。

        HF 缓存目录结构:
        {model_dir}/models--Systran--faster-whisper-{size}/
          blobs/         # 实际文件(可能是 .incomplete)
          snapshots/     # 软链接/指针
          refs/          # 引用信息

        半下载特征:
        - blobs/ 下有 .incomplete 文件
        - snapshots/ 下某个 commit 目录存在但缺关键文件
        - refs/main 指向不存在的 commit

        清理策略:直接删除整个 models--Systran--faster-whisper-{size} 目录,
        下次加载会重新下载(完整流程,可中断)。
        """
        from services.whisper_service import get_model_dir
        import shutil
        model_dir = Path(get_model_dir())
        repo_dir = model_dir / f"models--Systran--faster-whisper-{model_size}"
        if repo_dir.exists():
            try:
                shutil.rmtree(repo_dir)
                print(f"[TaskWorker] 已清理半下载模型: {repo_dir}")
            except Exception as e:
                print(f"[TaskWorker] 清理半下载模型失败: {e}")

    def _check_translation_config(self, config: AppConfig) -> tuple[bool, str]:
        """
        检查翻译配置是否完整

        Returns:
            (是否通过, 错误消息)
        """
        provider_cfg = config.get_current_provider_config()

        # Ollama 本地模型：只需检查 base_url 和 model_name
        if config.current_provider == "Ollama (本地模型)":
            if not provider_cfg.base_url or not provider_cfg.base_url.strip():
                return False, "Ollama 未配置 Base URL，请在设置中配置"
            if not provider_cfg.model_name or not provider_cfg.model_name.strip():
                return False, "Ollama 未选择模型，请在设置中配置"
        else:
            # 其他在线 API：检查 api_key, base_url, model_name
            if not provider_cfg.api_key or not provider_cfg.api_key.strip():
                return False, f"{config.current_provider} 未配置 API Key，请在设置中配置"
            if not provider_cfg.base_url or not provider_cfg.base_url.strip():
                return False, f"{config.current_provider} 未配置 Base URL，请在设置中配置"
            if not provider_cfg.model_name or not provider_cfg.model_name.strip():
                return False, f"{config.current_provider} 未选择模型，请在设置中配置"

        return True, ""

    def _process_task(self, task_id: int, file_path: str, config: AppConfig):
        """
        处理单个任务

        流程：提取字幕 → 翻译（可选）→ 导出格式 → 更新媒体库 → 标记完成
        """
        self._cancel_event.clear()  # 每个新任务开始前清除上次的取消信号

        try:
            # 更新任务状态
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=0,
                log="任务启动",
                append_log=True
            )

            # 检查文件是否存在
            if not os.path.exists(file_path):
                TaskDAO.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    log="文件丢失",
                    append_log=True
                )
                return

            # 步骤 1: 提取字幕（优先内置字幕，否则 Whisper）
            srt_path = self._extract_or_detect_subtitle(task_id, file_path, config)
            if not srt_path:
                return  # 提取失败或已取消，状态已在内部设置

            if self._check_cancelled(task_id):
                TaskDAO.update_task(task_id, status=TaskStatus.CANCELLED, log="已取消", append_log=True)
                return

            # 步骤 1.5: 检查翻译配置（如果启用）
            if config.translation.enabled:
                config_ok, config_msg = self._check_translation_config(config)
                if not config_ok:
                    TaskDAO.update_task(
                        task_id,
                        status=TaskStatus.FAILED,
                        log=f"配置错误: {config_msg}",
                        append_log=True
                    )
                    return

            # 步骤 2: 翻译字幕（如果启用）
            if config.translation.enabled:
                success = self._translate_subtitle(task_id, srt_path, config)
                if not success:
                    return  # 翻译失败或已取消

            if self._check_cancelled(task_id):
                TaskDAO.update_task(task_id, status=TaskStatus.CANCELLED, log="已取消", append_log=True)
                return

            # 步骤 3: 导出其他格式（在标记完成之前）
            self._export_formats(task_id, file_path, config)

            # 步骤 4: 更新媒体库（在标记完成之前）
            rescan_video_subtitles(file_path)

            # 步骤 5: 标记任务完成
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                log="完成",
                append_log=True
            )

            print(f"[TaskWorker] Task {task_id} completed")

        except InterruptedError:
            print(f"[TaskWorker] Task {task_id} cancelled")
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.CANCELLED,
                log="已取消",
                append_log=True
            )
        except Exception as e:
            print(f"[TaskWorker] Task {task_id} failed: {e}")
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                log=f"异常: {str(e)[:100]}",
                append_log=True
            )

    def _extract_or_detect_subtitle(
        self,
        task_id: int,
        file_path: str,
        config: AppConfig
    ) -> Optional[str]:
        """
        提取字幕：优先使用内置字幕，否则使用 Whisper 识别

        步骤：
        1. 如果已有 SRT 字幕，直接使用
        2. 如果开启"优先使用内置字幕"，检测并尝试提取内置字幕
        3. 以上都失败则回退到 Whisper 识别

        Returns:
            SRT 文件路径，失败则返回 None
        """
        srt_path = Path(file_path).with_suffix('.srt')

        # 如果字幕已存在，跳过提取
        if srt_path.exists():
            TaskDAO.update_task(task_id, progress=50, log="字幕已存在", append_log=True)
            return str(srt_path)

        # 如果开启"优先使用内置字幕"，尝试检测
        if config.translation.use_embedded_subtitle:
            embedded_srt = self._try_extract_embedded_subtitle(task_id, file_path, config)
            if embedded_srt:
                return embedded_srt

        # 回退到 Whisper 识别
        return self._extract_subtitle(task_id, file_path, config)

    def _try_extract_embedded_subtitle(
        self,
        task_id: int,
        file_path: str,
        config: AppConfig
    ) -> Optional[str]:
        """
        尝试提取内置字幕

        Returns:
            提取成功返回 SRT 路径，失败返回 None
        """
        try:
            from services.subtitle_extractor import SubtitleExtractor

            TaskDAO.update_task(task_id, progress=5, log="检测内置字幕...", append_log=True)

            # 检测字幕轨道
            tracks = SubtitleExtractor.detect_subtitle_tracks(file_path)
            if not tracks:
                TaskDAO.update_task(task_id, log="未检测到内置字幕，回退到 Whisper", append_log=True)
                return None

            # 选择最佳字幕轨道（非目标语言）
            target_lang = config.translation.target_language
            best_track = SubtitleExtractor.select_best_subtitle_track(tracks, target_lang)

            if not best_track:
                TaskDAO.update_task(task_id, log="无合适的字幕轨道，回退到 Whisper", append_log=True)
                return None

            TaskDAO.update_task(
                task_id,
                progress=10,
                log=f"提取内置字幕: {best_track.display_name}",
                append_log=True
            )

            # 提取字幕
            srt_path = SubtitleExtractor.extract_subtitle(
                file_path,
                best_track.stream_index,
                output_format='srt',
                embedded=True
            )

            if srt_path and Path(srt_path).exists():
                # 立即更新数据库：标记为内置字幕
                from database.media_dao import MediaDAO
                from core.models import SubtitleInfo
                embedded_subtitle = SubtitleInfo(
                    path=srt_path,
                    lang=best_track.language or 'unknown',
                    source='embedded'
                )
                # 更新该视频的字幕信息
                MediaDAO.update_media_subtitles(file_path, [embedded_subtitle], False)
                TaskDAO.update_task(task_id, progress=50, log="内置字幕提取成功", append_log=True)
                return srt_path
            else:
                TaskDAO.update_task(task_id, log="内置字幕提取失败，回退到 Whisper", append_log=True)
                return None

        except Exception as e:
            print(f"[TaskWorker] Failed to extract embedded subtitle: {e}")
            TaskDAO.update_task(task_id, log="内置字幕提取失败，回退到 Whisper", append_log=True)
            return None

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
            TaskDAO.update_task(task_id, progress=50, log="基础字幕已存在", append_log=True)
            return str(srt_path)

        try:
            TaskDAO.update_task(
                task_id,
                progress=5,
                log=f"加载 Whisper ({config.whisper.model_size})...",
                append_log=True
            )

            whisper = self._get_whisper_service(config)

            def progress_callback(stage, stage_progress, message):
                """
                v1.8.1 协议：callback(stage, stage_progress, message)
                - stage: 'download' / 'extract' / 'translate' / ...
                - stage_progress: 0.0-100.0 浮点（两位小数），或翻译阶段的 current_line 整数
                - 提取阶段既上报 'download'（whisper 模型下载中）也上报 'extract'（转写中）
                """
                # 进度回调仅覆盖 log（高频更新不写历史，避免数据库膨胀）
                TaskDAO.update_task(
                    task_id, log=message,
                    stage=stage, stage_progress=stage_progress,
                )
                # 在回调中检测取消，触发后通过异常中断 Whisper 提取
                if self._check_cancelled(task_id):
                    raise InterruptedError("任务已取消")

            whisper.extract_subtitle(
                file_path,
                str(srt_path),
                progress_callback
            )

            TaskDAO.update_task(task_id, log="字幕提取完成", append_log=True)
            return str(srt_path)

        except Exception as e:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                log=f"提取失败: {str(e)[:100]}",
                append_log=True
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
        TaskDAO.update_task(task_id, progress=50, log="准备翻译...", append_log=True)

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
                max_lines_per_batch=config.translation.max_lines_per_batch,
                timeout=config.translation.timeout
            )

            # 获取当前内容类型对应的提示词模板
            prompt_template = config.get_prompt_template(config.content_type)

            def progress_callback(stage, stage_progress, message):
                """
                v1.8.1 协议：翻译阶段 callback 上报 stage='translate'，stage_progress 是
                当前已翻译条数（不是百分比 —— LLM 流式输出长度不可预测）。
                """
                # 翻译进度高频更新，仅覆盖 log 不写历史
                TaskDAO.update_task(
                    task_id, log=message,
                    stage=stage, stage_progress=stage_progress,
                )

            success, msg = translate_srt_file(
                srt_path,
                trans_config,
                progress_callback=progress_callback,
                prompt_template=prompt_template
            )

            if not success:
                TaskDAO.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    progress=100,
                    log=f"翻译失败: {msg}",
                    append_log=True
                )
                return False

            TaskDAO.update_task(task_id, log="翻译完成", append_log=True)
            return True

        except ImportError:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=100,
                log="翻译模块未安装",
                append_log=True
            )
            return False
        except Exception as e:
            TaskDAO.update_task(
                task_id,
                status=TaskStatus.FAILED,
                progress=100,
                log=f"翻译异常: {str(e)[:100]}",
                append_log=True
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
