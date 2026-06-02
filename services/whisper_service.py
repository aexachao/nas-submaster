#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 字幕提取服务
负责从视频中提取字幕
"""

import os
import threading
from pathlib import Path
from typing import Optional, Callable
from faster_whisper import WhisperModel

from core.models import WhisperConfig, VADParameters
from utils.format_utils import format_timestamp, format_file_size


# CUDA/cuBLAS 相关错误的特征子串，用于判断是否需要回退到 CPU
_CUDA_ERROR_MARKERS = ("libcublas", "cuda", "cudnn")


class WhisperService:
    """Whisper 字幕提取服务"""
    
    def __init__(
        self,
        config: WhisperConfig,
        vad_params: VADParameters,
        model_dir: str = "/data/models"
    ):
        """
        初始化 Whisper 服务
        
        Args:
            config: Whisper 配置
            vad_params: VAD 参数
            model_dir: 模型存储目录
        """
        self.config = config
        self.vad_params = vad_params
        self.model_dir = model_dir
        self.model: Optional[WhisperModel] = None
    
    def _is_model_cached(self) -> bool:
        """检查模型文件是否已在本地缓存"""
        model_dir = Path(self.model_dir)
        # faster-whisper 在 download_root 下以模型名创建子目录
        # 目录存在且非空则视为已缓存
        for candidate in model_dir.glob(f"*{self.config.model_size}*"):
            if candidate.is_dir() and any(candidate.iterdir()):
                return True
        return False

    def _resolve_device(self) -> str:
        """解析实际使用的 device：WHISPER_DEVICE 环境变量优先于 config.whisper.device。"""
        env_device = os.environ.get("WHISPER_DEVICE")
        if env_device and env_device.strip():
            return env_device.strip()
        return self.config.device

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        """判断异常是否与 CUDA/cuBLAS 库加载失败相关（用于决定是否回退 CPU）。"""
        msg = str(exc).lower()
        return any(marker in msg for marker in _CUDA_ERROR_MARKERS)

    def load_model(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        加载 Whisper 模型

        行为：
        - device="auto"：先尝试 cuda，CUDA/cuBLAS 不可用时回退到 cpu
        - device="cuda" / "cpu"：显式指定，不静默回退，错误原样抛出
        - WHISPER_DEVICE 环境变量覆盖 config.whisper.device

        Args:
            progress_callback: 进度回调 (current, total, message)，用于在下载时上报进度
        """
        if self.model is not None:
            return

        is_cached = self._is_model_cached()

        if not is_cached and progress_callback:
            progress_callback(5, 100, f"首次使用，正在下载模型 {self.config.model_size}...")
            # 启动后台线程轮询下载目录大小，定时上报进度
            stop_event = threading.Event()
            model_dir = Path(self.model_dir)

            def _poll_download():
                while not stop_event.is_set():
                    try:
                        total_size = sum(
                            f.stat().st_size
                            for f in model_dir.rglob('*')
                            if f.is_file()
                        )
                        if total_size > 0:
                            progress_callback(
                                5, 100,
                                f"正在下载模型 {self.config.model_size}... "
                                f"已下载 {format_file_size(total_size)}"
                            )
                    except Exception:
                        pass
                    stop_event.wait(3)

            poll_thread = threading.Thread(target=_poll_download, daemon=True)
            poll_thread.start()
        else:
            stop_event = None

        # 决定本次加载的 device：env 优先；'auto' 表示先试 cuda 再回退 cpu
        device = self._resolve_device()
        compute_type = self.config.compute_type

        def _do_load(dev: str, ct: str):
            return WhisperModel(
                self.config.model_size,
                device=dev,
                compute_type=ct,
                download_root=self.model_dir
            )

        try:
            if device == "auto":
                # auto：先尝试 cuda，失败（CUDA/cuBLAS 不可用）回退到 cpu
                try:
                    self.model = _do_load("cuda", compute_type)
                    print(
                        f"[WhisperService] Model loaded: "
                        f"{self.config.model_size} (device=cuda)"
                    )
                except Exception as e:
                    if self._is_cuda_error(e):
                        print(
                            f"[WhisperService] CUDA 不可用 ({e})，回退到 CPU"
                        )
                        self.model = _do_load("cpu", compute_type)
                        print(
                            f"[WhisperService] Model loaded: "
                            f"{self.config.model_size} (device=cpu)"
                        )
                    else:
                        raise
            else:
                # 显式 cuda / cpu：不静默回退，错误原样向上抛
                self.model = _do_load(device, compute_type)
                print(f"[WhisperService] Model loaded: {self.config.model_size}")
        except Exception as e:
            print(f"[WhisperService] Failed to load model: {e}")
            raise
        finally:
            if stop_event:
                stop_event.set()
    
    def extract_subtitle(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        从视频中提取字幕
        
        Args:
            video_path: 视频文件路径
            output_path: 输出 SRT 文件路径（默认：同名 .srt）
            progress_callback: 进度回调函数 (current, total, message)
        
        Returns:
            生成的 SRT 文件路径
        """
        # 确保模型已加载（传入回调以支持下载进度上报）
        if self.model is None:
            self.load_model(progress_callback)
        
        # 确定输出路径
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.srt'))
        
        # 更新进度
        if progress_callback:
            progress_callback(5, 100, f"开始提取字幕...")
        
        # 准备转录参数
        transcribe_params = {
            'audio': video_path,
            'beam_size': 5,
            'vad_filter': True,
            'vad_parameters': self.vad_params.to_dict(),
            'word_timestamps': True,
            'condition_on_previous_text': True,
            'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        
        # 如果不是自动检测，指定语言
        if self.config.source_language != 'auto':
            transcribe_params['language'] = self.config.source_language
        
        try:
            # 执行转录
            segments, info = self.model.transcribe(**transcribe_params)
            
            # 更新进度
            if progress_callback:
                from utils.format_utils import get_lang_name
                lang_name = get_lang_name(info.language)
                progress_callback(15, 100, f"检测语言: {lang_name}")
            
            # 写入 SRT 文件
            with open(output_path, 'w', encoding='utf-8') as f:
                idx = 0
                for seg in segments:
                    idx += 1
                    
                    # 写入字幕条目
                    f.write(f"{idx}\n")
                    f.write(
                        f"{format_timestamp(seg.start)} --> "
                        f"{format_timestamp(seg.end)}\n"
                    )
                    f.write(f"{seg.text.strip()}\n\n")
                    
                    # 更新进度
                    if progress_callback and idx % 10 == 0:
                        progress = 15 + min(35, int(idx / 300 * 35))
                        progress_callback(progress, 100, f"已转写 {idx} 行")
            
            # 完成
            if progress_callback:
                progress_callback(50, 100, f"字幕提取完成 ({idx} 行)")
            
            return output_path
        
        except Exception as e:
            print(f"[WhisperService] Extraction failed: {e}")
            raise
    
    def unload_model(self):
        """卸载模型（释放内存）"""
        if self.model is not None:
            del self.model
            self.model = None
            print("[WhisperService] Model unloaded")


# ============================================================================
# 快捷函数
# ============================================================================

def extract_subtitle_from_video(
    video_path: str,
    config: WhisperConfig,
    vad_params: VADParameters,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> str:
    """
    从视频提取字幕（快捷函数）
    
    Args:
        video_path: 视频文件路径
        config: Whisper 配置
        vad_params: VAD 参数
        output_path: 输出路径（可选）
        progress_callback: 进度回调
    
    Returns:
        SRT 文件路径
    """
    service = WhisperService(config, vad_params)
    return service.extract_subtitle(video_path, output_path, progress_callback)
