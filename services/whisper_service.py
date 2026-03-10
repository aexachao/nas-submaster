#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 字幕提取服务
负责从视频中提取字幕
"""

from pathlib import Path
from typing import Optional, Callable
from faster_whisper import WhisperModel

from core.models import WhisperConfig, VADParameters
from utils.format_utils import format_timestamp


class WhisperService:
    """Whisper 字幕提取服务"""
    
    def __init__(
        self,
        config: WhisperConfig,
        vad_params: VADParameters,
        model_dir: str = "./data/models"
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
    
    def load_model(self):
        """加载 Whisper 模型"""
        if self.model is not None:
            return
        
        try:
            self.model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=self.model_dir
            )
            print(f"[WhisperService] Model loaded: {self.config.model_size}")
        except Exception as e:
            print(f"[WhisperService] Failed to load model: {e}")
            raise
    
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
        # 确保模型已加载
        if self.model is None:
            self.load_model()
        
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
