#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型定义
集中管理所有数据结构，避免重复定义
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ============================================================================
# 枚举类型
# ============================================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class ContentType(Enum):
    """视频内容类型（影响 VAD 参数）"""
    MOVIE = 'movie'              # 电影/剧集
    DOCUMENTARY = 'documentary'  # 纪录片/新闻
    VARIETY = 'variety'          # 综艺/访谈
    ANIMATION = 'animation'      # 动画/动漫
    LECTURE = 'lecture'          # 讲座/课程
    MUSIC = 'music'              # 音乐视频/MV
    CUSTOM = 'custom'            # 自定义


# ============================================================================
# 字幕相关模型
# ============================================================================

@dataclass
class SubtitleInfo:
    """字幕文件信息"""
    path: str
    lang: str
    tag: str
    
    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'lang': self.lang,
            'tag': self.tag
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleInfo':
        return cls(
            path=data['path'],
            lang=data['lang'],
            tag=data['tag']
        )


@dataclass
class SubtitleEntry:
    """通用字幕条目"""
    index: str
    timecode: str
    text: str
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'timecode': self.timecode,
            'text': self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleEntry':
        return cls(
            index=str(data.get('index', '1')),
            timecode=str(data.get('timecode', '00:00:00,000 --> 00:00:01,000')),
            text=str(data.get('text', ''))
        )


# ============================================================================
# 配置模型
# ============================================================================

@dataclass
class VADParameters:
    """VAD (Voice Activity Detection) 参数"""
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    
    def to_dict(self) -> Dict:
        return {
            'threshold': self.threshold,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'speech_pad_ms': self.speech_pad_ms
        }


@dataclass
class ProviderConfig:
    """LLM 提供商配置"""
    api_key: str = ''
    base_url: str = ''
    model_name: str = ''
    
    def to_dict(self) -> Dict:
        return {
            'api_key': self.api_key,
            'base_url': self.base_url,
            'model_name': self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProviderConfig':
        return cls(
            api_key=data.get('api_key', ''),
            base_url=data.get('base_url', ''),
            model_name=data.get('model_name', '')
        )


@dataclass
class WhisperConfig:
    """Whisper 模型配置"""
    model_size: str = 'base'
    compute_type: str = 'int8'
    device: str = 'cpu'
    source_language: str = 'auto'
    
    def to_dict(self) -> Dict:
        return {
            'model_size': self.model_size,
            'compute_type': self.compute_type,
            'device': self.device,
            'source_language': self.source_language
        }


@dataclass
class TranslationConfig:
    """翻译配置"""
    enabled: bool = False
    target_language: str = 'zh'
    max_lines_per_batch: int = 500
    max_retries: int = 3
    timeout: int = 180
    
    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'target_language': self.target_language,
            'max_lines_per_batch': self.max_lines_per_batch,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }


@dataclass
class ExportConfig:
    """导出配置"""
    formats: List[str] = field(default_factory=lambda: ['srt'])
    
    def to_dict(self) -> Dict:
        return {'formats': self.formats}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExportConfig':
        return cls(formats=data.get('formats', ['srt']))


# ============================================================================
# 任务模型
# ============================================================================

@dataclass
class Task:
    """任务实体"""
    id: int
    file_path: str
    status: TaskStatus
    progress: int = 0
    log: str = ''
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'file_path': self.file_path,
            'status': self.status.value if isinstance(self.status, TaskStatus) else self.status,
            'progress': self.progress,
            'log': self.log,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        status = data['status']
        if isinstance(status, str):
            status = TaskStatus(status)
        
        return cls(
            id=data['id'],
            file_path=data['file_path'],
            status=status,
            progress=data.get('progress', 0),
            log=data.get('log', ''),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


# ============================================================================
# 媒体文件模型
# ============================================================================

@dataclass
class MediaFile:
    """媒体文件实体"""
    id: int
    file_path: str
    file_name: str
    file_size: int
    subtitles: List[SubtitleInfo] = field(default_factory=list)
    has_translated: bool = False
    updated_at: Optional[str] = None
    
    @property
    def has_subtitle(self) -> bool:
        """是否有字幕"""
        return len(self.subtitles) > 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'subtitles': [s.to_dict() for s in self.subtitles],
            'has_subtitle': self.has_subtitle,
            'has_translated': self.has_translated,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MediaFile':
        subtitles_data = data.get('subtitles', [])
        if isinstance(subtitles_data, str):
            import json
            subtitles_data = json.loads(subtitles_data)
        
        subtitles = [SubtitleInfo.from_dict(s) for s in subtitles_data]
        
        return cls(
            id=data['id'],
            file_path=data['file_path'],
            file_name=data['file_name'],
            file_size=data['file_size'],
            subtitles=subtitles,
            has_translated=data.get('has_translated', False),
            updated_at=data.get('updated_at')
        )


# ============================================================================
# 常量定义
# ============================================================================

# 支持的视频格式
SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.mov', '.avi', 
    '.flv', '.wmv', '.m4v', '.webm', '.ts'
}

# 支持的字幕格式
SUPPORTED_SUBTITLE_FORMATS = {
    'srt', 'vtt', 'ass', 'ssa', 'sub'
}

# 语言代码映射
ISO_LANG_MAP = {
    'auto': '自动检测',
    'zh': '中文', 'en': '英语', 'ja': '日语', 'ko': '韩语',
    'fr': '法语', 'de': '德语', 'ru': '俄语', 'es': '西班牙语',
    'chs': '简中', 'cht': '繁中', 'eng': '英语', 
    'jpn': '日语', 'kor': '韩语',
    'unknown': '未知'
}

# 目标语言选项
TARGET_LANG_OPTIONS = ['zh', 'en', 'ja', 'ko']

# Whisper 源语言选项（仅包含合法的 ISO 639-1 代码）
WHISPER_SOURCE_LANG_MAP = {
    'auto': '自动检测',
    'zh': '中文',
    'en': '英语',
    'ja': '日语',
    'ko': '韩语',
    'fr': '法语',
    'de': '德语',
    'ru': '俄语',
    'es': '西班牙语',
}