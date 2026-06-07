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
    CANCELLED = 'cancelled'


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

# 字幕来源枚举
class SubtitleSource(Enum):
    """字幕来源"""
    EMBEDDED = 'embedded'    # 视频内置字幕
    ASR = 'asr'            # Whisper语音识别
    TRANSLATED = 'translated'  # 翻译生成


# 字幕来源显示标签
SUBTITLE_SOURCE_LABELS = {
    'embedded': '内置',
    'asr': 'AI提取',
    'translated': '已翻译'
}


@dataclass
class SubtitleInfo:
    """字幕文件信息"""
    path: str
    lang: str           # 语言代码
    source: str = 'embedded'  # 来源：embedded/asr/translated

    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'lang': self.lang,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleInfo':
        return cls(
            path=data['path'],
            lang=data.get('lang', data.get('tag', '')),
            source=data.get('source', 'embedded')
        )

    @property
    def source_label(self) -> str:
        """获取来源显示标签"""
        return SUBTITLE_SOURCE_LABELS.get(self.source, '内置')

    @property
    def display_name(self) -> str:
        """获取显示名称，如"内置：中文" """
        lang_names = {
            'zh': '中文', 'chs': '中文', 'cht': '中文',
            'en': '英文', 'eng': '英文',
            'ja': '日文', 'jpn': '日文',
            'ko': '韩文', 'kor': '韩文',
            'fr': '法文', 'de': '德文', 'ru': '俄文', 'es': '西班牙文'
        }
        lang_name = lang_names.get(self.lang.lower(), self.lang.upper())
        return f"{self.source_label}：{lang_name}"


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
    """Whisper 模型配置

    v1.8.2: 默认 model_size 从 'base' 改为 'medium'。
    原因：base 141MB 模型对长视频识别准确率偏低（特别是多语言/有口音场景），
    medium 1.5GB 准确率显著更好，NAS 场景下值得多花下载时间。
    内存 ≥ 8GB 都跑得动 int8 量化。
    升级后首次运行会下载 1.5GB 模型（之前 base 是 141MB）。
    """
    model_size: str = 'medium'  # v1.8.2+: 默认 medium
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
    """翻译配置

    v1.8.2: 默认 enabled 从 False 改为 True。
    原因：用户装这个工具就是为了"提取+翻译"，默认开翻译更符合使用场景。
    升级时已经存了 enable_translation=false 的用户不受影响（DB 里有值就用 DB 的）。
    """
    enabled: bool = True  # v1.8.2+: 默认开启
    target_language: str = 'zh'
    use_embedded_subtitle: bool = True  # 优先使用内置字幕（如果有）
    max_lines_per_batch: int = 500
    max_retries: int = 3
    timeout: int = 600

    def to_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'target_language': self.target_language,
            'use_embedded_subtitle': self.use_embedded_subtitle,
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


@dataclass
class PromptTemplate:
    """翻译提示词模板（三段式结构）"""
    role: str = ''           # 角色定义（你是什么专家）
    rules: str = ''          # 翻译规则（必须遵守的规则）
    style_guide: str = ''    # 风格指导（语气、表达方式等）

    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'rules': self.rules,
            'style_guide': self.style_guide
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptTemplate':
        return cls(
            role=data.get('role', ''),
            rules=data.get('rules', ''),
            style_guide=data.get('style_guide', '')
        )


@dataclass
class SubtitleTrack:
    """字幕轨道信息"""
    stream_index: int = 0     # 流索引（ffmpeg 用）
    codec_name: str = ''      # 编码格式 (srt, ass, ssa, etc.)
    language: str = ''         # 语言代码 (en, zh, etc.)
    title: str = ''           # 轨道标题
    is_soft_subtitle: bool = True  # 是否软字幕（可提取）

    def to_dict(self) -> Dict:
        return {
            'stream_index': self.stream_index,
            'codec_name': self.codec_name,
            'language': self.language,
            'title': self.title,
            'is_soft_subtitle': self.is_soft_subtitle
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleTrack':
        return cls(
            stream_index=data.get('stream_index', 0),
            codec_name=data.get('codec_name', ''),
            language=data.get('language', ''),
            title=data.get('title', ''),
            is_soft_subtitle=data.get('is_soft_subtitle', True)
        )

    @property
    def display_name(self) -> str:
        """获取显示名称"""
        lang_map = {
            'en': 'English',
            'zh': '中文',
            'ja': '日本語',
            'ko': '한국어',
            'fr': 'Français',
            'de': 'Deutsch',
            'es': 'Español',
            'ru': 'Русский',
        }
        lang = lang_map.get(self.language, self.language.upper())
        codec = self.codec_name.upper()
        title = f" ({self.title})" if self.title else ""
        return f"{lang}{title} [{codec}]"


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
    log_history: str = ''
    stage: str = 'pending'              # v1.8.1+: 当前阶段 (download/extract/translate/...)
    stage_progress: Optional[float] = None  # v1.8.1+: 段内进度 0.0-100.0（两位小数）
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'file_path': self.file_path,
            'status': self.status.value if isinstance(self.status, TaskStatus) else self.status,
            'progress': self.progress,
            'log': self.log,
            'log_history': self.log_history,
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
            log_history=data.get('log_history', ''),
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