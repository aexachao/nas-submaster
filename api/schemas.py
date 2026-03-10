#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 请求/响应 Pydantic 模型
"""

from typing import List, Optional, Dict
from pydantic import BaseModel


# ============================================================================
# 媒体库
# ============================================================================

class SubtitleInfoSchema(BaseModel):
    path: str
    lang: str
    tag: str


class MediaFileSchema(BaseModel):
    id: int
    file_path: str
    file_name: str
    file_size: int
    file_size_display: str
    subtitles: List[SubtitleInfoSchema]
    has_subtitle: bool
    has_translated: bool
    updated_at: Optional[str] = None


class MediaListResponse(BaseModel):
    items: List[MediaFileSchema]
    total: int
    page: int
    page_size: int
    total_pages: int


class ScanRequest(BaseModel):
    dirs: List[str] = []


class ScanResponse(BaseModel):
    updated: int


# ============================================================================
# 任务队列
# ============================================================================

class TaskSchema(BaseModel):
    id: int
    file_path: str
    file_name: str
    status: str
    progress: int
    log: str
    log_history: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AddTasksRequest(BaseModel):
    file_paths: List[str]


class AddTasksResponse(BaseModel):
    success: int
    failed: List[Dict[str, str]]


# ============================================================================
# 设置
# ============================================================================

class ProviderConfigSchema(BaseModel):
    api_key: str = ''
    base_url: str = ''
    model_name: str = ''


class WhisperConfigSchema(BaseModel):
    model_size: str = 'base'
    compute_type: str = 'int8'
    device: str = 'cpu'
    source_language: str = 'auto'


class TranslationConfigSchema(BaseModel):
    enabled: bool = False
    target_language: str = 'zh'
    max_lines_per_batch: int = 500


class ExportConfigSchema(BaseModel):
    formats: List[str] = ['srt']


class SettingsSchema(BaseModel):
    whisper: WhisperConfigSchema
    translation: TranslationConfigSchema
    export: ExportConfigSchema
    content_type: str = 'movie'
    current_provider: str = 'Ollama (本地模型)'
    provider_configs: Dict[str, ProviderConfigSchema] = {}


class LLMProviderDefault(BaseModel):
    base_url: str
    model: str
    help: str


class TestConnectionRequest(BaseModel):
    api_key: str
    base_url: str
    model_name: str


class TestConnectionResponse(BaseModel):
    ok: bool
    message: str
