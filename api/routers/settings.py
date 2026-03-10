#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统设置路由
"""

import concurrent.futures
from typing import Dict, List

import requests
from fastapi import APIRouter

from core.config import AppConfig, ConfigManager, LLM_PROVIDERS
from core.models import (
    ContentType, WhisperConfig, TranslationConfig, ExportConfig,
    ProviderConfig, ISO_LANG_MAP, TARGET_LANG_OPTIONS, WHISPER_SOURCE_LANG_MAP,
)
from database.connection import get_db_connection
from api.schemas import (
    SettingsSchema,
    WhisperConfigSchema,
    TranslationConfigSchema,
    ExportConfigSchema,
    ProviderConfigSchema,
    LLMProviderDefault,
    TestConnectionRequest,
    TestConnectionResponse,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])

_config_manager = ConfigManager(get_db_connection)


def _config_to_schema(config: AppConfig) -> SettingsSchema:
    return SettingsSchema(
        whisper=WhisperConfigSchema(**config.whisper.to_dict()),
        translation=TranslationConfigSchema(
            enabled=config.translation.enabled,
            target_language=config.translation.target_language,
            max_lines_per_batch=config.translation.max_lines_per_batch,
        ),
        export=ExportConfigSchema(formats=config.export.formats),
        content_type=config.content_type.value if isinstance(config.content_type, ContentType) else config.content_type,
        current_provider=config.current_provider,
        provider_configs={
            k: ProviderConfigSchema(**v.to_dict())
            for k, v in config.provider_configs.items()
        },
    )


@router.get("", response_model=SettingsSchema)
def get_settings():
    return _config_to_schema(_config_manager.load())


@router.put("")
def save_settings(body: SettingsSchema):
    config = _config_manager.load()

    config.whisper = WhisperConfig(**body.whisper.model_dump())
    config.translation = TranslationConfig(**body.translation.model_dump())
    config.export = ExportConfig(formats=body.export.formats)
    try:
        config.content_type = ContentType(body.content_type)
    except ValueError:
        config.content_type = ContentType.MOVIE
    config.current_provider = body.current_provider
    config.provider_configs = {
        k: ProviderConfig(**v.model_dump())
        for k, v in body.provider_configs.items()
    }

    saved = _config_manager.save(config)
    return {"saved": saved}


@router.get("/meta")
def get_meta():
    """返回前端需要的枚举/常量数据"""
    content_type_options = [
        {"value": ct.value, "label": _ct_label(ct), "description": _ct_desc(ct)}
        for ct in ContentType
    ]
    return {
        "providers": {
            k: {"base_url": v["base_url"], "model": v["model"], "help": v["help"]}
            for k, v in LLM_PROVIDERS.items()
        },
        "content_types": content_type_options,
        "whisper_source_languages": WHISPER_SOURCE_LANG_MAP,
        "target_languages": {k: ISO_LANG_MAP.get(k, k) for k in TARGET_LANG_OPTIONS},
        "whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "devices": ["cpu", "cuda", "mps"],
        "compute_types": ["int8", "float16"],
        "export_formats": [
            {"value": "srt", "label": "SRT", "desc": "最通用，几乎所有播放器支持"},
            {"value": "vtt", "label": "VTT", "desc": "Web/HTML5 播放器专用"},
            {"value": "ass", "label": "ASS", "desc": "支持丰富样式，动漫字幕常用"},
            {"value": "ssa", "label": "SSA", "desc": "ASS 的前身，兼容性更好"},
            {"value": "sub", "label": "SUB", "desc": "老式 DVD 播放器支持"},
        ],
    }


@router.get("/ollama-models")
def get_ollama_models(base_url: str = "http://ollama:11434/v1"):
    try:
        root_url = base_url.replace("/v1", "").rstrip("/")
        resp = requests.get(f"{root_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"models": models}
    except Exception:
        pass
    return {"models": []}


@router.post("/test-connection", response_model=TestConnectionResponse)
def test_connection(body: TestConnectionRequest):
    def _do_test():
        try:
            from services.translator import TranslationConfig as TC, SubtitleTranslator
            from core.models import SubtitleEntry
            cfg = TC(
                api_key=body.api_key,
                base_url=body.base_url,
                model_name=body.model_name,
                target_language="zh",
            )
            translator = SubtitleTranslator(cfg)
            test_entry = SubtitleEntry("1", "00:00:00,000 --> 00:00:01,000", "Hello")
            translator._translate_batch([test_entry])
            return True, "连接成功"
        except Exception as e:
            return False, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_do_test)
        try:
            ok, msg = future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            ok, msg = False, "连接超时 (10秒)"

    return TestConnectionResponse(ok=ok, message=msg)


def _ct_label(ct: ContentType) -> str:
    labels = {
        ContentType.MOVIE: "电影/剧集（标准）",
        ContentType.DOCUMENTARY: "纪录片/新闻",
        ContentType.VARIETY: "综艺/访谈",
        ContentType.ANIMATION: "动画/动漫",
        ContentType.LECTURE: "讲座/课程",
        ContentType.MUSIC: "音乐视频/MV",
        ContentType.CUSTOM: "自定义",
    }
    return labels.get(ct, ct.value)


def _ct_desc(ct: ContentType) -> str:
    from core.config import CONTENT_TYPE_DESCRIPTIONS
    return CONTENT_TYPE_DESCRIPTIONS.get(ct, "")
