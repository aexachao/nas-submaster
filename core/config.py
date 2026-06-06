#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
负责应用配置的加载、保存和验证
"""

import json
import copy  # ✅ 新增：用于深拷贝配置字典
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict


# 应用版本号（每次发版手动更新）
APP_VERSION = "v1.7.7"

from core.models import (
    ContentType,
    ProviderConfig,
    WhisperConfig,
    TranslationConfig,
    ExportConfig,
    VADParameters,
    PromptTemplate
)


# ============================================================================
# VAD 参数预设
# ============================================================================

VAD_PRESETS = {
    ContentType.MOVIE: VADParameters(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=400
    ),
    ContentType.DOCUMENTARY: VADParameters(
        threshold=0.45,
        min_speech_duration_ms=300,
        min_silence_duration_ms=1800,
        speech_pad_ms=500
    ),
    ContentType.VARIETY: VADParameters(
        threshold=0.6,
        min_speech_duration_ms=200,
        min_silence_duration_ms=2500,
        speech_pad_ms=300
    ),
    ContentType.ANIMATION: VADParameters(
        threshold=0.4,
        min_speech_duration_ms=150,
        min_silence_duration_ms=1500,
        speech_pad_ms=350
    ),
    ContentType.LECTURE: VADParameters(
        threshold=0.5,
        min_speech_duration_ms=400,
        min_silence_duration_ms=2500,
        speech_pad_ms=600
    ),
    ContentType.MUSIC: VADParameters(
        threshold=0.7,
        min_speech_duration_ms=500,
        min_silence_duration_ms=3000,
        speech_pad_ms=200
    ),
    ContentType.CUSTOM: VADParameters(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=400
    )
}

# 内容类型描述
CONTENT_TYPE_DESCRIPTIONS = {
    ContentType.MOVIE: '标准配置，适合电影、电视剧等有明确对话的影视内容。时间轴精准度高。',
    ContentType.DOCUMENTARY: '优化旁白识别，减少背景音乐干扰。适合纪录片、新闻、访谈节目。',
    ContentType.VARIETY: '高阈值过滤笑声、掌声、背景音。适合综艺节目、脱口秀、多人访谈。',
    ContentType.ANIMATION: '适配较快语速，减少卡顿。适合日本动漫、卡通片等快节奏内容。',
    ContentType.LECTURE: '注重完整语句识别，增加停顿缓冲。适合教学视频、演讲、培训课程。',
    ContentType.MUSIC: '极高阈值仅提取人声，忽略背景音乐。适合 MV、音乐会、歌唱节目。',
    ContentType.CUSTOM: '默认配置，也可以手动调整 VAD 参数以满足特殊需求。'
}


# ============================================================================
# 默认翻译提示词模板
# ============================================================================

TRANSLATION_PROMPTS = {
    ContentType.MOVIE: PromptTemplate(
        role="你是一位专业电影字幕翻译专家，擅长翻译电影和电视剧中的对话。",
        rules="1. 保持口语化对话风格，符合日常说话习惯\n2. 传达情感和语气，保留角色的情绪色彩\n3. 人名、地名等专有名词保持原文或使用常见译名\n4. 避免过度翻译，保持原意的同时让观众容易理解",
        style_guide="电影字幕应简洁，通常不超过15个字一行。注意断句的自然性，保留对话的节奏感和戏剧张力。"
    ),
    ContentType.DOCUMENTARY: PromptTemplate(
        role="你是一位专业纪录片字幕翻译专家，擅长翻译纪录片和新闻节目的旁白。",
        rules="1. 保持说明性语言的严谨性和准确性\n2. 准确翻译专业术语，必要时可在括号中标注原文\n3. 保持逻辑清晰，句式结构完整\n4. 注意中英文术语的使用习惯差异",
        style_guide="纪录片字幕应清晰准确，适合观众阅读。保持专业但不失流畅，让观众能够快速理解内容。"
    ),
    ContentType.VARIETY: PromptTemplate(
        role="你是一位专业综艺节目字幕翻译专家，擅长翻译综艺节目、脱口秀和访谈节目。",
        rules="1. 口语化翻译，符合综艺节目的轻松氛围\n2. 保留节目中的梗、双关语和幽默元素，尽量本地化\n3. 方言或口语表达尽量转化为目标语言中对应的表达\n4. 笑声、掌声等音效标注可用括号说明",
        style_guide="综艺字幕应活泼有趣，适合年轻观众群体。翻译可以更加自由，发挥创意让观众获得与原版相近的笑点。"
    ),
    ContentType.ANIMATION: PromptTemplate(
        role="你是一位专业动画字幕翻译专家，擅长翻译动画、动漫和卡通片。",
        rules="1. 保持动画角色的性格特点和说话风格\n2. 拟声词、感叹词等要符合动画的夸张风格\n3. 童趣表达要适合目标语言的儿童观众\n4. 保持对话的简洁和趣味性",
        style_guide="动画字幕应活泼可爱，适合儿童和家庭观看。翻译时可以更加夸张和有趣，保持动画的欢乐氛围。"
    ),
    ContentType.LECTURE: PromptTemplate(
        role="你是一位专业学术讲座字幕翻译专家，擅长翻译讲座、课程和演讲。",
        rules="1. 保持学术语言的严谨性和专业性\n2. 准确翻译专业术语，可保留英文原文供参考\n3. 保持逻辑连贯，句式结构清晰\n4. 演讲中的重复和强调部分可适当精简",
        style_guide="讲座字幕应清晰有条理，方便观众做笔记。句式可以稍长但要保持完整，适合学习者反复观看理解。"
    ),
    ContentType.MUSIC: PromptTemplate(
        role="你是一位专业歌词翻译专家，擅长翻译歌曲、MV和音乐节目。",
        rules="1. 尽量保持歌词的韵律和节奏感\n2. 意译优先，传达歌词的情感和意境\n3. 歌曲中的重复部分可适当精简\n4. 保留歌手特有的表达风格",
        style_guide="歌词字幕应优美流畅，尽量保留原词的韵脚和节奏感。翻译时可以调整语序以适应目标语言的表达习惯。"
    ),
    ContentType.CUSTOM: PromptTemplate(
        role="你是一位专业字幕翻译专家。",
        rules="1. 翻译应准确传达原意\n2. 保持语言自然流畅\n3. 符合目标语言的口语或书面习惯",
        style_guide="字幕翻译应根据具体内容类型调整风格，保持一致性和可读性。"
    )
}


# ============================================================================
# 模型推荐批处理行数（根据上下文窗口大小）
# ============================================================================

MODEL_BATCH_SIZES = {
    # DeepSeek
    "deepseek-v4-flash": 800,
    "deepseek-v4-pro": 1000,
    "deepseek-chat": 500,
    "deepseek-reasoner": 500,
    # Google Gemini
    "gemini-2.5-flash": 1500,
    "gemini-2.5-pro": 1500,
    "gemini-3-flash": 1500,
    "gemini-3.1-pro": 1500,
    # Moonshot / Kimi
    "kimi-k2.6": 800,
    "kimi-k2.5": 800,
    "kimi-k2-thinking": 800,
    "moonshot-v1-128k": 800,
    "moonshot-v1-32k": 300,
    "moonshot-v1-8k": 100,
    # Aliyun / Qwen
    "qwen3.7-max": 800,
    "qwen3.7-plus": 800,
    "qwen3.6-flash": 800,
    # ZhipuAI / GLM
    "GLM-5.1": 800,
    "GLM-5": 800,
    "GLM-4.7": 800,
    "GLM-4.7-Flash": 800,
    "GLM-4.5-Air": 800,
    "GLM-4-Long": 1500,
    # OpenAI
    "gpt-5.5": 1000,
    "gpt-5.4": 800,
    "gpt-5.4-mini": 800,
    "gpt-4o": 800,
    "gpt-4o-mini": 800,
}

DEFAULT_BATCH_SIZE = 500


def get_recommended_batch_size(model_name: str) -> int:
    """获取模型的推荐批处理行数，未知模型返回默认值"""
    return MODEL_BATCH_SIZES.get(model_name, DEFAULT_BATCH_SIZE)


# ============================================================================
# LLM 提供商配置
# ============================================================================

LLM_PROVIDERS = {
    "DeepSeek (深度求索)": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "models": [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "help": "国内推荐，性价比高"
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash",
            "gemini-3.1-pro",
        ],
        "help": "速度极快"
    },
    "Moonshot (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5",
        "models": [
            "kimi-k2.6",
            "kimi-k2.5",
            "kimi-k2-thinking",
            "moonshot-v1-128k",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
        ],
        "help": "长文本优化"
    },
    "Aliyun (通义千问)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
        "models": [
            "qwen3.7-max",
            "qwen3.7-plus",
            "qwen3.6-flash",
        ],
        "help": "阿里官方"
    },
    "ZhipuAI (智谱GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "GLM-4.7-Flash",
        "models": [
            "GLM-5.1",
            "GLM-5",
            "GLM-4.7",
            "GLM-4.7-Flash",
            "GLM-4.5-Air",
            "GLM-4-Long",
        ],
        "help": "智谱清言"
    },
    "OpenAI (官方)": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "models": [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "help": "需科学上网"
    },
    "Ollama (本地模型)": {
        "base_url": "http://ollama:11434/v1",
        "model": "qwen2.5:7b",
        "models": [],
        "help": "无需联网，使用本地算力，质量取决于本地模型"
    },
    "自定义 (Custom)": {
        "base_url": "",
        "model": "",
        "models": [],
        "help": "手动填写"
    }
}


# ============================================================================
# 应用配置类
# ============================================================================

@dataclass
class AppConfig:
    """应用配置（主配置类）"""
    
    # Whisper 配置
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    
    # 翻译配置
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    
    # 导出配置
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # 内容类型
    content_type: ContentType = ContentType.MOVIE
    
    # 当前使用的 LLM 提供商
    current_provider: str = 'DeepSeek (深度求索)'
    
    # 各提供商的配置
    provider_configs: Dict[str, ProviderConfig] = field(default_factory=dict)

    # 各内容类型的翻译提示词模板
    prompt_templates: Dict[ContentType, PromptTemplate] = field(default_factory=dict)

    # 自动扫描配置
    auto_scan_enabled: bool = True
    auto_scan_interval_minutes: int = 30

    # 自动更新配置
    auto_update_enabled: bool = False

    def get_vad_parameters(self) -> VADParameters:
        """获取当前内容类型的 VAD 参数"""
        return VAD_PRESETS.get(self.content_type, VAD_PRESETS[ContentType.MOVIE])

    def get_prompt_template(self, content_type: ContentType = None) -> PromptTemplate:
        """获取指定内容类型的翻译提示词模板"""
        if content_type is None:
            content_type = self.content_type
        # 用户已自定义模板则使用用户的，否则使用默认模板
        if content_type in self.prompt_templates:
            return self.prompt_templates[content_type]
        return TRANSLATION_PROMPTS.get(content_type, TRANSLATION_PROMPTS[ContentType.CUSTOM])

    def get_current_provider_config(self) -> ProviderConfig:
        """获取当前提供商的配置"""
        if self.current_provider not in self.provider_configs:
            default = LLM_PROVIDERS.get(self.current_provider, {})
            return ProviderConfig(
                api_key='',
                base_url=default.get('base_url', ''),
                model_name=default.get('model', '')
            )
        return self.provider_configs[self.current_provider]
    
    def update_provider_config(
        self, 
        provider: str, 
        api_key: str, 
        base_url: str, 
        model_name: str
    ):
        """更新指定提供商的配置"""
        self.provider_configs[provider] = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
        self.current_provider = provider
    
    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        return {
            'whisper': self.whisper.to_dict(),
            'translation': self.translation.to_dict(),
            'export': self.export.to_dict(),
            'content_type': self.content_type.value if isinstance(self.content_type, ContentType) else self.content_type,
            'current_provider': self.current_provider,
            'provider_configs': {
                k: v.to_dict() for k, v in self.provider_configs.items()
            },
            'prompt_templates': {
                k.value: v.to_dict() for k, v in self.prompt_templates.items()
            },
            'auto_scan_enabled': self.auto_scan_enabled,
            'auto_scan_interval_minutes': self.auto_scan_interval_minutes,
            'auto_update_enabled': self.auto_update_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AppConfig':
        """从字典创建配置对象"""
        # 解析 Whisper 配置
        whisper_data = data.get('whisper', {})
        whisper = WhisperConfig(**whisper_data)
        
        # 解析翻译配置
        translation_data = data.get('translation', {})
        translation = TranslationConfig(**translation_data)
        
        # 解析导出配置
        export_data = data.get('export', {'formats': ['srt']})
        export = ExportConfig.from_dict(export_data)
        
        # 解析内容类型
        content_type_str = data.get('content_type', 'movie')
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.MOVIE
        
        # 解析提供商配置
        provider_configs_data = data.get('provider_configs', {})
        provider_configs = {
            k: ProviderConfig.from_dict(v)
            for k, v in provider_configs_data.items()
        }

        # 解析提示词模板配置
        prompt_templates_data = data.get('prompt_templates', {})
        prompt_templates = {}
        for k, v in prompt_templates_data.items():
            try:
                ct = ContentType(k)
                prompt_templates[ct] = PromptTemplate.from_dict(v)
            except ValueError:
                continue

        return cls(
            whisper=whisper,
            translation=translation,
            export=export,
            content_type=content_type,
            current_provider=data.get('current_provider', 'DeepSeek (深度求索)'),
            provider_configs=provider_configs,
            prompt_templates=prompt_templates,
            auto_scan_enabled=data.get('auto_scan_enabled', True),
            auto_scan_interval_minutes=data.get('auto_scan_interval_minutes', 30),
            auto_update_enabled=data.get('auto_update_enabled', False)
        )


# ============================================================================
# 配置持久化（与数据库交互）
# ============================================================================

class ConfigManager:
    """配置管理器（负责配置的加载和保存）"""
    
    def __init__(self, db_connection):
        """
        初始化配置管理器
        
        Args:
            db_connection: 数据库连接工厂函数
        """
        self.get_db = db_connection
        self._last_saved_config_dict = {}  # ✅ 新增：缓存上一次保存或加载的配置
    
    def load(self) -> AppConfig:
        """从数据库加载配置"""
        conn = self.get_db()
        try:
            cursor = conn.execute("SELECT key, value FROM config")
            config_dict = {row[0]: row[1] for row in cursor.fetchall()}
            
            if not config_dict:
                # ✅ 修改：初始化默认配置时也记录缓存
                default_config = AppConfig()
                self._last_saved_config_dict = default_config.to_dict()
                return default_config
            
            # 构建嵌套配置字典
            data = {
                'whisper': {
                    'model_size': config_dict.get('whisper_model', 'base'),
                    'compute_type': config_dict.get('compute_type', 'int8'),
                    'device': config_dict.get('device', 'cpu'),
                    'source_language': config_dict.get('source_language', 'auto')
                },
                'translation': {
                    'enabled': config_dict.get('enable_translation', 'false') == 'true',
                    'target_language': config_dict.get('target_language', 'zh'),
                    'max_lines_per_batch': int(config_dict.get('max_lines_per_batch', 500)),
                    'timeout': int(config_dict.get('timeout', 600))
                },
                'export': json.loads(config_dict.get('export_formats', '{"formats": ["srt"]}')),
                'content_type': config_dict.get('content_type', 'movie'),
                'current_provider': config_dict.get('current_provider', 'DeepSeek (深度求索)'),
                'provider_configs': json.loads(config_dict.get('provider_configs', '{}')),
                'prompt_templates': json.loads(config_dict.get('prompt_templates', '{}')),
                'auto_scan_enabled': config_dict.get('auto_scan_enabled', 'true') == 'true',
                'auto_scan_interval_minutes': int(config_dict.get('auto_scan_interval_minutes', 30)),
                'auto_update_enabled': config_dict.get('auto_update_enabled', 'false') == 'true'
            }
            
            # ✅ 修改：加载完成后更新缓存
            loaded_config = AppConfig.from_dict(data)
            self._last_saved_config_dict = loaded_config.to_dict()
            return loaded_config
            
        finally:
            conn.close()
    
    def save(self, config: AppConfig) -> bool:
        """
        保存配置到数据库
        
        Returns:
            bool: True 表示实际执行了保存，False 表示未变更无需保存
        """
        # ✅ 新增：获取新配置的字典形式并比对
        new_config_dict = config.to_dict()
        
        if new_config_dict == self._last_saved_config_dict:
            # 如果配置内容完全一致，跳过数据库操作
            return False

        conn = self.get_db()
        try:
            # 扁平化配置
            flat_config = {
                'whisper_model': config.whisper.model_size,
                'compute_type': config.whisper.compute_type,
                'device': config.whisper.device,
                'source_language': config.whisper.source_language,
                'enable_translation': 'true' if config.translation.enabled else 'false',
                'target_language': config.translation.target_language,
                'max_lines_per_batch': str(config.translation.max_lines_per_batch),
                'timeout': str(config.translation.timeout),
                'export_formats': json.dumps(config.export.to_dict(), ensure_ascii=False),
                'content_type': config.content_type.value if isinstance(config.content_type, ContentType) else config.content_type,
                'current_provider': config.current_provider,
                'provider_configs': json.dumps(
                    {k: v.to_dict() for k, v in config.provider_configs.items()},
                    ensure_ascii=False
                ),
                'prompt_templates': json.dumps(
                    {k.value: v.to_dict() for k, v in config.prompt_templates.items()},
                    ensure_ascii=False
                ),
                'auto_scan_enabled': 'true' if config.auto_scan_enabled else 'false',
                'auto_scan_interval_minutes': str(config.auto_scan_interval_minutes),
                'auto_update_enabled': 'true' if config.auto_update_enabled else 'false'
            }
            
            for key, value in flat_config.items():
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    (key, str(value))
                )
            
            conn.commit()
            
            # ✅ 新增：保存成功后更新缓存
            self._last_saved_config_dict = copy.deepcopy(new_config_dict)
            return True
            
        except Exception as e:
            print(f"Failed to save config: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()


# ============================================================================
# 辅助函数
# ============================================================================

def get_content_type_display_name(content_type: ContentType) -> str:
    """获取内容类型的显示名称"""
    display_names = {
        ContentType.MOVIE: '🎬 电影/剧集（标准）',
        ContentType.DOCUMENTARY: '📺 纪录片/新闻',
        ContentType.VARIETY: '🎤 综艺/访谈',
        ContentType.ANIMATION: '🎨 动画/动漫',
        ContentType.LECTURE: '🎓 讲座/课程',
        ContentType.MUSIC: '🎵 音乐视频/MV',
        ContentType.CUSTOM: '⚙️ 自定义'
    }
    return display_names.get(content_type, content_type.value)


def get_content_type_description(content_type: ContentType) -> str:
    """获取内容类型的详细描述"""
    return CONTENT_TYPE_DESCRIPTIONS.get(
        content_type, 
        '默认配置'
    )