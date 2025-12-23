#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAS 字幕管家 (重构版) V7.0.1
主要改进：
1. 翻译模块独立化（translator.py）
2. 使用 JSON 格式强制结构化输出
3. 智能分段翻译策略
4. 完善错误处理和进度反馈
"""

import os
import sqlite3
import threading
import time
import json
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import streamlit as st
import pandas as pd
from faster_whisper import WhisperModel
import logging

# 导入新的翻译模块（延迟导入，避免循环依赖）
try:
    from translator import (
        SubtitleTranslator,
        TranslationConfig,
        parse_srt_file,
        save_srt_file,
        translate_srt_file
    )
    HAS_TRANSLATOR = True
except ImportError as e:
    print(f"[警告] 翻译模块未找到: {e}")
    HAS_TRANSLATOR = False

# 抑制 Tornado WebSocket 关闭警告
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.access').setLevel(logging.ERROR)

# ============================================================================
# 常量定义
# ============================================================================
DB_PATH = "/data/subtitle_manager.db"
MEDIA_ROOT = "/media"
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.m4v', '.webm', '.ts'}

ISO_LANG_MAP = {
    'auto': '自动检测',
    'zh': '中文', 'en': '英语', 'ja': '日语', 'ko': '韩语',
    'fr': '法语', 'de': '德语', 'ru': '俄语', 'es': '西班牙语',
    'chs': '简中', 'cht': '繁中', 'eng': '英语', 'jpn': '日语', 'kor': '韩语',
    'unknown': '未知'
}

TARGET_LANG_OPTIONS = ['zh', 'en', 'ja', 'ko']

LLM_PROVIDERS = {
    "Ollama (本地模型)": {
        "base_url": "http://ollama:11434/v1", 
        "model": "qwen2.5:7b", 
        "help": "无需联网，使用本地算力"
    },
    "DeepSeek (深度求索)": {
        "base_url": "https://api.deepseek.com", 
        "model": "deepseek-chat", 
        "help": "国内推荐"
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", 
        "model": "gemini-1.5-flash", 
        "help": "速度极快"
    },
    "Moonshot (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1", 
        "model": "moonshot-v1-8k", 
        "help": "长文本优化"
    },
    "Aliyun (通义千问)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
        "model": "qwen-turbo", 
        "help": "阿里官方"
    },
    "ZhipuAI (智谱GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4", 
        "model": "glm-4-flash", 
        "help": "智谱清言"
    },
    "OpenAI (官方)": {
        "base_url": "https://api.openai.com/v1", 
        "model": "gpt-4o-mini", 
        "help": "需科学上网"
    },
    "自定义 (Custom)": {
        "base_url": "", 
        "model": "", 
        "help": "手动填写"
    }
}

HERO_CSS = """
<style>
    .stApp {
        background-color: #09090b;
        color: #e4e4e7;
    }
    
    h1 { font-size: 32px !important; font-weight: 700 !important; padding-bottom: 0.5rem; }
    h2, h3 { font-size: 16px !important; font-weight: 600 !important; }
    
    section[data-testid="stSidebar"] {
        background-color: #111114;
        border-right: 1px solid #27272a;
    }

    .hero-card {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 12px 16px;
        transition: border-color 0.2s;
        margin-bottom: 16px;
    }
    .hero-card:hover {
        border-color: #3f3f46;
    }
    
    .status-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 6px;
    }
    .chip-gray { background: #27272a; color: #a1a1aa; border: 1px solid #3f3f46; }
    .chip-blue { background: #172554; color: #60a5fa; border: 1px solid #1e3a8a; }
    .chip-green { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
    .chip-red { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }

    .stButton button {
        background-color: transparent !important;
        border: 1px solid #3f3f46 !important;
        color: #d4d4d8 !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        height: 32px !important;
        padding: 0 12px !important;
    }
    .stButton button:hover {
        border-color: #71717a !important;
        background-color: #27272a !important;
        color: #fff !important;
    }
    
    div[data-testid="stVerticalBlock"] button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #2563eb !important;
        color: white !important;
    }
    div[data-testid="stVerticalBlock"] button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    .stProgress > div > div > div > div { background-color: #2563eb; }
    
    div[data-testid="stCheckbox"] label {
        min-height: 0px !important;
        margin-bottom: 0px !important;
    }
    
    .task-card-wrapper {
        position: relative;
        margin-bottom: 24px;
    }
    
    .task-card-wrapper + div[data-testid="column"] {
        margin-top: -48px !important;
        margin-bottom: 12px !important;
        padding-right: 16px !important;
    }
</style>
"""

# ============================================================================
# 数据模型
# ============================================================================
class TaskStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

@dataclass
class ProviderConfig:
    """单个提供商的配置"""
    api_key: str = ''
    base_url: str = ''
    model_name: str = ''

@dataclass
class AppConfig:
    whisper_model: str = 'base'
    compute_type: str = 'int8'
    device: str = 'cpu'
    source_language: str = 'auto'
    enable_translation: bool = False
    target_language: str = 'zh'
    current_provider: str = 'Ollama (本地模型)'
    provider_configs: Dict[str, ProviderConfig] = None
    max_lines_per_batch: int = 500  # 每批最多翻译行数
    content_type: str = 'movie'  # 新增：内容类型（影响 VAD 参数）
    export_formats: List[str] = None 
    
    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {}
        # 👇👇👇 建议也加上这一行，防止为 None 👇👇👇
        if self.export_formats is None:
            self.export_formats = ['srt']
    
    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {}
    
    def get_vad_parameters(self) -> dict:
        """根据内容类型返回 VAD 参数"""
        vad_presets = {
            'movie': {
                'name': '电影/剧集（标准）',
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 2000,
                'speech_pad_ms': 400,
                'description': '适合有明确对话的影视内容，时间轴精准'
            },
            'documentary': {
                'name': '纪录片/新闻',
                'threshold': 0.45,
                'min_speech_duration_ms': 300,
                'min_silence_duration_ms': 1800,
                'speech_pad_ms': 500,
                'description': '适合旁白较多的内容，减少背景音干扰'
            },
            'variety': {
                'name': '综艺/访谈',
                'threshold': 0.6,
                'min_speech_duration_ms': 200,
                'min_silence_duration_ms': 2500,
                'speech_pad_ms': 300,
                'description': '过滤笑声/掌声等噪音，适合多人对话'
            },
            'animation': {
                'name': '动画/动漫',
                'threshold': 0.4,
                'min_speech_duration_ms': 150,
                'min_silence_duration_ms': 1500,
                'speech_pad_ms': 350,
                'description': '适合语速较快的动画内容'
            },
            'lecture': {
                'name': '讲座/课程',
                'threshold': 0.5,
                'min_speech_duration_ms': 400,
                'min_silence_duration_ms': 2500,
                'speech_pad_ms': 600,
                'description': '适合单人演讲，注重完整语句'
            },
            'music': {
                'name': '音乐视频/MV',
                'threshold': 0.7,
                'min_speech_duration_ms': 500,
                'min_silence_duration_ms': 3000,
                'speech_pad_ms': 200,
                'description': '高阈值过滤背景音乐，仅提取歌词/对话'
            },
            'custom': {
                'name': '自定义',
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 2000,
                'speech_pad_ms': 400,
                'description': '默认配置，可在高级选项中调整'
            }
        }
        
        preset = vad_presets.get(self.content_type, vad_presets['movie'])
        return {
            'threshold': preset['threshold'],
            'min_speech_duration_ms': preset['min_speech_duration_ms'],
            'min_silence_duration_ms': preset['min_silence_duration_ms'],
            'speech_pad_ms': preset['speech_pad_ms'],
        }
    
    @staticmethod
    def get_content_type_options() -> Dict[str, str]:
        """获取内容类型选项（key: 内部名称, value: 显示名称）"""
        return {
            'movie': '🎬 电影/剧集（标准）',
            'documentary': '📺 纪录片/新闻',
            'variety': '🎤 综艺/访谈',
            'animation': '🎨 动画/动漫',
            'lecture': '🎓 讲座/课程',
            'music': '🎵 音乐视频/MV',
            'custom': '⚙️ 自定义'
        }
    
    @staticmethod
    def get_content_type_description(content_type: str) -> str:
        """获取内容类型的详细说明"""
        descriptions = {
            'movie': '标准配置，适合电影、电视剧等有明确对话的影视内容。时间轴精准度高。',
            'documentary': '优化旁白识别，减少背景音乐干扰。适合纪录片、新闻、访谈节目。',
            'variety': '高阈值过滤笑声、掌声、背景音。适合综艺节目、脱口秀、多人访谈。',
            'animation': '适配较快语速，减少停顿。适合日本动漫、卡通片等快节奏内容。',
            'lecture': '注重完整语句识别，增加停顿缓冲。适合教学视频、演讲、培训课程。',
            'music': '极高阈值仅提取人声，忽略背景音乐。适合 MV、音乐会、歌唱节目。',
            'custom': '默认配置，也可以手动调整 VAD 参数以满足特殊需求。'
        }
        return descriptions.get(content_type, '')
    
    @classmethod
    def load_from_db(cls) -> 'AppConfig':
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT key, value FROM config")
            config_dict = {row[0]: row[1] for row in cursor.fetchall()}
            
            provider_configs_json = config_dict.get('provider_configs', '{}')
            try:
                provider_configs_data = json.loads(provider_configs_json)
                provider_configs = {
                    k: ProviderConfig(**v) for k, v in provider_configs_data.items()
                }
            except:
                provider_configs = {}
            
            # 加载导出格式
            export_formats_json = config_dict.get('export_formats', '["srt"]')
            try:
                export_formats = json.loads(export_formats_json)
            except:
                export_formats = ['srt']
            
            return cls(
                whisper_model=config_dict.get('whisper_model', 'base'),
                compute_type=config_dict.get('compute_type', 'int8'),
                device=config_dict.get('device', 'cpu'),
                source_language=config_dict.get('source_language', 'auto'),
                enable_translation=config_dict.get('enable_translation', 'false') == 'true',
                target_language=config_dict.get('target_language', 'zh'),
                current_provider=config_dict.get('current_provider', 'Ollama (本地模型)'),
                provider_configs=provider_configs,
                max_lines_per_batch=int(config_dict.get('max_lines_per_batch', 500)),
                content_type=config_dict.get('content_type', 'movie'),
                export_formats=export_formats
            )
        finally:
            conn.close()
    
    def save_to_db(self):
        conn = get_db_connection()
        try:
            basic_config = {
                'whisper_model': self.whisper_model,
                'compute_type': self.compute_type,
                'device': self.device,
                'source_language': self.source_language,
                'enable_translation': 'true' if self.enable_translation else 'false',
                'target_language': self.target_language,
                'current_provider': self.current_provider,
                'max_lines_per_batch': str(self.max_lines_per_batch),
                'content_type': self.content_type,
                'export_formats': json.dumps(self.export_formats)
            }
            
            provider_configs_data = {
                k: asdict(v) for k, v in self.provider_configs.items()
            }
            basic_config['provider_configs'] = json.dumps(provider_configs_data, ensure_ascii=False)
            
            for key, value in basic_config.items():
                conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))
            conn.commit()
        except Exception as e:
            print(f"Failed to save config: {e}")
            conn.rollback()
        finally:
            conn.close()
    
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
    
    def update_provider_config(self, provider: str, api_key: str, base_url: str, model_name: str):
        """更新指定提供商的配置"""
        self.provider_configs[provider] = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
        self.current_provider = provider

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_database():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                subtitles_json TEXT DEFAULT '[]',
                has_translated INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                log TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Database init failed: {e}")
        conn.rollback()
    finally:
        conn.close()

class TaskDAO:
    @staticmethod
    def add_task(file_path: str) -> Tuple[bool, str]:
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO tasks (file_path, status, log) VALUES (?, 'pending', '准备中')", (file_path,))
            conn.commit()
            return True, "任务已添加"
        except sqlite3.IntegrityError:
            return False, "任务已存在"
        except Exception as e:
            print(f"Failed to add task: {e}")
            return False, f"添加失败: {str(e)}"
        finally:
            conn.close()
    
    @staticmethod
    def get_all_tasks() -> List[Dict]:
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT id, file_path, status, progress, log, created_at FROM tasks ORDER BY id DESC")
            return [{'id': r[0], 'file_path': r[1], 'status': r[2], 'progress': r[3], 'log': r[4], 'created_at': r[5]} for r in cursor.fetchall()]
        finally:
            conn.close()
    
    @staticmethod
    def get_pending_task() -> Optional[Dict]:
        conn = get_db_connection()
        try:
            result = conn.execute("SELECT id, file_path FROM tasks WHERE status='pending' LIMIT 1").fetchone()
            return {'id': result[0], 'file_path': result[1]} if result else None
        finally:
            conn.close()
    
    @staticmethod
    def update_task(task_id: int, status=None, progress=None, log=None):
        conn = get_db_connection()
        try:
            updates, params = [], []
            if status:
                updates.append("status=?")
                params.append(status)
            if progress is not None:
                updates.append("progress=?")
                params.append(progress)
            if log:
                updates.append("log=?")
                params.append(log)
            if updates:
                updates.append("updated_at=CURRENT_TIMESTAMP")
                params.append(task_id)
                conn.execute(f"UPDATE tasks SET {','.join(updates)} WHERE id=?", params)
                conn.commit()
        except Exception as e:
            print(f"Failed to update task {task_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def delete_task(task_id: int):
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
        finally:
            conn.close()
    
    @staticmethod
    def clear_completed_tasks():
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM tasks WHERE status IN ('completed', 'failed')")
            conn.commit()
        finally:
            conn.close()

class MediaDAO:
    @staticmethod
    def get_media_files(filter_type: str = "all") -> List[Dict]:
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT id, file_path, file_name, file_size, subtitles_json, has_translated FROM media_files ORDER BY file_name")
            result = []
            for row in cursor.fetchall():
                subtitles = json.loads(row[4])
                has_subtitle = len(subtitles) > 0
                media = {'id': row[0], 'file_path': row[1], 'file_name': row[2], 'file_size': row[3], 'subtitles': subtitles, 'has_subtitle': has_subtitle, 'has_translated': row[5]}
                if filter_type == "no_subtitle" and has_subtitle:
                    continue
                if filter_type == "has_subtitle" and not has_subtitle:
                    continue
                result.append(media)
            return result
        finally:
            conn.close()
    
    @staticmethod
    def update_media_subtitles(file_path: str, subtitles_json: str, has_translated: bool):
        conn = get_db_connection()
        try:
            conn.execute("UPDATE media_files SET subtitles_json=?, has_translated=?, updated_at=CURRENT_TIMESTAMP WHERE file_path=?",
                        (subtitles_json, 1 if has_translated else 0, file_path))
            conn.commit()
        except Exception as e:
            print(f"Failed to update media subtitles: {e}")
            conn.rollback()
        finally:
            conn.close()

# ============================================================================
# 辅助函数（保持原有功能）
# ============================================================================
def get_lang_name(code: str) -> str:
    return ISO_LANG_MAP.get(code.lower(), code)

def format_timestamp(seconds: float) -> str:
    h, m, s, ms = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60), int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def detect_lang_by_content(srt_path: str) -> str:
    try:
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read(4096)
        content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', '', raw)
        content = re.sub(r'^\d+$', '', content, flags=re.MULTILINE)
        total = len(re.sub(r'\s+', '', content))
        if total < 50:
            return 'unknown'
        cn = len(re.findall(r'[\u4e00-\u9fa5]', content))
        hira = len(re.findall(r'[\u3040-\u309f]', content))
        kata = len(re.findall(r'[\u30a0-\u30ff]', content))
        hang = len(re.findall(r'[\uac00-\ud7af]', content))
        trad_m = ['臺', '灣', '繁', '體', '於', '與', '個', '們', '裡', '這', '妳', '臉', '廳', '學', '習']
        trad = sum(1 for c in trad_m if c in content)
        eng_w = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        eng_c = sum(len(w) for w in eng_w)
        if hira >= 5 or kata >= 5:
            return 'ja'
        if hang >= 10:
            return 'ko'
        if cn >= 10:
            if trad >= 3 and trad / cn >= 0.2:
                return 'cht'
            return 'chs'
        if total > 0 and eng_c / total >= 0.5:
            return 'en'
        return 'unknown'
    except Exception as e:
        print(f"Language detection failed for {srt_path}: {e}")
        return 'unknown'

def scan_file_subtitles(video_path: Path) -> str:
    subs_list, base_name, parent_dir = [], video_path.stem, video_path.parent
    try:
        all_files = list(parent_dir.iterdir())
        potential_subs = [p for p in all_files if p.is_file() and p.name.lower().endswith('.srt') and p.name.lower().startswith(base_name.lower())]
        for sub_path in potential_subs:
            sub_name, lang_code, tag = sub_path.name, 'unknown', '未知'
            suffix_part, detected = sub_name[len(base_name):].lower(), False
            for code in ['chs', 'cht', 'eng', 'jpn', 'kor']:
                if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                    lang_code, tag, detected = code, ISO_LANG_MAP[code], True
                    break
            if not detected:
                for code in ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'ru', 'es']:
                    if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                        lang_code, tag, detected = code, ISO_LANG_MAP[code], True
                        break
            if not detected:
                detected_lang = detect_lang_by_content(str(sub_path))
                if detected_lang in ISO_LANG_MAP:
                    lang_code, tag = detected_lang, ISO_LANG_MAP[detected_lang]
            if sub_path.stem.lower() == base_name.lower():
                tag += " (默认)"
            subs_list.append({"path": str(sub_path), "lang": lang_code, "tag": tag})
    except Exception as e:
        print(f"Failed to scan subtitles for {video_path}: {e}")
    return json.dumps(subs_list, ensure_ascii=False)

def scan_media_directory(directory: str = MEDIA_ROOT, debug: bool = False) -> Tuple[int, List[str]]:
    conn = get_db_connection()
    added, debug_logs, path = 0, [], Path(directory)
    if not path.exists():
        conn.close()
        return 0, ["路径不存在"]
    batch_data = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    try:
                        subs = scan_file_subtitles(file_path)
                        has_trans = 1 if ".zh.srt" in subs or ".chs.srt" in subs else 0
                        batch_data.append((str(file_path), file, file_path.stat().st_size, subs, has_trans))
                        added += 1
                        if debug:
                            debug_logs.append(f"发现: {file}")
                    except Exception as e:
                        if debug:
                            debug_logs.append(f"错误 {file}: {e}")
        if batch_data:
            conn.executemany("INSERT OR REPLACE INTO media_files (file_path, file_name, file_size, subtitles_json, has_translated, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", batch_data)
            conn.commit()
    except Exception as e:
        print(f"Scan failed: {e}")
        if debug:
            debug_logs.append(f"DB错误: {e}")
        conn.rollback()
    finally:
        conn.close()
    return added, debug_logs

def fetch_ollama_models(base_url_v1: str) -> List[str]:
    try:
        root_url = base_url_v1.replace("/v1", "").rstrip("/")
        resp = requests.get(f"{root_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return [m['name'] for m in resp.json().get('models', [])]
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
    return []

def test_api_connection(api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
    """测试 API 连接（兼容新旧版本）"""
    if not HAS_TRANSLATOR:
        # 降级到旧版测试方法
        from openai import OpenAI
        try:
            if "ollama" in base_url.lower() or "host.docker.internal" in base_url:
                api_key = "ollama"
            client = OpenAI(api_key=api_key, base_url=base_url)
            client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": "Hi"}], 
                max_tokens=1, 
                timeout=10
            )
            return True, "连接成功"
        except Exception as e:
            return False, str(e)
    
    # 新版测试方法
    try:
        config = TranslationConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            target_language='zh'
        )
        translator = SubtitleTranslator(config)
        # 简单测试：翻译一条假字幕
        from translator import SubtitleEntry
        test_entry = SubtitleEntry("1", "00:00:00,000 --> 00:00:01,000", "Hello")
        translator._translate_batch([test_entry])
        return True, "连接成功"
    except Exception as e:
        return False, str(e)

# ============================================================================
# 核心处理函数（使用新的翻译模块）
# ============================================================================
def process_video_file(task_id: int, file_path: str, config: AppConfig):
    """处理视频文件（Whisper 提取 + 翻译）"""
    try:
        TaskDAO.update_task(task_id, status='processing', progress=0, log="任务启动")
        if not os.path.exists(file_path):
            TaskDAO.update_task(task_id, status='failed', log="文件丢失")
            return
        
        srt_path = Path(file_path).with_suffix('.srt')
        
        # 步骤 1: Whisper 提取字幕
        if srt_path.exists():
            TaskDAO.update_task(task_id, progress=50, log="基础字幕已存在")
        else:
            TaskDAO.update_task(task_id, progress=5, log=f"加载 Whisper ({config.whisper_model})...")
            try:
                model = WhisperModel(config.whisper_model, device=config.device, compute_type=config.compute_type, download_root="/data/models")
            except Exception as e:
                TaskDAO.update_task(task_id, status='failed', log=f"模型加载失败: {e}")
                return
            
            TaskDAO.update_task(task_id, progress=10, log="正在提取...")
            
            params = {
                'audio': file_path,
                'beam_size': 5,
                'vad_filter': True,
                'vad_parameters': {
                    'threshold': 0.3,
                    'min_speech_duration_ms': 100,
                    'min_silence_duration_ms': 1500,
                    'speech_pad_ms': 300,
                },
                'word_timestamps': True,
                'condition_on_previous_text': True,
                'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }
            
            if config.source_language != 'auto':
                params['language'] = config.source_language
            
            try:
                segments, info = model.transcribe(**params)
                TaskDAO.update_task(task_id, progress=15, log=f"语言: {get_lang_name(info.language)}")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for idx, seg in enumerate(segments, 1):
                        f.write(f"{idx}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n")
                        if idx % 10 == 0:
                            progress = 15 + min(35, int(idx / 300 * 35))
                            TaskDAO.update_task(task_id, progress=progress, log=f"已转写 {idx} 行")
            except Exception as e:
                TaskDAO.update_task(task_id, status='failed', log=f"提取失败: {e}")
                return
        
        # 步骤 2: 翻译字幕（使用新模块或降级到旧版）
        if config.enable_translation:
            TaskDAO.update_task(task_id, progress=50, log="准备翻译...")
            
            if HAS_TRANSLATOR:
                # 新版翻译模块
                provider_cfg = config.get_current_provider_config()
                trans_config = TranslationConfig(
                    api_key=provider_cfg.api_key,
                    base_url=provider_cfg.base_url,
                    model_name=provider_cfg.model_name,
                    target_language=config.target_language,
                    source_language=config.source_language,
                    max_lines_per_batch=config.max_lines_per_batch
                )
                
                # 进度回调
                def progress_callback(current, total, message):
                    progress = 50 + int((current / total) * 45)
                    TaskDAO.update_task(task_id, progress=progress, log=message)
                
                # 执行翻译
                success, msg = translate_srt_file(
                    str(srt_path),
                    trans_config,
                    progress_callback=progress_callback
                )
                
                if success:
                    TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
                else:
                    TaskDAO.update_task(task_id, status='failed', progress=100, log=f"翻译失败: {msg}")
            else:
                # 降级到旧版翻译（保持兼容性）
                TaskDAO.update_task(task_id, status='failed', progress=100, log="翻译模块未安装，请检查 translator.py")
        else:
            TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
        
        # 更新媒体库
        subs_json = scan_file_subtitles(Path(file_path))
        has_translated = ".zh.srt" in subs_json or ".chs.srt" in subs_json
        MediaDAO.update_media_subtitles(file_path, subs_json, has_translated)
        
        # 导出多格式字幕（如果配置了）
        try:
            from subtitle_converter import SubtitleConverter
            
            exported_formats = []
            for fmt in config.export_formats:
                if fmt == 'srt':
                    continue  # SRT 已经生成
                
                try:
                    # 转换原始字幕
                    SubtitleConverter.convert_file(str(srt_path), fmt)
                    exported_formats.append(fmt.upper())
                    
                    # 如果有翻译版本，也转换
                    if config.enable_translation:
                        trans_srt_path = Path(file_path).parent / f"{Path(file_path).stem}.{config.target_language}.srt"
                        if trans_srt_path.exists():
                            SubtitleConverter.convert_file(str(trans_srt_path), fmt)
                except Exception as e:
                    print(f"Failed to export {fmt} format: {e}")
            
            if exported_formats:
                TaskDAO.update_task(task_id, log=f"完成（已导出: {', '.join(exported_formats)}）")
        except ImportError:
            pass  # 如果没有转换器模块，跳过
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        TaskDAO.update_task(task_id, status='failed', log=f"异常: {e}")

def worker_thread():
    """后台工作线程"""
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = get_db_connection()
            conn.execute("SELECT 1 FROM config LIMIT 1")
            conn.close()
            print("[Worker] Database ready, starting...")
            break
        except:
            if i == 0:
                print("[Worker] Waiting for database...")
            time.sleep(1)
    else:
        print("[Worker] ERROR: Database timeout")
        return
    
    while True:
        try:
            config = AppConfig.load_from_db()
            task = TaskDAO.get_pending_task()
            if task:
                process_video_file(task['id'], task['file_path'], config)
            else:
                time.sleep(5)
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(10)

# ============================================================================
# UI 渲染函数
# ============================================================================
def render_config_sidebar():
    """渲染配置侧边栏"""
    with st.sidebar:
        st.caption("参数配置")
        debug_mode = st.toggle("调试日志", value=False)
        config = AppConfig.load_from_db()
        
        with st.expander("Whisper 设置", expanded=False):
            # 内容类型选择（新增）
            content_type_options = config.get_content_type_options()
            content_type_keys = list(content_type_options.keys())
            content_type_values = list(content_type_options.values())
            
            current_index = content_type_keys.index(config.content_type) if config.content_type in content_type_keys else 0
            
            content_type = st.selectbox(
                "内容类型",
                content_type_keys,
                format_func=lambda x: content_type_options[x],
                index=current_index,
                help="选择内容类型以自动优化 VAD 参数"
            )
            
            # 显示当前选择的说明
            if content_type:
                st.caption(f"💡 {config.get_content_type_description(content_type)}")
            
            # 显示当前 VAD 参数（只读）
            temp_config = AppConfig(content_type=content_type)
            vad = temp_config.get_vad_parameters()
            with st.expander("📊 当前 VAD 参数（自动）", expanded=False):
                st.caption(f"阈值: {vad['threshold']}")
                st.caption(f"最小语音时长: {vad['min_speech_duration_ms']}ms")
                st.caption(f"最小静音时长: {vad['min_silence_duration_ms']}ms")
                st.caption(f"语音填充: {vad['speech_pad_ms']}ms")
            
            st.divider()
            
            model_size = st.selectbox("模型大小", ["tiny", "base", "small", "medium", "large-v3"], index=["tiny", "base", "small", "medium", "large-v3"].index(config.whisper_model))
            compute_type = st.selectbox("计算类型", ["int8", "float16"], index=["int8", "float16"].index(config.compute_type))
            device = st.selectbox("设备", ["cpu", "cuda"], index=["cpu", "cuda"].index(config.device))
            s_keys = list(ISO_LANG_MAP.keys())
            source_language = st.selectbox("视频原声", s_keys, format_func=lambda x: ISO_LANG_MAP[x], index=s_keys.index(config.source_language))
            
            st.divider()
            
            # 导出格式选择
            st.caption("🎬 导出格式")
            format_options = {
                'srt': 'SRT - 最通用（默认）',
                'vtt': 'VTT - Web 视频',
                'ass': 'ASS - 样式丰富',
                'ssa': 'SSA - 兼容性好',
                'sub': 'SUB - 老式播放器'
            }
            
            # 使用多选框
            selected_formats = []
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox('SRT', value='srt' in config.export_formats, key='fmt_srt'):
                    selected_formats.append('srt')
                if st.checkbox('VTT', value='vtt' in config.export_formats, key='fmt_vtt'):
                    selected_formats.append('vtt')
                if st.checkbox('ASS', value='ass' in config.export_formats, key='fmt_ass'):
                    selected_formats.append('ass')
            with col2:
                if st.checkbox('SSA', value='ssa' in config.export_formats, key='fmt_ssa'):
                    selected_formats.append('ssa')
                if st.checkbox('SUB', value='sub' in config.export_formats, key='fmt_sub'):
                    selected_formats.append('sub')
            
            if not selected_formats:
                st.warning("⚠️ 至少选择一种格式")
                selected_formats = ['srt']
            
            with st.expander("ℹ️ 格式说明", expanded=False):
                st.caption("**SRT**: 最通用，几乎所有播放器支持")
                st.caption("**VTT**: Web/HTML5 播放器专用")
                st.caption("**ASS**: 支持丰富样式，动漫字幕常用")
                st.caption("**SSA**: ASS 的前身，兼容性更好")
                st.caption("**SUB**: 老式 DVD 播放器支持")
        
        with st.expander("翻译设置", expanded=True):
            enable_translation = st.checkbox("启用翻译", value=config.enable_translation)
            target_lang = st.selectbox("目标语言", TARGET_LANG_OPTIONS, format_func=lambda x: ISO_LANG_MAP.get(x, x), index=TARGET_LANG_OPTIONS.index(config.target_language))
            
            # 新增：分批大小配置
            max_lines = st.number_input(
                "每批最多翻译行数", 
                min_value=100, 
                max_value=2000, 
                value=config.max_lines_per_batch,
                step=100,
                help="短视频会一次性翻译，长视频会按此数量分批"
            )
            
            provider = st.selectbox("AI 提供商", list(LLM_PROVIDERS.keys()), index=list(LLM_PROVIDERS.keys()).index(config.current_provider) if config.current_provider in LLM_PROVIDERS else 0)
            
            provider_cfg = config.provider_configs.get(provider)
            if not provider_cfg:
                default = LLM_PROVIDERS[provider]
                provider_cfg = ProviderConfig(api_key='', base_url=default.get('base_url', ''), model_name=default.get('model', ''))
            
            base_url = st.text_input("Base URL", value=provider_cfg.base_url, help=f"当前提供商: {provider}")
            
            if "Ollama" in provider:
                ollama_models = fetch_ollama_models(base_url)
                if ollama_models:
                    try:
                        idx = ollama_models.index(provider_cfg.model_name)
                    except ValueError:
                        idx = 0
                    model_name = st.selectbox("选择模型", ollama_models, index=idx)
                    if st.button("刷新模型列表", use_container_width=True):
                        st.rerun()
                else:
                    st.error("未检测到本地模型，请检查 Ollama 服务")
                    model_name = st.text_input("手动输入模型", value=provider_cfg.model_name)
                    if st.button("重试连接", use_container_width=True):
                        st.rerun()
                api_key = ""
            else:
                api_key = st.text_input("API Key", value=provider_cfg.api_key, type="password", help="该 Key 仅保存给当前提供商")
                model_name = st.text_input("模型名称", value=provider_cfg.model_name)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                if st.button("测试", use_container_width=True):
                    with st.spinner("连接中..."):
                        ok, msg = test_api_connection(api_key, base_url, model_name)
                        if ok:
                            st.toast("✅ 连接成功")
                        else:
                            st.error(f"❌ {msg}")
            
            with col_t2:
                if st.button("保存", type="primary", use_container_width=True):
                    config.whisper_model = model_size
                    config.compute_type = compute_type
                    config.device = device
                    config.source_language = source_language
                    config.enable_translation = enable_translation
                    config.target_language = target_lang
                    config.max_lines_per_batch = max_lines
                    config.content_type = content_type
                    config.export_formats = selected_formats  # 保存导出格式
                    config.update_provider_config(provider, api_key, base_url, model_name)
                    config.save_to_db()
                    
                    formats_str = ', '.join([f.upper() for f in selected_formats])
                    st.toast(f"✅ 已保存配置（导出: {formats_str}）")
    
    return debug_mode

def render_media_library(debug_mode: bool):
    """渲染媒体库页面"""
    col_filter, col_refresh, col_start = st.columns([2, 2, 2])
    
    with col_filter:
        filter_type = st.radio("筛选", ["全部", "有字幕", "无字幕"], horizontal=True, label_visibility="collapsed")
    
    filter_map = {"全部": "all", "有字幕": "has_subtitle", "无字幕": "no_subtitle"}
    
    with col_refresh:
        if st.button("刷新媒体库", use_container_width=True):
            with st.spinner("扫描中..."):
                cnt, logs = scan_media_directory(debug=debug_mode)
                st.toast(f"更新 {cnt} 个文件")
    
    files = MediaDAO.get_media_files(filter_map[filter_type])
    
    selected_count = sum(1 for f in files if st.session_state.get(f"s_{f['id']}", False))
    
    with col_start:
        btn_txt = f"开始处理 ({selected_count})" if selected_count > 0 else "开始处理"
        if st.button(btn_txt, type="primary", use_container_width=True, disabled=(selected_count == 0)):
            success_count, failed_files = 0, []
            for f in files:
                if st.session_state.get(f"s_{f['id']}", False):
                    ok, msg = TaskDAO.add_task(f['file_path'])
                    if ok:
                        success_count += 1
                    else:
                        failed_files.append((f['file_name'], msg))
            if failed_files:
                st.warning(f"已添加 {success_count} 个任务，{len(failed_files)} 个失败")
                for fname, reason in failed_files[:3]:
                    st.caption(f"❌ {fname}: {reason}")
            else:
                st.toast(f"已添加 {success_count} 个任务")
            time.sleep(1)
            st.rerun()
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    if not files:
        st.info("🔭 暂无文件")
        return
    
    current_select_all = st.checkbox("全选", key="select_all_box")
    last_select_all = st.session_state.get("_last_select_all", False)
    
    if current_select_all != last_select_all:
        if current_select_all:
            for f in files:
                st.session_state[f"s_{f['id']}"] = True
        else:
            for f in files:
                st.session_state[f"s_{f['id']}"] = False
        
        st.session_state["_last_select_all"] = current_select_all
        st.rerun()
    
    for f in files:
        subs, badges = f['subtitles'], ""
        if not subs:
            badges = "<span class='status-chip chip-red'>无字幕</span>"
        else:
            for sub in subs:
                lang = sub['lang'].lower()
                cls = "chip-green" if lang in ['zh', 'chs', 'cht'] else "chip-blue" if lang in ['en', 'eng'] else "chip-gray"
                badges += f"<span class='status-chip {cls}'>{sub['tag']}</span>"
        
        c_check, c_card = st.columns([0.5, 20], gap="medium", vertical_alignment="center")
        with c_check:
            key = f"s_{f['id']}"
            if key not in st.session_state:
                st.session_state[key] = False
            st.checkbox("选", key=key, label_visibility="collapsed")
        
        with c_card:
            st.markdown(f"""<div class="hero-card"><div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;"><div style="font-weight:600; font-size:15px; color:#f4f4f5; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">{f['file_name']}</div><div style="font-size:12px; color:#71717a; min-width:60px; text-align:right;">{format_file_size(f['file_size'])}</div></div><div style="font-size:12px; color:#52525b; margin-bottom:12px; font-family:monospace;">{f['file_path']}</div><div>{badges}</div></div>""", unsafe_allow_html=True)

def render_task_queue():
    """渲染任务队列页面"""
    col_space, col_clear = st.columns([8, 2])
    with col_clear:
        if st.button("清理记录", use_container_width=True):
            TaskDAO.clear_completed_tasks()
            st.rerun()
    
    tasks = TaskDAO.get_all_tasks()
    if not tasks:
        st.info("🔭 队列为空")
        return
    
    has_processing = any(t['status'] == 'processing' for t in tasks)
    
    for t in tasks:
        status_map = {'pending': ('chip-gray', '等待中'), 'processing': ('chip-blue', '处理中'), 'completed': ('chip-green', '完成'), 'failed': ('chip-red', '失败')}
        css_class, status_text = status_map.get(t['status'], ('chip-gray', t['status']))
        progress_html = f"""<div style="margin-top:12px; margin-bottom:8px;"><div style="width:100%; height:4px; background-color:#27272a; border-radius:2px; overflow:hidden;"><div style="width:{t['progress']}%; height:100%; background-color:#2563eb; transition:width 0.3s;"></div></div><div style="font-size:11px; color:#71717a; margin-top:4px; text-align:right;">{t['progress']}%</div></div>""" if t['status'] == 'processing' else ""
        button_space = '<div style="height:40px;"></div>'
        st.markdown(f"""<div class="task-card-wrapper"><div class="hero-card"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div style="flex:1;"><div style="font-weight:600; margin-bottom:8px;">{Path(t['file_path']).name}</div><div style="font-size:13px; color:#a1a1aa;">> {t['log']}</div></div><div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px; margin-left:16px;"><span style="font-size:11px; color:#71717a;">{t['created_at']}</span><span class="status-chip {css_class}">{status_text}</span></div></div>{progress_html}{button_space}</div></div>""", unsafe_allow_html=True)
        col_space, col_ops = st.columns([8, 2])
        with col_ops:
            if t['status'] == 'failed':
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    if st.button("重试", key=f"retry_{t['id']}", use_container_width=True):
                        conn = get_db_connection()
                        conn.execute("UPDATE tasks SET status='pending', progress=0, log='重试中...', updated_at=CURRENT_TIMESTAMP WHERE id=?", (t['id'],))
                        conn.commit()
                        conn.close()
                        st.rerun()
                with subcol2:
                    if st.button("删除", key=f"del_{t['id']}", use_container_width=True):
                        TaskDAO.delete_task(t['id'])
                        st.rerun()
            else:
                if st.button("删除", key=f"del_{t['id']}", use_container_width=True):
                    TaskDAO.delete_task(t['id'])
                    st.rerun()
    
    if has_processing:
        time.sleep(3)
        st.rerun()

def main():
    st.set_page_config(page_title="NAS 字幕管家", page_icon="🎬", layout="wide")
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    st.markdown("<h1 style='margin-bottom: 24px;'>NAS 字幕管家</h1>", unsafe_allow_html=True)
    debug_mode = render_config_sidebar()
    tab1, tab2 = st.tabs(["媒体库", "任务队列"])
    with tab1:
        render_media_library(debug_mode)
    with tab2:
        render_task_queue()

if __name__ == "__main__":
    os.makedirs("/data/models", exist_ok=True)
    init_database()
    
    if 'worker_started' not in st.session_state:
        print("[Main] Starting worker thread...")
        threading.Thread(target=worker_thread, daemon=True).start()
        st.session_state.worker_started = True
        print("[Main] Worker thread started")
    
    main()