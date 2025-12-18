#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAS 字幕管家 (重构版) V6.0.0
主要改进：
1. 分离数据访问层、业务逻辑层、UI层
2. 统一配置管理，修复状态丢失问题
3. 改进错误处理，所有异常都有日志
4. 修复UI布局问题
5. 新增字幕筛选功能
6. 修复字幕语言识别逻辑
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
from openai import OpenAI

# ============================================================================
# 常量定义
# ============================================================================
DB_PATH = "/data/subtitle_manager.db"
MEDIA_ROOT = "/media"
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.m4v', '.webm', '.ts'}
BATCH_SIZE = 15

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

# ============================================================================
# CSS 样式
# ============================================================================
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
    
    /* 任务卡片特殊样式 */
    .task-card-wrapper {
        position: relative;
        margin-bottom: 24px;
    }
    
    /* 强制按钮容器向上移动，嵌入卡片底部 */
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
class AppConfig:
    """应用配置统一管理"""
    whisper_model: str = 'base'
    compute_type: str = 'int8'
    device: str = 'cpu'
    source_language: str = 'auto'
    enable_translation: bool = False
    target_language: str = 'zh'
    api_key: str = ''
    base_url: str = 'http://ollama:11434/v1'
    model_name: str = 'qwen2.5:7b'
    provider: str = 'Ollama (本地模型)'
    
    @classmethod
    def load_from_db(cls) -> 'AppConfig':
        """从数据库加载配置"""
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT key, value FROM config")
            config_dict = {row[0]: row[1] for row in cursor.fetchall()}
            
            return cls(
                whisper_model=config_dict.get('whisper_model', 'base'),
                compute_type=config_dict.get('compute_type', 'int8'),
                device=config_dict.get('device', 'cpu'),
                source_language=config_dict.get('source_language', 'auto'),
                enable_translation=config_dict.get('enable_translation', 'false') == 'true',
                target_language=config_dict.get('target_language', 'zh'),
                api_key=config_dict.get('api_key', ''),
                base_url=config_dict.get('base_url', 'http://ollama:11434/v1'),
                model_name=config_dict.get('model_name', 'qwen2.5:7b'),
                provider=config_dict.get('provider', 'Ollama (本地模型)')
            )
        finally:
            conn.close()
    
    def save_to_db(self):
        """保存配置到数据库"""
        conn = get_db_connection()
        try:
            config_dict = asdict(self)
            config_dict['enable_translation'] = 'true' if self.enable_translation else 'false'
            
            for key, value in config_dict.items():
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    (key, str(value))
                )
            conn.commit()
        except Exception as e:
            print(f"Failed to save config: {e}")
            conn.rollback()
        finally:
            conn.close()

# ============================================================================
# 数据访问层
# ============================================================================
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
    """任务数据访问对象"""
    
    @staticmethod
    def add_task(file_path: str) -> Tuple[bool, str]:
        """添加任务，返回 (成功?, 消息)"""
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO tasks (file_path, status, log) VALUES (?, 'pending', '准备中')",
                (file_path,)
            )
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
            cursor = conn.execute("""
                SELECT id, file_path, status, progress, log, created_at 
                FROM tasks 
                ORDER BY id DESC
            """)
            return [
                {
                    'id': row[0],
                    'file_path': row[1],
                    'status': row[2],
                    'progress': row[3],
                    'log': row[4],
                    'created_at': row[5]
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
    
    @staticmethod
    def get_pending_task() -> Optional[Dict]:
        conn = get_db_connection()
        try:
            result = conn.execute(
                "SELECT id, file_path FROM tasks WHERE status='pending' LIMIT 1"
            ).fetchone()
            return {'id': result[0], 'file_path': result[1]} if result else None
        finally:
            conn.close()
    
    @staticmethod
    def update_task(task_id: int, status=None, progress=None, log=None):
        conn = get_db_connection()
        try:
            updates = []
            params = []
            
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
                conn.execute(
                    f"UPDATE tasks SET {','.join(updates)} WHERE id=?",
                    params
                )
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
    """媒体文件数据访问对象"""
    
    @staticmethod
    def get_media_files(filter_type: str = "all") -> List[Dict]:
        conn = get_db_connection()
        try:
            cursor = conn.execute("""
                SELECT id, file_path, file_name, file_size, subtitles_json, has_translated 
                FROM media_files 
                ORDER BY file_name
            """)
            
            result = []
            for row in cursor.fetchall():
                subtitles = json.loads(row[4])
                has_subtitle = len(subtitles) > 0
                
                media = {
                    'id': row[0],
                    'file_path': row[1],
                    'file_name': row[2],
                    'file_size': row[3],
                    'subtitles': subtitles,
                    'has_subtitle': has_subtitle,
                    'has_translated': row[5]
                }
                
                # 应用筛选
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
            conn.execute("""
                UPDATE media_files 
                SET subtitles_json=?, has_translated=?, updated_at=CURRENT_TIMESTAMP 
                WHERE file_path=?
            """, (subtitles_json, 1 if has_translated else 0, file_path))
            conn.commit()
        except Exception as e:
            print(f"Failed to update media subtitles: {e}")
            conn.rollback()
        finally:
            conn.close()

# ============================================================================
# 业务逻辑层
# ============================================================================
def get_lang_name(code: str) -> str:
    return ISO_LANG_MAP.get(code.lower(), code)

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def detect_lang_by_content(srt_path: str) -> str:
    """改进的语言检测逻辑 - 更精确"""
    try:
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4096)
        
        # 去掉时间戳和序号，只保留文本内容
        # 移除 SRT 格式的时间戳行（例如：00:00:01,000 --> 00:00:03,000）
        content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', '', content)
        # 移除纯数字行（序号）
        content = re.sub(r'^\d+

def scan_file_subtitles(video_path: Path) -> str:
    """扫描视频文件的字幕"""
    subs_list = []
    base_name = video_path.stem
    parent_dir = video_path.parent
    
    try:
        all_files = list(parent_dir.iterdir())
        potential_subs = [
            p for p in all_files 
            if p.is_file() 
            and p.name.lower().endswith('.srt') 
            and p.name.lower().startswith(base_name.lower())
        ]
        
        for sub_path in potential_subs:
            sub_name = sub_path.name
            lang_code = 'unknown'
            tag = '未知'
            
            # 从文件名提取语言代码（改进逻辑）
            # 例如：MIDA-412-C.srt, MIDA-412-C.en.srt, MIDA-412-C.ja.srt
            suffix_part = sub_name[len(base_name):].lower()  # 例如：".srt" 或 ".en.srt"
            
            # 精确匹配语言代码（按长度从长到短，避免误匹配）
            detected = False
            
            # 优先匹配三字母代码（chs, cht, eng, jpn, kor）
            for code in ['chs', 'cht', 'eng', 'jpn', 'kor']:
                if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                    lang_code = code
                    tag = ISO_LANG_MAP[code]
                    detected = True
                    break
            
            # 再匹配两字母代码（zh, en, ja, ko, fr, de, ru, es）
            if not detected:
                for code in ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'ru', 'es']:
                    if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                        lang_code = code
                        tag = ISO_LANG_MAP[code]
                        detected = True
                        break
            
            # 如果文件名没有语言信息，分析内容
            if not detected:
                detected_lang = detect_lang_by_content(str(sub_path))
                if detected_lang in ISO_LANG_MAP:
                    lang_code = detected_lang
                    tag = ISO_LANG_MAP[detected_lang]
            
            # 标记默认字幕（完全匹配视频文件名）
            if sub_path.stem.lower() == base_name.lower():
                tag += " (默认)"
            
            subs_list.append({
                "path": str(sub_path),
                "lang": lang_code,
                "tag": tag
            })
    except Exception as e:
        print(f"Failed to scan subtitles for {video_path}: {e}")
    
    return json.dumps(subs_list, ensure_ascii=False)

def scan_media_directory(directory: str = MEDIA_ROOT, debug: bool = False) -> Tuple[int, List[str]]:
    """扫描媒体目录"""
    conn = get_db_connection()
    added = 0
    debug_logs = []
    path = Path(directory)
    
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
                        batch_data.append((
                            str(file_path),
                            file,
                            file_path.stat().st_size,
                            subs,
                            has_trans
                        ))
                        added += 1
                        if debug:
                            debug_logs.append(f"发现: {file}")
                    except Exception as e:
                        if debug:
                            debug_logs.append(f"错误 {file}: {e}")
        
        if batch_data:
            conn.executemany("""
                INSERT OR REPLACE INTO media_files 
                (file_path, file_name, file_size, subtitles_json, has_translated, updated_at) 
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, batch_data)
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
    """获取 Ollama 模型列表"""
    try:
        root_url = base_url_v1.replace("/v1", "").rstrip("/")
        url = f"{root_url}/api/tags"
        resp = requests.get(url, timeout=2.0)
        
        if resp.status_code == 200:
            return [m['name'] for m in resp.json().get('models', [])]
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
    
    return []

def test_api_connection(api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
    """测试 API 连接"""
    if "ollama" in base_url.lower() or "host.docker.internal" in base_url:
        api_key = "ollama"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            timeout=10
        )
        return True, "连接成功"
    except Exception as e:
        return False, str(e)

# ============================================================================
# 字幕处理逻辑
# ============================================================================
def parse_srt_content(content: str) -> List[Dict]:
    """解析 SRT 内容"""
    blocks = content.strip().split('\n\n')
    result = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                result.append({
                    'index': lines[0],
                    'timecode': lines[1],
                    'text': '\n'.join(lines[2:])
                })
            except:
                continue
    
    return result

def rebuild_srt(subs: List[Dict]) -> str:
    """重建 SRT 文件"""
    return '\n'.join([
        f"{s['index']}\n{s['timecode']}\n{s['text']}\n"
        for s in subs
    ])

def translate_subtitles(srt_path: str, config: AppConfig, task_id: int) -> bool:
    """翻译字幕文件"""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        subs = parse_srt_content(content)
        if not subs:
            TaskDAO.update_task(task_id, log="字幕解析失败")
            return False
        
        # 处理 API Key
        api_key = config.api_key
        if "ollama" in config.base_url.lower() or "host.docker.internal" in config.base_url:
            api_key = "ollama"
        
        client = OpenAI(api_key=api_key, base_url=config.base_url)
        target_name = get_lang_name(config.target_language)
        
        trans_subs = []
        total_batches = (len(subs) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(subs), BATCH_SIZE):
            batch = subs[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            TaskDAO.update_task(task_id, log=f"正在翻译第 {batch_num}/{total_batches} 批...")
            
            prompt = f"""你是一名资深的电影字幕翻译。请将以下字幕翻译成{target_name}。
翻译原则：
1. 信达雅：译文要通顺、符合中文口语习惯。
2. 意译优先：遇到俗语或梗，请转换为中文对应的表达。
3. 简洁：字幕不宜过长。
4. 【死命令】绝对不要修改序号和时间轴！

内容如下：
{rebuild_srt(batch)}"""
            
            try:
                resp = client.chat.completions.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=90
                )
                
                parsed = parse_srt_content(resp.choices[0].message.content.strip())
                
                if len(parsed) == len(batch):
                    trans_subs.extend(parsed)
                else:
                    trans_subs.extend(batch)
                    TaskDAO.update_task(task_id, log=f"⚠️ 第 {batch_num} 批格式异常，保留原文")
            except Exception as e:
                print(f"Translation batch {batch_num} failed: {e}")
                trans_subs.extend(batch)
                TaskDAO.update_task(task_id, log=f"❌ API错误: {str(e)}")
            
            # 更新进度
            progress = 50 + int((batch_num / total_batches) * 45)
            TaskDAO.update_task(task_id, progress=progress)
        
        # 保存翻译结果
        out_path = Path(srt_path).parent / f"{Path(srt_path).stem}.{config.target_language}.srt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(rebuild_srt(trans_subs))
        
        return True
    except Exception as e:
        print(f"Translation failed: {e}")
        TaskDAO.update_task(task_id, log=f"翻译异常: {e}")
        return False

def process_video_file(task_id: int, file_path: str, config: AppConfig):
    """处理视频文件"""
    try:
        TaskDAO.update_task(task_id, status='processing', progress=0, log="任务启动")
        
        if not os.path.exists(file_path):
            TaskDAO.update_task(task_id, status='failed', log="文件丢失")
            return
        
        srt_path = Path(file_path).with_suffix('.srt')
        
        # 检查基础字幕是否存在
        if srt_path.exists():
            TaskDAO.update_task(task_id, progress=50, log="基础字幕已存在")
        else:
            # 提取字幕
            TaskDAO.update_task(task_id, progress=5, log=f"加载 Whisper ({config.whisper_model})...")
            
            try:
                model = WhisperModel(
                    config.whisper_model,
                    device=config.device,
                    compute_type=config.compute_type,
                    download_root="/data/models"
                )
            except Exception as e:
                TaskDAO.update_task(task_id, status='failed', log=f"模型加载失败: {e}")
                return
            
            TaskDAO.update_task(task_id, progress=10, log="正在提取...")
            
            params = {
                'audio': file_path,
                'beam_size': 5,
                'vad_filter': True
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
        
        # 翻译字幕
        if config.enable_translation:
            TaskDAO.update_task(task_id, progress=50, log="准备翻译...")
            success = translate_subtitles(str(srt_path), config, task_id)
            
            if success:
                TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
            else:
                TaskDAO.update_task(task_id, status='failed', progress=100, log="翻译失败")
        else:
            TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
        
        # 更新媒体库状态
        subs_json = scan_file_subtitles(Path(file_path))
        has_translated = ".zh.srt" in subs_json or ".chs.srt" in subs_json
        MediaDAO.update_media_subtitles(file_path, subs_json, has_translated)
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        TaskDAO.update_task(task_id, status='failed', log=f"异常: {e}")

def worker_thread():
    """后台工作线程"""
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
# UI 层
# ============================================================================
def render_config_sidebar():
    """渲染配置侧边栏"""
    with st.sidebar:
        st.caption("参数配置")
        debug_mode = st.toggle("调试日志", value=False)
        
        # 加载当前配置
        config = AppConfig.load_from_db()
        
        # Whisper 设置
        with st.expander("Whisper 设置", expanded=False):
            model_size = st.selectbox(
                "模型大小",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v3"].index(config.whisper_model)
            )
            compute_type = st.selectbox(
                "计算类型",
                ["int8", "float16"],
                index=["int8", "float16"].index(config.compute_type)
            )
            device = st.selectbox(
                "设备",
                ["cpu", "cuda"],
                index=["cpu", "cuda"].index(config.device)
            )
            
            s_keys = list(ISO_LANG_MAP.keys())
            source_language = st.selectbox(
                "视频原声",
                s_keys,
                format_func=lambda x: ISO_LANG_MAP[x],
                index=s_keys.index(config.source_language)
            )
        
        # 翻译设置
        with st.expander("翻译设置", expanded=True):
            enable_translation = st.checkbox(
                "启用翻译",
                value=config.enable_translation
            )
            
            target_lang = st.selectbox(
                "目标语言",
                TARGET_LANG_OPTIONS,
                format_func=lambda x: ISO_LANG_MAP.get(x, x),
                index=TARGET_LANG_OPTIONS.index(config.target_language)
            )
            
            provider = st.selectbox(
                "AI 提供商",
                list(LLM_PROVIDERS.keys()),
                index=list(LLM_PROVIDERS.keys()).index(config.provider) if config.provider in LLM_PROVIDERS else 0
            )
            
            sel_prov = LLM_PROVIDERS[provider]
            
            # 如果切换了 provider，使用新的默认值
            if provider != config.provider:
                default_base = sel_prov['base_url']
                default_model = sel_prov['model']
            else:
                default_base = config.base_url
                default_model = config.model_name
            
            base_url = st.text_input("Base URL", value=default_base)
            
            # Ollama 特殊处理
            if "Ollama" in provider:
                ollama_models = fetch_ollama_models(base_url)
                
                if ollama_models:
                    try:
                        idx = ollama_models.index(default_model)
                    except ValueError:
                        idx = 0
                    
                    model_name = st.selectbox("选择模型", ollama_models, index=idx)
                    
                    if st.button("刷新模型列表", use_container_width=True):
                        st.rerun()
                else:
                    st.error("未检测到本地模型，请检查 Ollama 服务")
                    model_name = st.text_input("手动输入模型", value=default_model)
                    
                    if st.button("重试连接", use_container_width=True):
                        st.rerun()
                
                api_key = ""
            else:
                api_key = st.text_input("API Key", value=config.api_key, type="password")
                model_name = st.text_input("模型名称", value=default_model)
            
            # 测试和保存按钮
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
                    new_config = AppConfig(
                        whisper_model=model_size,
                        compute_type=compute_type,
                        device=device,
                        source_language=source_language,
                        enable_translation=enable_translation,
                        target_language=target_lang,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        provider=provider
                    )
                    new_config.save_to_db()
                    st.toast("✅ 已保存")
    
    return debug_mode

def render_media_library(debug_mode: bool):
    """渲染媒体库页面"""
    # 筛选器
    col_filter, col_refresh, col_start = st.columns([2, 2, 2])
    
    with col_filter:
        filter_type = st.radio(
            "筛选",
            ["全部", "有字幕", "无字幕"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    filter_map = {
        "全部": "all",
        "有字幕": "has_subtitle",
        "无字幕": "no_subtitle"
    }
    
    with col_refresh:
        if st.button("刷新媒体库", use_container_width=True):
            with st.spinner("扫描中..."):
                cnt, logs = scan_media_directory(debug=debug_mode)
                st.toast(f"更新 {cnt} 个文件")
    
    # 获取文件列表
    files = MediaDAO.get_media_files(filter_map[filter_type])
    
    # 计算选中数量
    selected_count = sum(1 for f in files if st.session_state.get(f"s_{f['id']}", False))
    
    with col_start:
        btn_txt = f"开始处理 ({selected_count})" if selected_count > 0 else "开始处理"
        if st.button(btn_txt, type="primary", use_container_width=True, disabled=(selected_count == 0)):
            success_count = 0
            failed_files = []
            
            for f in files:
                if st.session_state.get(f"s_{f['id']}", False):
                    ok, msg = TaskDAO.add_task(f['file_path'])
                    if ok:
                        success_count += 1
                    else:
                        failed_files.append((f['file_name'], msg))
            
            if failed_files:
                st.warning(f"已添加 {success_count} 个任务，{len(failed_files)} 个失败")
                for fname, reason in failed_files[:3]:  # 只显示前3个
                    st.caption(f"❌ {fname}: {reason}")
            else:
                st.toast(f"已添加 {success_count} 个任务")
            
            time.sleep(1)
            st.rerun()
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    if not files:
        st.info("🔭 暂无文件")
    else:
        # 全选按钮
        if st.checkbox("全选", key="select_all_box"):
            for f in files:
                st.session_state[f"s_{f['id']}"] = True
        else:
            # 取消全选时，清除所有选择
            if st.session_state.get("_last_select_all", False):
                for f in files:
                    st.session_state[f"s_{f['id']}"] = False
        
        st.session_state["_last_select_all"] = st.session_state.get("select_all_box", False)
        
        # 渲染文件列表
        for f in files:
            subs = f['subtitles']
            badges = ""
            
            if not subs:
                badges = "<span class='status-chip chip-red'>无字幕</span>"
            else:
                for sub in subs:
                    lang = sub['lang'].lower()
                    # 使用更准确的语言分类
                    if lang in ['zh', 'chs', 'cht']:
                        cls = "chip-green"
                    elif lang in ['en', 'eng']:
                        cls = "chip-blue"
                    else:
                        cls = "chip-gray"
                    
                    badges += f"<span class='status-chip {cls}'>{sub['tag']}</span>"
            
            # 布局：checkbox + 卡片
            c_check, c_card = st.columns([0.5, 20], gap="medium", vertical_alignment="center")
            
            with c_check:
                key = f"s_{f['id']}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.checkbox("选", key=key, label_visibility="collapsed")
            
            with c_card:
                st.markdown(f"""
                <div class="hero-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div style="font-weight:600; font-size:15px; color:#f4f4f5; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">{f['file_name']}</div>
                        <div style="font-size:12px; color:#71717a; min-width:60px; text-align:right;">{format_file_size(f['file_size'])}</div>
                    </div>
                    <div style="font-size:12px; color:#52525b; margin-bottom:12px; font-family:monospace;">{f['file_path']}</div>
                    <div>{badges}</div>
                </div>
                """, unsafe_allow_html=True)

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
    
    for t in tasks:
        status_map = {
            'pending': ('chip-gray', '等待中'),
            'processing': ('chip-blue', '处理中'),
            'completed': ('chip-green', '完成'),
            'failed': ('chip-red', '失败')
        }
        css_class, status_text = status_map.get(t['status'], ('chip-gray', t['status']))
        
        # 构建进度条 HTML
        progress_html = ""
        if t['status'] == 'processing':
            progress_html = f"""
            <div style="margin-top:12px; margin-bottom:8px;">
                <div style="width:100%; height:4px; background-color:#27272a; border-radius:2px; overflow:hidden;">
                    <div style="width:{t['progress']}%; height:100%; background-color:#2563eb; transition:width 0.3s;"></div>
                </div>
                <div style="font-size:11px; color:#71717a; margin-top:4px; text-align:right;">{t['progress']}%</div>
            </div>
            """
        
        # 预留按钮空间（40px 高度）
        button_space = '<div style="height:40px;"></div>'
        
        # 卡片 HTML
        st.markdown(f"""
        <div class="task-card-wrapper">
            <div class="hero-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <div style="font-weight:600; margin-bottom:8px;">{Path(t['file_path']).name}</div>
                        <div style="font-size:13px; color:#a1a1aa;">> {t['log']}</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px; margin-left:16px;">
                        <span style="font-size:11px; color:#71717a;">{t['created_at']}</span>
                        <span class="status-chip {css_class}">{status_text}</span>
                    </div>
                </div>
                {progress_html}
                {button_space}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 操作按钮（通过 CSS 负边距嵌入卡片底部）
        col_space, col_ops = st.columns([8, 2])
        with col_ops:
            if t['status'] == 'failed':
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    if st.button("重试", key=f"retry_{t['id']}", use_container_width=True):
                        conn = get_db_connection()
                        conn.execute(
                            "UPDATE tasks SET status='pending', progress=0, log='重试中...', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                            (t['id'],)
                        )
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
    
    # 自动刷新
    time.sleep(3)
    st.rerun()

def main():
    st.set_page_config(page_title="NAS 字幕管家", page_icon="🎬", layout="wide")
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    st.markdown("<h1 style='margin-bottom: 24px;'>NAS 字幕管家</h1>", unsafe_allow_html=True)
    
    # 侧边栏配置
    debug_mode = render_config_sidebar()
    
    # 主区域
    tab1, tab2 = st.tabs(["媒体库", "任务队列"])
    
    with tab1:
        render_media_library(debug_mode)
    
    with tab2:
        render_task_queue()

if __name__ == "__main__":
    os.makedirs("/data/models", exist_ok=True)
    init_database()
    
    if 'worker_started' not in st.session_state:
        threading.Thread(target=worker_thread, daemon=True).start()
        st.session_state.worker_started = True
    
    main(), '', content, flags=re.MULTILINE)
        
        # 统计各种字符数量
        chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', content))
        hiragana_count = len(re.findall(r'[\u3040-\u309f]', content))  # 平假名
        katakana_count = len(re.findall(r'[\u30a0-\u30ff]', content))  # 片假名
        hangul_count = len(re.findall(r'[\uac00-\ud7af]', content))    # 韩文
        
        # 繁体特征字（这些字在简体中很少出现）
        traditional_markers = ['臺', '灣', '繁', '體', '於', '與', '個', '們', '裡', '這', '妳', '臉', '廳', '學', '習']
        traditional_count = sum(1 for char in traditional_markers if char in content)
        
        # 英文单词数量（至少3个字母）
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        
        # 决策逻辑
        if hiragana_count > 5 or katakana_count > 5:
            return 'ja'  # 日语
        
        if hangul_count > 10:
            return 'ko'  # 韩语
        
        if traditional_count >= 3 and chinese_count > 10:
            return 'cht'  # 繁体中文
        
        if chinese_count > 10:
            return 'chs'  # 简体中文
        
        if len(english_words) > 15:
            return 'en'  # 英语
        
        return 'unknown'
    except Exception as e:
        print(f"Language detection failed for {srt_path}: {e}")
        return 'unknown'

def scan_file_subtitles(video_path: Path) -> str:
    """扫描视频文件的字幕"""
    subs_list = []
    base_name = video_path.stem
    parent_dir = video_path.parent
    
    try:
        all_files = list(parent_dir.iterdir())
        potential_subs = [
            p for p in all_files 
            if p.is_file() 
            and p.name.lower().endswith('.srt') 
            and p.name.lower().startswith(base_name.lower())
        ]
        
        for sub_path in potential_subs:
            sub_name = sub_path.name
            lang_code = 'unknown'
            tag = '未知'
            
            # 从文件名提取语言代码（改进逻辑）
            # 例如：MIDA-412-C.srt, MIDA-412-C.en.srt, MIDA-412-C.ja.srt
            suffix_part = sub_name[len(base_name):].lower()  # 例如：".srt" 或 ".en.srt"
            
            # 精确匹配语言代码（按长度从长到短，避免误匹配）
            detected = False
            
            # 优先匹配三字母代码（chs, cht, eng, jpn, kor）
            for code in ['chs', 'cht', 'eng', 'jpn', 'kor']:
                if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                    lang_code = code
                    tag = ISO_LANG_MAP[code]
                    detected = True
                    break
            
            # 再匹配两字母代码（zh, en, ja, ko, fr, de, ru, es）
            if not detected:
                for code in ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'ru', 'es']:
                    if f".{code}." in suffix_part or suffix_part.endswith(f".{code}"):
                        lang_code = code
                        tag = ISO_LANG_MAP[code]
                        detected = True
                        break
            
            # 如果文件名没有语言信息，分析内容
            if not detected:
                detected_lang = detect_lang_by_content(str(sub_path))
                if detected_lang in ISO_LANG_MAP:
                    lang_code = detected_lang
                    tag = ISO_LANG_MAP[detected_lang]
            
            # 标记默认字幕（完全匹配视频文件名）
            if sub_path.stem.lower() == base_name.lower():
                tag += " (默认)"
            
            subs_list.append({
                "path": str(sub_path),
                "lang": lang_code,
                "tag": tag
            })
    except Exception as e:
        print(f"Failed to scan subtitles for {video_path}: {e}")
    
    return json.dumps(subs_list, ensure_ascii=False)

def scan_media_directory(directory: str = MEDIA_ROOT, debug: bool = False) -> Tuple[int, List[str]]:
    """扫描媒体目录"""
    conn = get_db_connection()
    added = 0
    debug_logs = []
    path = Path(directory)
    
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
                        batch_data.append((
                            str(file_path),
                            file,
                            file_path.stat().st_size,
                            subs,
                            has_trans
                        ))
                        added += 1
                        if debug:
                            debug_logs.append(f"发现: {file}")
                    except Exception as e:
                        if debug:
                            debug_logs.append(f"错误 {file}: {e}")
        
        if batch_data:
            conn.executemany("""
                INSERT OR REPLACE INTO media_files 
                (file_path, file_name, file_size, subtitles_json, has_translated, updated_at) 
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, batch_data)
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
    """获取 Ollama 模型列表"""
    try:
        root_url = base_url_v1.replace("/v1", "").rstrip("/")
        url = f"{root_url}/api/tags"
        resp = requests.get(url, timeout=2.0)
        
        if resp.status_code == 200:
            return [m['name'] for m in resp.json().get('models', [])]
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
    
    return []

def test_api_connection(api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
    """测试 API 连接"""
    if "ollama" in base_url.lower() or "host.docker.internal" in base_url:
        api_key = "ollama"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            timeout=10
        )
        return True, "连接成功"
    except Exception as e:
        return False, str(e)

# ============================================================================
# 字幕处理逻辑
# ============================================================================
def parse_srt_content(content: str) -> List[Dict]:
    """解析 SRT 内容"""
    blocks = content.strip().split('\n\n')
    result = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                result.append({
                    'index': lines[0],
                    'timecode': lines[1],
                    'text': '\n'.join(lines[2:])
                })
            except:
                continue
    
    return result

def rebuild_srt(subs: List[Dict]) -> str:
    """重建 SRT 文件"""
    return '\n'.join([
        f"{s['index']}\n{s['timecode']}\n{s['text']}\n"
        for s in subs
    ])

def translate_subtitles(srt_path: str, config: AppConfig, task_id: int) -> bool:
    """翻译字幕文件"""
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        subs = parse_srt_content(content)
        if not subs:
            TaskDAO.update_task(task_id, log="字幕解析失败")
            return False
        
        # 处理 API Key
        api_key = config.api_key
        if "ollama" in config.base_url.lower() or "host.docker.internal" in config.base_url:
            api_key = "ollama"
        
        client = OpenAI(api_key=api_key, base_url=config.base_url)
        target_name = get_lang_name(config.target_language)
        
        trans_subs = []
        total_batches = (len(subs) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(subs), BATCH_SIZE):
            batch = subs[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            TaskDAO.update_task(task_id, log=f"正在翻译第 {batch_num}/{total_batches} 批...")
            
            prompt = f"""你是一名资深的电影字幕翻译。请将以下字幕翻译成{target_name}。
翻译原则：
1. 信达雅：译文要通顺、符合中文口语习惯。
2. 意译优先：遇到俗语或梗，请转换为中文对应的表达。
3. 简洁：字幕不宜过长。
4. 【死命令】绝对不要修改序号和时间轴！

内容如下：
{rebuild_srt(batch)}"""
            
            try:
                resp = client.chat.completions.create(
                    model=config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=90
                )
                
                parsed = parse_srt_content(resp.choices[0].message.content.strip())
                
                if len(parsed) == len(batch):
                    trans_subs.extend(parsed)
                else:
                    trans_subs.extend(batch)
                    TaskDAO.update_task(task_id, log=f"⚠️ 第 {batch_num} 批格式异常，保留原文")
            except Exception as e:
                print(f"Translation batch {batch_num} failed: {e}")
                trans_subs.extend(batch)
                TaskDAO.update_task(task_id, log=f"❌ API错误: {str(e)}")
            
            # 更新进度
            progress = 50 + int((batch_num / total_batches) * 45)
            TaskDAO.update_task(task_id, progress=progress)
        
        # 保存翻译结果
        out_path = Path(srt_path).parent / f"{Path(srt_path).stem}.{config.target_language}.srt"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(rebuild_srt(trans_subs))
        
        return True
    except Exception as e:
        print(f"Translation failed: {e}")
        TaskDAO.update_task(task_id, log=f"翻译异常: {e}")
        return False

def process_video_file(task_id: int, file_path: str, config: AppConfig):
    """处理视频文件"""
    try:
        TaskDAO.update_task(task_id, status='processing', progress=0, log="任务启动")
        
        if not os.path.exists(file_path):
            TaskDAO.update_task(task_id, status='failed', log="文件丢失")
            return
        
        srt_path = Path(file_path).with_suffix('.srt')
        
        # 检查基础字幕是否存在
        if srt_path.exists():
            TaskDAO.update_task(task_id, progress=50, log="基础字幕已存在")
        else:
            # 提取字幕
            TaskDAO.update_task(task_id, progress=5, log=f"加载 Whisper ({config.whisper_model})...")
            
            try:
                model = WhisperModel(
                    config.whisper_model,
                    device=config.device,
                    compute_type=config.compute_type,
                    download_root="/data/models"
                )
            except Exception as e:
                TaskDAO.update_task(task_id, status='failed', log=f"模型加载失败: {e}")
                return
            
            TaskDAO.update_task(task_id, progress=10, log="正在提取...")
            
            params = {
                'audio': file_path,
                'beam_size': 5,
                'vad_filter': True
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
        
        # 翻译字幕
        if config.enable_translation:
            TaskDAO.update_task(task_id, progress=50, log="准备翻译...")
            success = translate_subtitles(str(srt_path), config, task_id)
            
            if success:
                TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
            else:
                TaskDAO.update_task(task_id, status='failed', progress=100, log="翻译失败")
        else:
            TaskDAO.update_task(task_id, status='completed', progress=100, log="完成")
        
        # 更新媒体库状态
        subs_json = scan_file_subtitles(Path(file_path))
        has_translated = ".zh.srt" in subs_json or ".chs.srt" in subs_json
        MediaDAO.update_media_subtitles(file_path, subs_json, has_translated)
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        TaskDAO.update_task(task_id, status='failed', log=f"异常: {e}")

def worker_thread():
    """后台工作线程"""
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
# UI 层
# ============================================================================
def render_config_sidebar():
    """渲染配置侧边栏"""
    with st.sidebar:
        st.caption("参数配置")
        debug_mode = st.toggle("调试日志", value=False)
        
        # 加载当前配置
        config = AppConfig.load_from_db()
        
        # Whisper 设置
        with st.expander("Whisper 设置", expanded=False):
            model_size = st.selectbox(
                "模型大小",
                ["tiny", "base", "small", "medium", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v3"].index(config.whisper_model)
            )
            compute_type = st.selectbox(
                "计算类型",
                ["int8", "float16"],
                index=["int8", "float16"].index(config.compute_type)
            )
            device = st.selectbox(
                "设备",
                ["cpu", "cuda"],
                index=["cpu", "cuda"].index(config.device)
            )
            
            s_keys = list(ISO_LANG_MAP.keys())
            source_language = st.selectbox(
                "视频原声",
                s_keys,
                format_func=lambda x: ISO_LANG_MAP[x],
                index=s_keys.index(config.source_language)
            )
        
        # 翻译设置
        with st.expander("翻译设置", expanded=True):
            enable_translation = st.checkbox(
                "启用翻译",
                value=config.enable_translation
            )
            
            target_lang = st.selectbox(
                "目标语言",
                TARGET_LANG_OPTIONS,
                format_func=lambda x: ISO_LANG_MAP.get(x, x),
                index=TARGET_LANG_OPTIONS.index(config.target_language)
            )
            
            provider = st.selectbox(
                "AI 提供商",
                list(LLM_PROVIDERS.keys()),
                index=list(LLM_PROVIDERS.keys()).index(config.provider) if config.provider in LLM_PROVIDERS else 0
            )
            
            sel_prov = LLM_PROVIDERS[provider]
            
            # 如果切换了 provider，使用新的默认值
            if provider != config.provider:
                default_base = sel_prov['base_url']
                default_model = sel_prov['model']
            else:
                default_base = config.base_url
                default_model = config.model_name
            
            base_url = st.text_input("Base URL", value=default_base)
            
            # Ollama 特殊处理
            if "Ollama" in provider:
                ollama_models = fetch_ollama_models(base_url)
                
                if ollama_models:
                    try:
                        idx = ollama_models.index(default_model)
                    except ValueError:
                        idx = 0
                    
                    model_name = st.selectbox("选择模型", ollama_models, index=idx)
                    
                    if st.button("刷新模型列表", use_container_width=True):
                        st.rerun()
                else:
                    st.error("未检测到本地模型，请检查 Ollama 服务")
                    model_name = st.text_input("手动输入模型", value=default_model)
                    
                    if st.button("重试连接", use_container_width=True):
                        st.rerun()
                
                api_key = ""
            else:
                api_key = st.text_input("API Key", value=config.api_key, type="password")
                model_name = st.text_input("模型名称", value=default_model)
            
            # 测试和保存按钮
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
                    new_config = AppConfig(
                        whisper_model=model_size,
                        compute_type=compute_type,
                        device=device,
                        source_language=source_language,
                        enable_translation=enable_translation,
                        target_language=target_lang,
                        api_key=api_key,
                        base_url=base_url,
                        model_name=model_name,
                        provider=provider
                    )
                    new_config.save_to_db()
                    st.toast("✅ 已保存")
    
    return debug_mode

def render_media_library(debug_mode: bool):
    """渲染媒体库页面"""
    # 筛选器
    col_filter, col_refresh, col_start = st.columns([2, 2, 2])
    
    with col_filter:
        filter_type = st.radio(
            "筛选",
            ["全部", "有字幕", "无字幕"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    filter_map = {
        "全部": "all",
        "有字幕": "has_subtitle",
        "无字幕": "no_subtitle"
    }
    
    with col_refresh:
        if st.button("刷新媒体库", use_container_width=True):
            with st.spinner("扫描中..."):
                cnt, logs = scan_media_directory(debug=debug_mode)
                st.toast(f"更新 {cnt} 个文件")
    
    # 获取文件列表
    files = MediaDAO.get_media_files(filter_map[filter_type])
    
    # 计算选中数量
    selected_count = sum(1 for f in files if st.session_state.get(f"s_{f['id']}", False))
    
    with col_start:
        btn_txt = f"开始处理 ({selected_count})" if selected_count > 0 else "开始处理"
        if st.button(btn_txt, type="primary", use_container_width=True, disabled=(selected_count == 0)):
            success_count = 0
            failed_files = []
            
            for f in files:
                if st.session_state.get(f"s_{f['id']}", False):
                    ok, msg = TaskDAO.add_task(f['file_path'])
                    if ok:
                        success_count += 1
                    else:
                        failed_files.append((f['file_name'], msg))
            
            if failed_files:
                st.warning(f"已添加 {success_count} 个任务，{len(failed_files)} 个失败")
                for fname, reason in failed_files[:3]:  # 只显示前3个
                    st.caption(f"❌ {fname}: {reason}")
            else:
                st.toast(f"已添加 {success_count} 个任务")
            
            time.sleep(1)
            st.rerun()
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    if not files:
        st.info("🔭 暂无文件")
    else:
        # 全选按钮
        if st.checkbox("全选", key="select_all_box"):
            for f in files:
                st.session_state[f"s_{f['id']}"] = True
        else:
            # 取消全选时，清除所有选择
            if st.session_state.get("_last_select_all", False):
                for f in files:
                    st.session_state[f"s_{f['id']}"] = False
        
        st.session_state["_last_select_all"] = st.session_state.get("select_all_box", False)
        
        # 渲染文件列表
        for f in files:
            subs = f['subtitles']
            badges = ""
            
            if not subs:
                badges = "<span class='status-chip chip-red'>无字幕</span>"
            else:
                for sub in subs:
                    lang = sub['lang'].lower()
                    # 使用更准确的语言分类
                    if lang in ['zh', 'chs', 'cht']:
                        cls = "chip-green"
                    elif lang in ['en', 'eng']:
                        cls = "chip-blue"
                    else:
                        cls = "chip-gray"
                    
                    badges += f"<span class='status-chip {cls}'>{sub['tag']}</span>"
            
            # 布局：checkbox + 卡片
            c_check, c_card = st.columns([0.5, 20], gap="medium", vertical_alignment="center")
            
            with c_check:
                key = f"s_{f['id']}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.checkbox("选", key=key, label_visibility="collapsed")
            
            with c_card:
                st.markdown(f"""
                <div class="hero-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div style="font-weight:600; font-size:15px; color:#f4f4f5; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">{f['file_name']}</div>
                        <div style="font-size:12px; color:#71717a; min-width:60px; text-align:right;">{format_file_size(f['file_size'])}</div>
                    </div>
                    <div style="font-size:12px; color:#52525b; margin-bottom:12px; font-family:monospace;">{f['file_path']}</div>
                    <div>{badges}</div>
                </div>
                """, unsafe_allow_html=True)

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
    
    for t in tasks:
        status_map = {
            'pending': ('chip-gray', '等待中'),
            'processing': ('chip-blue', '处理中'),
            'completed': ('chip-green', '完成'),
            'failed': ('chip-red', '失败')
        }
        css_class, status_text = status_map.get(t['status'], ('chip-gray', t['status']))
        
        # 构建进度条 HTML
        progress_html = ""
        if t['status'] == 'processing':
            progress_html = f"""
            <div style="margin-top:12px; margin-bottom:8px;">
                <div style="width:100%; height:4px; background-color:#27272a; border-radius:2px; overflow:hidden;">
                    <div style="width:{t['progress']}%; height:100%; background-color:#2563eb; transition:width 0.3s;"></div>
                </div>
                <div style="font-size:11px; color:#71717a; margin-top:4px; text-align:right;">{t['progress']}%</div>
            </div>
            """
        
        # 预留按钮空间（40px 高度）
        button_space = '<div style="height:40px;"></div>'
        
        # 卡片 HTML
        st.markdown(f"""
        <div class="task-card-wrapper">
            <div class="hero-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex:1;">
                        <div style="font-weight:600; margin-bottom:8px;">{Path(t['file_path']).name}</div>
                        <div style="font-size:13px; color:#a1a1aa;">> {t['log']}</div>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px; margin-left:16px;">
                        <span style="font-size:11px; color:#71717a;">{t['created_at']}</span>
                        <span class="status-chip {css_class}">{status_text}</span>
                    </div>
                </div>
                {progress_html}
                {button_space}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 操作按钮（通过 CSS 负边距嵌入卡片底部）
        col_space, col_ops = st.columns([8, 2])
        with col_ops:
            if t['status'] == 'failed':
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    if st.button("重试", key=f"retry_{t['id']}", use_container_width=True):
                        conn = get_db_connection()
                        conn.execute(
                            "UPDATE tasks SET status='pending', progress=0, log='重试中...', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                            (t['id'],)
                        )
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
    
    # 自动刷新
    time.sleep(3)
    st.rerun()

def main():
    st.set_page_config(page_title="NAS 字幕管家", page_icon="🎬", layout="wide")
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    st.markdown("<h1 style='margin-bottom: 24px;'>NAS 字幕管家</h1>", unsafe_allow_html=True)
    
    # 侧边栏配置
    debug_mode = render_config_sidebar()
    
    # 主区域
    tab1, tab2 = st.tabs(["媒体库", "任务队列"])
    
    with tab1:
        render_media_library(debug_mode)
    
    with tab2:
        render_task_queue()

if __name__ == "__main__":
    os.makedirs("/data/models", exist_ok=True)
    init_database()
    
    if 'worker_started' not in st.session_state:
        threading.Thread(target=worker_thread, daemon=True).start()
        st.session_state.worker_started = True
    
    main()