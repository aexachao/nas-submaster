#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接管理
提供统一的数据库访问接口
"""

import sqlite3
from pathlib import Path
from typing import Optional


# 数据库路径
DB_PATH = "/data/subtitle_manager.db"


def get_db_connection() -> sqlite3.Connection:
    """
    获取数据库连接
    
    Returns:
        sqlite3.Connection: 数据库连接对象
    """
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_database():
    """
    初始化数据库表结构
    如果表不存在则创建
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 创建媒体文件表
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
        
        # 创建任务表
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
        
        # 创建配置表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        conn.commit()
        print("[Database] Tables initialized successfully")
        
    except Exception as e:
        print(f"[Database] Initialization failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def check_database_health() -> bool:
    """
    检查数据库健康状态
    
    Returns:
        bool: 数据库是否可用
    """
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1 FROM config LIMIT 1")
        conn.close()
        return True
    except Exception as e:
        print(f"[Database] Health check failed: {e}")
        return False


def wait_for_database(max_retries: int = 30, retry_interval: float = 1.0) -> bool:
    """
    等待数据库就绪（用于容器启动时）
    
    Args:
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）
    
    Returns:
        bool: 数据库是否就绪
    """
    import time
    
    for i in range(max_retries):
        if check_database_health():
            if i > 0:
                print(f"[Database] Ready after {i+1} attempts")
            return True
        
        if i == 0:
            print("[Database] Waiting for database to be ready...")
        
        time.sleep(retry_interval)
    
    print(f"[Database] Timeout after {max_retries} attempts")
    return False


class DatabaseConnection:
    """
    数据库连接上下文管理器
    用法：
        with DatabaseConnection() as conn:
            cursor = conn.execute("SELECT * FROM tasks")
    """
    
    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self) -> sqlite3.Connection:
        self.conn = get_db_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
        return False  # 不抑制异常


# ============================================================================
# 数据库工具函数
# ============================================================================

def execute_query(query: str, params: tuple = ()) -> list:
    """
    执行查询语句
    
    Args:
        query: SQL 查询语句
        params: 查询参数
    
    Returns:
        list: 查询结果
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(query, params)
        return cursor.fetchall()
    finally:
        conn.close()


def execute_update(query: str, params: tuple = ()) -> int:
    """
    执行更新语句（INSERT/UPDATE/DELETE）
    
    Args:
        query: SQL 更新语句
        params: 更新参数
    
    Returns:
        int: 受影响的行数
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def execute_many(query: str, params_list: list) -> int:
    """
    批量执行语句
    
    Args:
        query: SQL 语句
        params_list: 参数列表
    
    Returns:
        int: 受影响的总行数
    """
    conn = get_db_connection()
    try:
        cursor = conn.executemany(query, params_list)
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
