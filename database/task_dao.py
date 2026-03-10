#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务数据访问对象（DAO）
负责任务相关的数据库操作
"""

import sqlite3
from typing import List, Optional, Tuple

from database.connection import get_db_connection
from core.models import Task, TaskStatus


class TaskDAO:
    """任务数据访问对象"""
    
    @staticmethod
    def add_task(file_path: str) -> Tuple[bool, str]:
        """
        添加新任务
        
        Args:
            file_path: 文件路径
        
        Returns:
            (成功标志, 消息)
        """
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
            print(f"[TaskDAO] Failed to add task: {e}")
            return False, f"添加失败: {str(e)}"
        finally:
            conn.close()
    
    @staticmethod
    def get_all_tasks() -> List[Task]:
        """
        获取所有任务
        
        Returns:
            任务列表
        """
        conn = get_db_connection()
        try:
            cursor = conn.execute(
                "SELECT id, file_path, status, progress, log, created_at, updated_at "
                "FROM tasks ORDER BY id DESC"
            )
            
            tasks = []
            for row in cursor.fetchall():
                try:
                    task = Task(
                        id=row[0],
                        file_path=row[1],
                        status=TaskStatus(row[2]),
                        progress=row[3],
                        log=row[4],
                        created_at=row[5],
                        updated_at=row[6]
                    )
                    tasks.append(task)
                except Exception as e:
                    print(f"[TaskDAO] Failed to parse task {row[0]}: {e}")
                    continue
            
            return tasks
        finally:
            conn.close()
    
    @staticmethod
    def get_pending_task() -> Optional[Task]:
        """
        获取第一个待处理任务
        
        Returns:
            任务对象，如果没有则返回 None
        """
        conn = get_db_connection()
        try:
            result = conn.execute(
                "SELECT id, file_path, status, progress, log, created_at, updated_at "
                "FROM tasks WHERE status='pending' LIMIT 1"
            ).fetchone()
            
            if not result:
                return None
            
            return Task(
                id=result[0],
                file_path=result[1],
                status=TaskStatus(result[2]),
                progress=result[3],
                log=result[4],
                created_at=result[5],
                updated_at=result[6]
            )
        finally:
            conn.close()
    
    @staticmethod
    def get_task_by_id(task_id: int) -> Optional[Task]:
        """
        根据 ID 获取任务
        
        Args:
            task_id: 任务 ID
        
        Returns:
            任务对象，如果不存在则返回 None
        """
        conn = get_db_connection()
        try:
            result = conn.execute(
                "SELECT id, file_path, status, progress, log, created_at, updated_at "
                "FROM tasks WHERE id=?",
                (task_id,)
            ).fetchone()
            
            if not result:
                return None
            
            return Task(
                id=result[0],
                file_path=result[1],
                status=TaskStatus(result[2]),
                progress=result[3],
                log=result[4],
                created_at=result[5],
                updated_at=result[6]
            )
        finally:
            conn.close()
    
    @staticmethod
    def update_task(
        task_id: int,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        log: Optional[str] = None
    ):
        """
        更新任务状态
        
        Args:
            task_id: 任务 ID
            status: 新状态（可选）
            progress: 进度（可选）
            log: 日志（可选）
        """
        conn = get_db_connection()
        try:
            updates = []
            params = []
            
            if status is not None:
                updates.append("status=?")
                params.append(status.value if isinstance(status, TaskStatus) else status)
            
            if progress is not None:
                updates.append("progress=?")
                params.append(progress)
            
            if log is not None:
                updates.append("log=?")
                params.append(log)
            
            if not updates:
                return
            
            updates.append("updated_at=CURRENT_TIMESTAMP")
            params.append(task_id)
            
            query = f"UPDATE tasks SET {','.join(updates)} WHERE id=?"
            conn.execute(query, params)
            conn.commit()
            
        except Exception as e:
            print(f"[TaskDAO] Failed to update task {task_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def delete_task(task_id: int):
        """
        删除任务
        
        Args:
            task_id: 任务 ID
        """
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()
        except Exception as e:
            print(f"[TaskDAO] Failed to delete task {task_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def clear_completed_tasks():
        """删除所有已完成和失败的任务"""
        conn = get_db_connection()
        try:
            conn.execute(
                "DELETE FROM tasks WHERE status IN ('completed', 'failed')"
            )
            conn.commit()
        except Exception as e:
            print(f"[TaskDAO] Failed to clear completed tasks: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def reset_task(task_id: int):
        """
        重置任务为待处理状态
        
        Args:
            task_id: 任务 ID
        """
        conn = get_db_connection()
        try:
            conn.execute(
                "UPDATE tasks SET status='pending', progress=0, log='重试中...', "
                "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (task_id,)
            )
            conn.commit()
        except Exception as e:
            print(f"[TaskDAO] Failed to reset task {task_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def get_task_count_by_status(status: TaskStatus) -> int:
        """
        获取指定状态的任务数量
        
        Args:
            status: 任务状态
        
        Returns:
            任务数量
        """
        conn = get_db_connection()
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status=?",
                (status.value,)
            ).fetchone()
            return result[0] if result else 0
        finally:
            conn.close()
    
    @staticmethod
    def has_processing_task() -> bool:
        """
        检查是否有正在处理的任务

        Returns:
            bool: 是否有处理中的任务
        """
        count = TaskDAO.get_task_count_by_status(TaskStatus.PROCESSING)
        return count > 0

    @staticmethod
    def reset_stale_processing_tasks():
        """
        将遗留的 PROCESSING 状态任务重置为 PENDING
        用于 Worker 启动时处理上次崩溃留下的死锁任务
        """
        conn = get_db_connection()
        try:
            cursor = conn.execute(
                "UPDATE tasks SET status='pending', progress=0, log='重启后自动重置', "
                "updated_at=CURRENT_TIMESTAMP WHERE status='processing'"
            )
            if cursor.rowcount > 0:
                print(f"[TaskDAO] Reset {cursor.rowcount} stale processing task(s) to pending")
            conn.commit()
        except Exception as e:
            print(f"[TaskDAO] Failed to reset stale tasks: {e}")
            conn.rollback()
        finally:
            conn.close()