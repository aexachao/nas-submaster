#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
媒体文件数据访问对象（DAO）
负责媒体文件相关的数据库操作
"""

import json
from typing import List, Optional, Tuple

from database.connection import get_db_connection, execute_many
from core.models import MediaFile, SubtitleInfo


class MediaDAO:
    """媒体文件数据访问对象"""
    
    @staticmethod
    def get_all_media_files() -> List[MediaFile]:
        """获取所有媒体文件"""
        return MediaDAO.get_media_files_filtered(None)

    @staticmethod
    def _build_where_clause(has_subtitle: Optional[bool]) -> Tuple[str, list]:
        """构建 WHERE 子句和参数列表"""
        if has_subtitle is True:
            return " WHERE json_array_length(subtitles_json) > 0", []
        elif has_subtitle is False:
            return " WHERE subtitles_json IS NULL OR json_array_length(subtitles_json) = 0", []
        return "", []

    @staticmethod
    def get_media_files_count(has_subtitle: Optional[bool] = None) -> int:
        """
        获取筛选后的媒体文件总数（用于分页）

        Args:
            has_subtitle: 是否有字幕（None=全部, True=有字幕, False=无字幕）

        Returns:
            文件数量
        """
        conn = get_db_connection()
        try:
            where, params = MediaDAO._build_where_clause(has_subtitle)
            result = conn.execute(
                f"SELECT COUNT(*) FROM media_files{where}", params
            ).fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    @staticmethod
    def get_media_files_filtered(
        has_subtitle: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[MediaFile]:
        """
        获取筛选后的媒体文件（筛选条件下推到 SQL，避免全表加载后内存过滤）

        Args:
            has_subtitle: 是否有字幕（None=全部, True=有字幕, False=无字幕）
            limit: 返回数量上限（None=全部）
            offset: 跳过的行数（用于分页）

        Returns:
            媒体文件列表
        """
        conn = get_db_connection()
        try:
            base = (
                "SELECT id, file_path, file_name, file_size, subtitles_json, "
                "has_translated, updated_at FROM media_files"
            )
            where, params = MediaDAO._build_where_clause(has_subtitle)
            query = base + where + " ORDER BY file_name"
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params = params + [limit, offset]

            cursor = conn.execute(query, params)
            media_files = []
            for row in cursor.fetchall():
                try:
                    media = MediaFile(
                        id=row[0],
                        file_path=row[1],
                        file_name=row[2],
                        file_size=row[3],
                        subtitles=MediaDAO._parse_subtitles(row[4]),
                        has_translated=bool(row[5]),
                        updated_at=row[6]
                    )
                    media_files.append(media)
                except Exception as e:
                    print(f"[MediaDAO] Failed to parse media file {row[0]}: {e}")
                    continue
            return media_files
        finally:
            conn.close()
    
    @staticmethod
    def get_media_by_path(file_path: str) -> Optional[MediaFile]:
        """
        根据文件路径获取媒体文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            媒体文件对象，如果不存在则返回 None
        """
        conn = get_db_connection()
        try:
            result = conn.execute(
                "SELECT id, file_path, file_name, file_size, subtitles_json, "
                "has_translated, updated_at FROM media_files WHERE file_path=?",
                (file_path,)
            ).fetchone()
            
            if not result:
                return None
            
            return MediaFile(
                id=result[0],
                file_path=result[1],
                file_name=result[2],
                file_size=result[3],
                subtitles=MediaDAO._parse_subtitles(result[4]),
                has_translated=bool(result[5]),
                updated_at=result[6]
            )
        finally:
            conn.close()
    
    @staticmethod
    def add_or_update_media_file(
        file_path: str,
        file_name: str,
        file_size: int,
        subtitles: List[SubtitleInfo],
        has_translated: bool = False
    ):
        """
        添加或更新媒体文件
        
        Args:
            file_path: 文件路径
            file_name: 文件名
            file_size: 文件大小
            subtitles: 字幕列表
            has_translated: 是否有翻译
        """
        conn = get_db_connection()
        try:
            subtitles_json = json.dumps(
                [s.to_dict() for s in subtitles],
                ensure_ascii=False
            )
            
            conn.execute(
                "INSERT OR REPLACE INTO media_files "
                "(file_path, file_name, file_size, subtitles_json, has_translated, updated_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (file_path, file_name, file_size, subtitles_json, int(has_translated))
            )
            conn.commit()
        except Exception as e:
            print(f"[MediaDAO] Failed to add/update media file: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def batch_add_or_update_media_files(media_files: List[tuple]):
        """
        批量添加或更新媒体文件
        
        Args:
            media_files: 元组列表 [(file_path, file_name, file_size, subtitles_json, has_translated), ...]
        """
        try:
            execute_many(
                "INSERT OR REPLACE INTO media_files "
                "(file_path, file_name, file_size, subtitles_json, has_translated, updated_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                media_files
            )
        except Exception as e:
            print(f"[MediaDAO] Failed to batch add/update media files: {e}")
            raise
    
    @staticmethod
    def update_media_subtitles(
        file_path: str,
        subtitles: List[SubtitleInfo],
        has_translated: bool
    ):
        """
        更新媒体文件的字幕信息
        
        Args:
            file_path: 文件路径
            subtitles: 字幕列表
            has_translated: 是否有翻译
        """
        conn = get_db_connection()
        try:
            subtitles_json = json.dumps(
                [s.to_dict() for s in subtitles],
                ensure_ascii=False
            )
            
            conn.execute(
                "UPDATE media_files SET subtitles_json=?, has_translated=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE file_path=?",
                (subtitles_json, int(has_translated), file_path)
            )
            conn.commit()
        except Exception as e:
            print(f"[MediaDAO] Failed to update media subtitles: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def delete_media_file(file_path: str):
        """
        删除媒体文件记录
        
        Args:
            file_path: 文件路径
        """
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM media_files WHERE file_path=?", (file_path,))
            conn.commit()
        except Exception as e:
            print(f"[MediaDAO] Failed to delete media file: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    @staticmethod
    def get_media_count() -> int:
        """
        获取媒体文件总数
        
        Returns:
            文件数量
        """
        conn = get_db_connection()
        try:
            result = conn.execute("SELECT COUNT(*) FROM media_files").fetchone()
            return result[0] if result else 0
        finally:
            conn.close()
    
    @staticmethod
    def _parse_subtitles(subtitles_json: str) -> List[SubtitleInfo]:
        """
        解析字幕 JSON
        
        Args:
            subtitles_json: JSON 字符串
        
        Returns:
            字幕信息列表
        """
        try:
            data = json.loads(subtitles_json)
            return [SubtitleInfo.from_dict(s) for s in data]
        except Exception as e:
            print(f"[MediaDAO] Failed to parse subtitles JSON: {e}")
            return []