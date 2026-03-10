#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
媒体库路由
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException

from database.media_dao import MediaDAO
from database.task_dao import TaskDAO
from services.media_scanner import (
    scan_media_directory,
    discover_media_subdirectories,
    rescan_video_subtitles,
    MEDIA_ROOT,
)
from utils.format_utils import format_file_size
from api.schemas import (
    MediaFileSchema,
    SubtitleInfoSchema,
    MediaListResponse,
    ScanRequest,
    ScanResponse,
)

router = APIRouter(prefix="/api/media", tags=["media"])


def _to_schema(f) -> MediaFileSchema:
    return MediaFileSchema(
        id=f.id,
        file_path=f.file_path,
        file_name=f.file_name,
        file_size=f.file_size,
        file_size_display=format_file_size(f.file_size),
        subtitles=[SubtitleInfoSchema(path=s.path, lang=s.lang, tag=s.tag) for s in f.subtitles],
        has_subtitle=f.has_subtitle,
        has_translated=f.has_translated,
        updated_at=f.updated_at,
    )


def _filter_by_dirs(files: list, dirs: List[str]) -> list:
    if not dirs:
        return files
    result = []
    for f in files:
        fpath = Path(f.file_path)
        for d in dirs:
            dir_path = Path(MEDIA_ROOT) / d
            try:
                fpath.relative_to(dir_path)
                result.append(f)
                break
            except ValueError:
                continue
    return result


@router.get("", response_model=MediaListResponse)
def list_media(
    filter: Optional[str] = Query("all", description="all | has | no"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    dirs: List[str] = Query(default=[]),
):
    has_subtitle: Optional[bool] = None
    if filter == "has":
        has_subtitle = True
    elif filter == "no":
        has_subtitle = False

    offset = (page - 1) * page_size

    # 有目录过滤时全量加载后 Python 侧过滤
    if dirs:
        all_files = MediaDAO.get_media_files_filtered(has_subtitle)
        filtered = _filter_by_dirs(all_files, dirs)
        total = len(filtered)
        page_files = filtered[offset: offset + page_size]
    else:
        total = MediaDAO.get_media_files_count(has_subtitle)
        page_files = MediaDAO.get_media_files_filtered(has_subtitle, limit=page_size, offset=offset)

    import math
    total_pages = max(1, math.ceil(total / page_size))

    return MediaListResponse(
        items=[_to_schema(f) for f in page_files],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/subdirs")
def get_subdirs():
    return {"subdirs": discover_media_subdirectories(max_depth=3)}


@router.post("/scan", response_model=ScanResponse)
def scan_media(body: ScanRequest):
    total = 0
    dirs_to_scan = body.dirs if body.dirs else [None]
    for d in dirs_to_scan:
        cnt, _ = scan_media_directory(subdirectory=d)
        total += cnt
    return ScanResponse(updated=total)


@router.post("/{media_id}/rescan")
def rescan_media(media_id: int):
    from database.media_dao import MediaDAO
    conn = __import__('database.connection', fromlist=['get_db_connection']).get_db_connection()
    try:
        row = conn.execute(
            "SELECT file_path FROM media_files WHERE id=?", (media_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Media file not found")
    rescan_video_subtitles(row[0])
    return {"ok": True}


@router.post("/add-tasks")
def add_tasks_for_files(body: dict):
    """从媒体库直接提交任务（接收 file_paths 列表）"""
    file_paths = body.get("file_paths", [])
    success = 0
    failed = []
    for fp in file_paths:
        ok, msg = TaskDAO.add_task(fp)
        if ok:
            success += 1
        else:
            failed.append({"file": Path(fp).name, "reason": msg})
    return {"success": success, "failed": failed}
