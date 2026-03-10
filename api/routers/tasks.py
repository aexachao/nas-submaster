#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务队列路由 + SSE 实时推送
"""

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from database.task_dao import TaskDAO
from core.worker import get_worker
from api.schemas import TaskSchema, AddTasksRequest, AddTasksResponse

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _to_schema(t) -> TaskSchema:
    return TaskSchema(
        id=t.id,
        file_path=t.file_path,
        file_name=Path(t.file_path).name,
        status=t.status.value,
        progress=t.progress,
        log=t.log,
        log_history=t.log_history,
        created_at=str(t.created_at) if t.created_at else None,
        updated_at=str(t.updated_at) if t.updated_at else None,
    )


@router.get("", response_model=list[TaskSchema])
def list_tasks():
    return [_to_schema(t) for t in TaskDAO.get_all_tasks()]


@router.post("", response_model=AddTasksResponse)
def add_tasks(body: AddTasksRequest):
    success = 0
    failed = []
    for fp in body.file_paths:
        ok, msg = TaskDAO.add_task(fp)
        if ok:
            success += 1
        else:
            failed.append({"file": Path(fp).name, "reason": msg})
    return AddTasksResponse(success=success, failed=failed)


@router.delete("/completed")
def clear_completed():
    TaskDAO.clear_completed_tasks()
    return {"ok": True}


@router.post("/{task_id}/cancel")
def cancel_task(task_id: int):
    TaskDAO.cancel_task(task_id)
    worker = get_worker()
    if worker:
        worker.request_cancel()
    return {"ok": True}


@router.post("/{task_id}/retry")
def retry_task(task_id: int):
    TaskDAO.reset_task(task_id)
    return {"ok": True}


@router.delete("/{task_id}")
def delete_task(task_id: int):
    TaskDAO.delete_task(task_id)
    return {"ok": True}


@router.get("/stream")
async def task_stream():
    """SSE：每 2 秒推送最新任务列表"""

    async def generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                tasks = TaskDAO.get_all_tasks()
                data = json.dumps([_to_schema(t).model_dump() for t in tasks], ensure_ascii=False)
                yield f"data: {data}\n\n"
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
