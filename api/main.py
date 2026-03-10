#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAS 字幕管家 — FastAPI 主入口
负责：应用启动、Worker 初始化、静态文件托管
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database.connection import init_database
from core.worker import start_worker
from api.routers import media, tasks, settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库和 Worker"""
    os.makedirs("./data/models", exist_ok=True)
    init_database()
    start_worker()
    print("[Main] FastAPI started, worker running")
    yield
    print("[Main] FastAPI shutting down")


app = FastAPI(
    title="NAS 字幕管家",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

# 注册路由
app.include_router(media.router)
app.include_router(tasks.router)
app.include_router(settings.router)

# 托管前端静态文件（生产构建产物）
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        # API 路由已在前面注册，这里只处理前端路由
        index = static_dir / "index.html"
        return FileResponse(str(index))
