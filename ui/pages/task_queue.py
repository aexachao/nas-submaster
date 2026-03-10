#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务队列页面
显示和管理任务
"""

import time
from pathlib import Path
import streamlit as st

from database.task_dao import TaskDAO
from core.models import TaskStatus
from core.worker import get_worker


def render_task_queue_page():
    """渲染任务队列页面"""

    # 顶部工具栏
    col_space, col_clear = st.columns([8, 2])

    with col_clear:
        if st.button("清理记录", use_container_width=True):
            TaskDAO.clear_completed_tasks()
            st.rerun()

    # 加载任务列表
    tasks = TaskDAO.get_all_tasks()

    # 空状态
    if not tasks:
        st.info("队列为空")
        return

    # 检查是否有处理中的任务
    has_processing = any(t.status == TaskStatus.PROCESSING for t in tasks)

    # 渲染任务列表
    for task in tasks:
        _render_task_card(task)

    # 如果有处理中的任务，自动刷新
    if has_processing:
        time.sleep(3)
        st.rerun()


def _render_task_card(task):
    """渲染单个任务卡片"""

    # 状态映射
    status_map = {
        TaskStatus.PENDING: ('chip-gray', '等待中'),
        TaskStatus.PROCESSING: ('chip-blue', '处理中'),
        TaskStatus.COMPLETED: ('chip-green', '完成'),
        TaskStatus.FAILED: ('chip-red', '失败'),
        TaskStatus.CANCELLED: ('chip-gray', '已取消'),
    }

    css_class, status_text = status_map.get(
        task.status,
        ('chip-gray', task.status.value)
    )

    # 进度条 HTML (单行)
    progress_html = ""
    if task.status == TaskStatus.PROCESSING:
        progress_html = f"""<div style="margin-top:12px; margin-bottom:8px;"><div style="width:100%; height:4px; background-color:#27272a; border-radius:2px; overflow:hidden;"><div style="width:{task.progress}%; height:100%; background-color:#2563eb; transition:width 0.3s;"></div></div><div style="font-size:11px; color:#71717a; margin-top:4px; text-align:right;">{task.progress}%</div></div>"""

    html_content = f"""<div class="task-card-wrapper"><div class="hero-card"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div style="flex:1;"><div style="font-weight:600; margin-bottom:8px;">{Path(task.file_path).name}</div><div style="font-size:13px; color:#a1a1aa;">> {task.log}</div></div><div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px; margin-left:16px;"><span style="font-size:11px; color:#71717a;">{task.created_at}</span><span class="status-chip {css_class}">{status_text}</span></div></div>{progress_html}</div></div>"""

    st.markdown(html_content, unsafe_allow_html=True)

    # 历史日志（有内容时展示可折叠区域）
    if task.log_history:
        with st.expander("查看执行日志", expanded=False):
            st.code(task.log_history, language=None)

    # 操作按钮（使用独立的列）
    col_space, col_ops = st.columns([8, 2])

    with col_ops:
        if task.status == TaskStatus.FAILED:
            # 失败任务：重试 + 删除
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("重试", key=f"retry_{task.id}", use_container_width=True):
                    TaskDAO.reset_task(task.id)
                    st.rerun()
            with subcol2:
                if st.button("删除", key=f"del_{task.id}", use_container_width=True):
                    TaskDAO.delete_task(task.id)
                    st.rerun()
        elif task.status == TaskStatus.PROCESSING:
            # 处理中：取消 + 删除
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("取消", key=f"cancel_{task.id}", use_container_width=True):
                    TaskDAO.cancel_task(task.id)
                    worker = get_worker()
                    if worker:
                        worker.request_cancel()
                    st.rerun()
            with subcol2:
                if st.button("删除", key=f"del_{task.id}", use_container_width=True):
                    TaskDAO.delete_task(task.id)
                    st.rerun()
        else:
            # 其他状态（等待中、已取消、已完成）：仅删除
            if st.button("删除", key=f"del_{task.id}", use_container_width=True):
                TaskDAO.delete_task(task.id)
                st.rerun()
