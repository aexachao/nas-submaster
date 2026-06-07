#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务队列页面
显示和管理任务
"""

import html
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

    _render_task_list()


@st.fragment(run_every=3)
def _render_task_list():
    """渲染任务列表（处理中时每 3 秒自动刷新）"""
    try:
        tasks = TaskDAO.get_all_tasks()
    except Exception as e:
        st.error(f"加载任务列表失败: {e}")
        return

    if not tasks:
        st.info("队列为空")
        return

    # 用 enumerate 提供稳定的位置索引，作为 widget key 前缀
    # 避免 fragment 在多次 run_every 之间的 list 顺序变化时，
    # 旧 widget 残留与新 widget 同 key 冲突（StreamlitDuplicateElementKey）
    for idx, task in enumerate(tasks):
        _render_task_card(task, idx)


def _render_task_card(task, idx: int):
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

    file_name = html.escape(Path(task.file_path).name)
    log_text = html.escape(task.log)
    created_at = html.escape(str(task.created_at or ''))
    status_text_escaped = html.escape(status_text)

    # v1.8.1: 步骤条 HTML (代替旧版单行进度条)
    progress_html = build_stage_progress_html(task)

    html_content = f"""<div class="task-card-wrapper"><div class="hero-card"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div style="flex:1;"><div style="font-weight:600; margin-bottom:8px;">{file_name}</div><div style="font-size:13px; color:#a1a1aa;">&gt; {log_text}</div></div><div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px; margin-left:16px;"><span style="font-size:11px; color:#71717a;">{created_at}</span><span class="status-chip {css_class}">{status_text_escaped}</span></div></div>{progress_html}</div></div>"""

    st.markdown(html_content, unsafe_allow_html=True)

    # 历史日志（有内容时展示可折叠区域）
    if task.log_history:
        with st.expander("查看执行日志", expanded=False):
            st.code(task.log_history, language=None)

    # 操作按钮（使用独立的列）
    # 关键：每个分支的 widget key 必须完全唯一
    #  - 用 idx 前缀应对 list 顺序变化（fragment run_every 跨 tick）
    #  - 在 key 里嵌入 status 名，避免同一 task 状态切换时跨分支 key 冲突
    #    （fragment 内部不同分支若共用 'del_{id}'，旧 widget 残留会冲突）
    col_space, col_ops = st.columns([8, 2])

    with col_ops:
        if task.status == TaskStatus.FAILED:
            # 失败任务：重试 + 删除
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("重试", key=f"t{idx}_{task.id}_failed_retry", use_container_width=True):
                    TaskDAO.reset_task(task.id)
                    st.rerun()
            with subcol2:
                if st.button("删除", key=f"t{idx}_{task.id}_failed_del", use_container_width=True):
                    TaskDAO.delete_task(task.id)
                    st.rerun()
        elif task.status == TaskStatus.PROCESSING:
            # 处理中：取消 + 删除
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("取消", key=f"t{idx}_{task.id}_proc_cancel", use_container_width=True):
                    TaskDAO.cancel_task(task.id)
                    worker = get_worker()
                    if worker:
                        worker.request_cancel()
                    st.rerun()
            with subcol2:
                if st.button("删除", key=f"t{idx}_{task.id}_proc_del", use_container_width=True):
                    TaskDAO.delete_task(task.id)
                    st.rerun()
        else:
            # 其他状态（等待中、已取消、已完成）：仅删除
            if st.button("删除", key=f"t{idx}_{task.id}_{task.status.value}_del", use_container_width=True):
                TaskDAO.delete_task(task.id)
                st.rerun()


# ============================================================================
# 步骤条渲染 (v1.8.1) — 抽离成纯函数方便单测
# ============================================================================

# 阶段显示顺序（与 task.lifecycle 严格匹配）
STAGE_ORDER = ("download", "extract", "translate")

# 阶段中文标签
STAGE_LABELS = {
    "download": "下载",
    "extract": "提取",
    "translate": "翻译",
}

# 状态色：已完成 / 进行中 / 未开始
STATUS_DONE = "done"
STATUS_ACTIVE = "active"
STATUS_PENDING = "pending"


def _stage_state(stage: str, current_stage: str) -> str:
    """
    判断 stage 相对 current_stage 的状态。

    - current_stage 之前的 → done
    - current_stage 本身 → active
    - current_stage 之后的 → pending

    边界：
    - current_stage='completed' → 全部 done
    - current_stage='failed'/'cancelled' → 保留最后 stage 状态
    """
    if current_stage == "completed":
        return STATUS_DONE
    try:
        cur_idx = STAGE_ORDER.index(current_stage)
    except ValueError:
        # failed/cancelled/pending 等异常 stage
        return STATUS_PENDING
    try:
        idx = STAGE_ORDER.index(stage)
    except ValueError:
        return STATUS_PENDING
    if idx < cur_idx:
        return STATUS_DONE
    if idx == cur_idx:
        return STATUS_ACTIVE
    return STATUS_PENDING


def _format_stage_progress(stage: str, stage_progress, current_stage: str) -> str:
    """
    格式化阶段的进度文字。

    - download/extract: "47.32%" (两位小数)
    - translate: "已翻译 47 条" (整数 = current_line)
    - done: "100%"
    """
    if stage_progress is None:
        return ""
    if stage == "translate":
        # stage_progress 是 current_line（整数）
        try:
            return f"已翻译 {int(stage_progress)} 条"
        except (TypeError, ValueError):
            return ""
    # download / extract：两位小数
    try:
        return f"{float(stage_progress):.2f}%"
    except (TypeError, ValueError):
        return ""


def _status_marker(state: str) -> str:
    """返回状态的视觉标识符"""
    return {
        STATUS_DONE: "✓",
        STATUS_ACTIVE: "⏳",
        STATUS_PENDING: "○",
    }.get(state, "○")


def _status_color(state: str) -> str:
    """返回状态对应的颜色"""
    return {
        STATUS_DONE: "#10b981",      # 绿
        STATUS_ACTIVE: "#2563eb",    # 蓝
        STATUS_PENDING: "#52525b",   # 灰
    }.get(state, "#52525b")


def build_stage_progress_html(task) -> str:
    """
    渲染任务的多阶段步骤条 HTML（v1.8.1）。

    Args:
        task: Task 数据类实例

    Returns:
        完整的 HTML 片段（CSS 内联，可直接 st.markdown）

    行为：
    - pending/无 stage 数据 → fallback 旧版 progress 字段单行进度条
    - processing/completed/failed → 渲染三阶段步骤条
    """
    # Fallback：旧 DB 没有 stage 字段
    if not task.stage or task.stage == "pending" and task.status == TaskStatus.PENDING:
        # 旧版单行进度条
        if task.status == TaskStatus.PROCESSING:
            return (
                f'<div style="margin-top:12px; margin-bottom:8px;">'
                f'<div style="width:100%; height:4px; background-color:#27272a; border-radius:2px; overflow:hidden;">'
                f'<div style="width:{task.progress}%; height:100%; background-color:#2563eb; transition:width 0.3s;"></div>'
                f'</div>'
                f'<div style="font-size:11px; color:#71717a; margin-top:4px; text-align:right;">{task.progress}%</div>'
                f'</div>'
            )
        return ""

    # 阶段列表（仅显示有意义的阶段；未来可加 export）
    current_stage = task.stage

    # 构建每个阶段的 HTML
    stage_chips = []
    for s in STAGE_ORDER:
        state = _stage_state(s, current_stage)
        marker = _status_marker(state)
        color = _status_color(state)
        label = STAGE_LABELS[s]

        # 当前阶段的进度文字
        progress_text = ""
        if state == STATUS_DONE:
            progress_text = "100%"
        elif state == STATUS_ACTIVE:
            progress_text = _format_stage_progress(s, task.stage_progress, current_stage)

        chip = (
            f'<div style="display:inline-flex; align-items:center; gap:6px; '
            f'padding:4px 10px; background-color:{color}22; border:1px solid {color}; '
            f'border-radius:12px; font-size:12px;">'
            f'<span style="color:{color}; font-weight:bold;">{marker}</span>'
            f'<span style="color:#e4e4e7;">{label}</span>'
            f'<span style="color:{color}; font-size:11px;">{html.escape(progress_text)}</span>'
            f'</div>'
        )
        stage_chips.append(chip)

    # 用箭头连接
    chips_html = '<span style="color:#52525b; margin:0 6px;">→</span>'.join(stage_chips)

    return (
        f'<div style="margin-top:12px; margin-bottom:8px; display:flex; flex-wrap:wrap; gap:4px; align-items:center;">'
        f'{chips_html}'
        f'</div>'
    )
