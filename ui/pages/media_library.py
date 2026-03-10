#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
媒体库页面
"""

import html
import time
from pathlib import Path
from typing import Optional
import streamlit as st

from database.media_dao import MediaDAO
from database.task_dao import TaskDAO
from services.media_scanner import (
    scan_media_directory,
    discover_media_subdirectories,
    rescan_video_subtitles,
    MEDIA_ROOT
)
from utils.format_utils import format_file_size

PAGE_SIZE_OPTIONS = [25, 50, 100]
DEFAULT_PAGE_SIZE = 50


def render_media_library_page(debug_mode: bool = False):
    """渲染媒体库页面"""

    # 顶部工具栏
    # 比例: 筛选(2.2) | 空白(1.3) | 目录选择(3) | 扫描(0.8) | 开始(0.8)
    col_filter, col_spacer, col_dir, col_scan, col_start = st.columns(
        [2.2, 1.3, 3, 0.8, 0.8], vertical_alignment="bottom"
    )

    # ========== 列 1: 筛选器 ==========
    with col_filter:
        filter_type = st.radio(
            "筛选",
            ["全部", "有字幕", "无字幕"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # ========== 列 2: 空白 ==========
    with col_spacer:
        st.empty()

    # ========== 列 3: 目录选择器 ==========
    with col_dir:
        if 'subdirs' not in st.session_state or st.session_state.get('refresh_subdirs'):
            with st.spinner("扫描目录结构..."):
                st.session_state.subdirs = discover_media_subdirectories(max_depth=3)
                st.session_state.refresh_subdirs = False

        subdirs = st.session_state.subdirs

        selected_dirs = st.multiselect(
            "选择目录",
            subdirs,
            placeholder="选择一个或多个目录 (留空显示全部)",
            label_visibility="collapsed"
        )

    # ========== 列 4: 扫描按钮 ==========
    with col_scan:
        refresh_text = "扫描全部" if not selected_dirs else f"扫描 ({len(selected_dirs)})"
        if st.button(refresh_text, use_container_width=True):
            _perform_scan(selected_dirs, debug_mode)

    filter_map = {"全部": None, "有字幕": True, "无字幕": False}
    has_subtitle_filter = filter_map[filter_type]

    # 筛选条件或目录变化时重置到第 1 页
    filter_key = f"{filter_type}|{','.join(selected_dirs)}"
    if st.session_state.get('_last_filter_key') != filter_key:
        st.session_state['_media_page'] = 0
        st.session_state['_last_filter_key'] = filter_key

    current_page = st.session_state.get('_media_page', 0)
    page_size = st.session_state.get('_media_page_size', DEFAULT_PAGE_SIZE)

    # ========== 加载文件 ==========
    # 有目录过滤时需要 Python 端过滤，加载全量后切片；否则走 SQL LIMIT/OFFSET
    try:
        if selected_dirs:
            all_files = MediaDAO.get_media_files_filtered(has_subtitle_filter)
            filtered_files = []
            for f in all_files:
                fpath = Path(f.file_path)
                for d in selected_dirs:
                    dir_path = Path(MEDIA_ROOT) / d
                    try:
                        fpath.relative_to(dir_path)
                        filtered_files.append(f)
                        break
                    except ValueError:
                        continue
            total_count = len(filtered_files)
            start = current_page * page_size
            page_files = filtered_files[start:start + page_size]
        else:
            total_count = MediaDAO.get_media_files_count(has_subtitle_filter)
            start = current_page * page_size
            page_files = MediaDAO.get_media_files_filtered(
                has_subtitle_filter, limit=page_size, offset=start
            )
    except Exception as e:
        st.error(f"加载媒体库失败: {e}")
        return

    # 维护 id→file_path 缓存，供跨页"开始处理"使用
    if '_id_to_path' not in st.session_state:
        st.session_state['_id_to_path'] = {}
    for f in page_files:
        st.session_state['_id_to_path'][f.id] = f.file_path

    # ========== 列 5: 开始按钮 ==========
    with col_start:
        selected_count = sum(
            1 for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith('s_') and v is True
        )
        btn_text = f"处理 ({selected_count})" if selected_count > 0 else "开始处理"
        if st.button(
            btn_text,
            type="primary",
            use_container_width=True,
            disabled=(selected_count == 0)
        ):
            _add_tasks_for_selected_files()

    # ========== 统计信息 ==========
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _render_statistics(total_count, selected_count, selected_dirs, filter_type)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ========== 空状态 ==========
    if total_count == 0:
        st.info(f"选中目录下暂无{filter_type}文件" if selected_dirs else "暂无文件，请先扫描媒体库")
        return

    # ========== 全选（作用于当前页）==========
    current_select_all = st.checkbox("全选（当前页）", key="select_all_box")
    last_select_all = st.session_state.get("_last_select_all", False)

    if current_select_all != last_select_all:
        for f in page_files:
            st.session_state[f"s_{f.id}"] = current_select_all
        st.session_state["_last_select_all"] = current_select_all
        st.rerun()

    # ========== 渲染当前页文件列表 ==========
    for f in page_files:
        _render_media_card(f)

    # ========== 分页控件 ==========
    _render_pagination(total_count, current_page, page_size)


def _render_pagination(total: int, current_page: int, page_size: int):
    """渲染分页控件"""
    total_pages = max(1, (total + page_size - 1) // page_size)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_prev, col_info, col_next, col_size = st.columns([1, 3, 1, 2])

    with col_prev:
        if st.button("← 上一页", use_container_width=True, disabled=(current_page == 0)):
            st.session_state['_media_page'] = current_page - 1
            st.session_state['_last_select_all'] = False
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px; font-size:13px; color:#71717a;'>"
            f"第 {current_page + 1} / {total_pages} 页 &nbsp;·&nbsp; 共 {total} 个文件"
            f"</div>",
            unsafe_allow_html=True
        )

    with col_next:
        if st.button("下一页 →", use_container_width=True, disabled=(current_page >= total_pages - 1)):
            st.session_state['_media_page'] = current_page + 1
            st.session_state['_last_select_all'] = False
            st.rerun()

    with col_size:
        new_size = st.selectbox(
            "每页",
            PAGE_SIZE_OPTIONS,
            index=PAGE_SIZE_OPTIONS.index(page_size) if page_size in PAGE_SIZE_OPTIONS else 1,
            label_visibility="collapsed",
            key="page_size_select"
        )
        if new_size != page_size:
            st.session_state['_media_page_size'] = new_size
            st.session_state['_media_page'] = 0
            st.rerun()


def _render_statistics(total: int, selected: int, selected_dirs: list, filter_type: str):
    """渲染统计信息栏"""
    info_parts = []

    if selected_dirs:
        if len(selected_dirs) == 1:
            d = selected_dirs[0]
            display = d if len(d) <= 30 else "..." + d[-27:]
            info_parts.append(f"`{display}`")
        else:
            info_parts.append(f"已选 {len(selected_dirs)} 个目录")
    else:
        info_parts.append("全部目录")

    info_parts.append(f"{filter_type}: **{total}** 个文件")

    if selected > 0:
        info_parts.append(f"已选: **{selected}** 个")

    st.caption(" | ".join(info_parts))


def _add_tasks_for_selected_files():
    """为选中的文件添加任务（从 session_state 缓存中读取）"""
    success_count = 0
    failed_files = []
    id_to_path = st.session_state.get('_id_to_path', {})

    for key, selected in list(st.session_state.items()):
        if not (isinstance(key, str) and key.startswith('s_') and selected is True):
            continue
        try:
            file_id = int(key[2:])
        except ValueError:
            continue
        file_path = id_to_path.get(file_id)
        if not file_path:
            continue
        ok, msg = TaskDAO.add_task(file_path)
        if ok:
            success_count += 1
        else:
            file_name = Path(file_path).name
            failed_files.append((file_name, msg))

    if failed_files:
        st.warning(f"已添加 {success_count} 个任务，{len(failed_files)} 个失败")
        for fname, reason in failed_files[:3]:
            st.caption(f"{fname}: {reason}")
    else:
        st.toast(f"已添加 {success_count} 个任务")

    time.sleep(1)
    st.rerun()


def _perform_scan(subdirectories: list, debug_mode: bool):
    """执行扫描操作"""
    with st.spinner("扫描中..."):
        total_cnt = 0
        all_logs = []

        dirs_to_scan = subdirectories if subdirectories else [None]

        for d in dirs_to_scan:
            cnt, logs = scan_media_directory(subdirectory=d, debug=debug_mode)
            total_cnt += cnt
            if logs:
                all_logs.extend(logs)

        st.toast(f"扫描完成，更新 {total_cnt} 个文件")

        if debug_mode and all_logs:
            with st.expander("调试日志", expanded=True):
                for log in all_logs[:20]:
                    st.text(log)

    st.session_state.refresh_subdirs = True
    st.session_state['_media_page'] = 0  # 扫描后回到第 1 页
    st.rerun()


def _render_media_card(media_file):
    """渲染单个媒体文件卡片"""
    if not media_file.subtitles:
        badges = "<span class='status-chip chip-red'>无字幕</span>"
    else:
        badges = ""
        for sub in media_file.subtitles:
            lang = sub.lang.lower()
            if lang in ['zh', 'chs', 'cht']:
                cls = "chip-green"
            elif lang in ['en', 'eng']:
                cls = "chip-blue"
            else:
                cls = "chip-gray"
            badges += f"<span class='status-chip {cls}'>{html.escape(sub.tag)}</span>"

    file_name = html.escape(media_file.file_name)
    file_path = html.escape(media_file.file_path)
    file_size = html.escape(format_file_size(media_file.file_size))

    # 布局：复选框 + 卡片 + 刷新按钮
    c_check, c_card, c_rescan = st.columns([0.5, 19, 1.5], gap="medium", vertical_alignment="center")

    with c_check:
        key = f"s_{media_file.id}"
        if key not in st.session_state:
            st.session_state[key] = False
        st.checkbox("选", key=key, label_visibility="collapsed")

    with c_card:
        st.markdown(
            f"""
            <div class="hero-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="font-weight:600; font-size:15px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">
                        {file_name}
                    </div>
                    <div style="font-size:12px; color:#71717a; min-width:60px; text-align:right;">
                        {file_size}
                    </div>
                </div>
                <div style="font-size:12px; color:#52525b; margin-bottom:12px; font-family:monospace;">
                    {file_path}
                </div>
                <div>{badges}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_rescan:
        if st.button(
            "↻",
            key=f"rescan_{media_file.id}",
            help="重新扫描该文件的字幕",
            use_container_width=True
        ):
            rescan_video_subtitles(media_file.file_path)
            st.toast(f"已刷新: {media_file.file_name}")
            st.rerun()
