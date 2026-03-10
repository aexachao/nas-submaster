#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
媒体库页面（优化版）
- 移除快速扫描按钮
- 优化多层目录显示
"""

import time
from pathlib import Path
from typing import Optional
import streamlit as st

from database.media_dao import MediaDAO
from database.task_dao import TaskDAO
from services.media_scanner import (
    scan_media_directory,
    discover_media_subdirectories,
    MEDIA_ROOT
)
from utils.format_utils import format_file_size


def render_media_library_page(debug_mode: bool = False):
    """渲染媒体库页面"""
    
    # 顶部工具栏 - 布局调整
    # 比例: 筛选(2.2) | 空白(1.3) | 目录选择(3) | 扫描(0.8) | 开始(0.8)
    col_filter, col_spacer, col_dir, col_scan, col_start = st.columns([2.2, 1.3, 3, 0.8, 0.8], vertical_alignment="bottom")
    
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
        # 获取子目录列表（使用缓存）
        if 'subdirs' not in st.session_state or st.session_state.get('refresh_subdirs'):
            with st.spinner("扫描目录结构..."):
                st.session_state.subdirs = discover_media_subdirectories(max_depth=3)
                st.session_state.refresh_subdirs = False
        
        subdirs = st.session_state.subdirs
        
        # 目录多选框
        selected_dirs = st.multiselect(
            "选择目录",
            subdirs,
            placeholder="选择一个或多个目录 (留空显示全部)",
            label_visibility="collapsed"
        )

    # ========== 列 4: 扫描按钮 ==========
    with col_scan:
        # 刷新按钮（去掉 emoji）
        if not selected_dirs:
            refresh_text = "扫描全部"
        else:
            refresh_text = f"扫描 ({len(selected_dirs)})"
        
        if st.button(refresh_text, use_container_width=True):
            _perform_scan(selected_dirs, debug_mode)
            
    # ========== 列 5: 开始按钮 ==========
    with col_start:
        # 加载媒体文件
        filter_map = {
            "全部": None,
            "有字幕": True,
            "无字幕": False
        }
        
        files = MediaDAO.get_media_files_filtered(filter_map[filter_type])
        
        # 如果选择了子目录，使用路径前缀精确过滤，避免目录名子串误匹配
        if selected_dirs:
            filtered_files = []
            for f in files:
                fpath = Path(f.file_path)
                for d in selected_dirs:
                    dir_path = Path(MEDIA_ROOT) / d
                    try:
                        fpath.relative_to(dir_path)
                        filtered_files.append(f)
                        break
                    except ValueError:
                        continue
            files = filtered_files
        
        # 统计选中文件
        selected_count = sum(
            1 for f in files if st.session_state.get(f"s_{f.id}", False)
        )
        
        # 开始处理按钮（去掉 emoji）
        if selected_count > 0:
            btn_text = f"处理 ({selected_count})"
            btn_disabled = False
        else:
            btn_text = "开始处理"
            btn_disabled = True
        
        if st.button(
            btn_text,
            type="primary",
            use_container_width=True,
            disabled=btn_disabled
        ):
            _add_tasks_for_selected_files(files)
    
    # ========== 显示统计信息 ==========
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _render_statistics(len(files), selected_count, selected_dirs, filter_type)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    # ========== 空状态 ==========
    if not files:
        if selected_dirs:
            st.info(f"选中目录下暂无{filter_type}文件")
        else:
            st.info("暂无文件，请先扫描媒体库")
        return
    
    # ========== 全选功能 ==========
    current_select_all = st.checkbox("全选", key="select_all_box")  # 去掉 emoji
    last_select_all = st.session_state.get("_last_select_all", False)
    
    if current_select_all != last_select_all:
        for f in files:
            st.session_state[f"s_{f.id}"] = current_select_all
        st.session_state["_last_select_all"] = current_select_all
        st.rerun()
    
    # ========== 渲染文件列表 ==========
    for f in files:
        _render_media_card(f)


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


def _add_tasks_for_selected_files(files: list):
    """为选中的文件添加任务"""
    success_count = 0
    failed_files = []
    
    for f in files:
        if st.session_state.get(f"s_{f.id}", False):
            ok, msg = TaskDAO.add_task(f.file_path)
            if ok:
                success_count += 1
            else:
                failed_files.append((f.file_name, msg))
    
    # 显示结果
    if failed_files:
        st.warning(f"已添加 {success_count} 个任务， {len(failed_files)} 个失败")
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
        
        # 如果未选择子目录，则扫描根目录
        dirs_to_scan = subdirectories if subdirectories else [None]
        
        for d in dirs_to_scan:
            cnt, logs = scan_media_directory(
                subdirectory=d,
                debug=debug_mode
            )
            total_cnt += cnt
            if logs:
                all_logs.extend(logs)
        
        st.toast(f"扫描完成，更新 {total_cnt} 个文件")
        
        if debug_mode and all_logs:
            with st.expander("调试日志", expanded=True):
                for log in all_logs[:20]:
                    st.text(log)
    
    # 刷新目录列表
    st.session_state.refresh_subdirs = True
    st.rerun()


def _render_media_card(media_file):
    """渲染单个媒体文件卡片"""
    # 构建字幕徽章
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
            badges += f"<span class='status-chip {cls}'>{sub.tag}</span>"
    
    # 布局：复选框 + 卡片
    c_check, c_card = st.columns([0.5, 20], gap="medium", vertical_alignment="center")
    
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
                        {media_file.file_name}
                    </div>
                    <div style="font-size:12px; color:#71717a; min-width:60px; text-align:right;">
                        {format_file_size(media_file.file_size)}
                    </div>
                </div>
                <div style="font-size:12px; color:#52525b; margin-bottom:12px; font-family:monospace;">
                    {media_file.file_path}
                </div>
                <div>{badges}</div>
            </div>
            """,
            unsafe_allow_html=True
        )