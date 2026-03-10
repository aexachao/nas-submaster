#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置弹窗 UI
使用 st.dialog 和 st.tabs 重构原侧边栏配置
"""

import streamlit as st
import requests
from typing import List, Tuple

from core.config import (
    ConfigManager,
    LLM_PROVIDERS,
    get_content_type_display_name,
    get_content_type_description
)
from core.models import ContentType, ISO_LANG_MAP, TARGET_LANG_OPTIONS, WHISPER_SOURCE_LANG_MAP
from database.connection import get_db_connection


# ============================================================================
# 辅助函数 (原 sidebar.py)
# ============================================================================

def test_api_connection(api_key: str, base_url: str, model: str) -> Tuple[bool, str]:
    """测试 API 连接 (10秒超时)"""
    import concurrent.futures
    
    def _do_test():
        try:
            from services.translator import TranslationConfig, SubtitleTranslator, SubtitleEntry
            
            config = TranslationConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=model,
                target_language='zh'
            )
            translator = SubtitleTranslator(config)
            
            # 简单测试
            test_entry = SubtitleEntry("1", "00:00:00,000 --> 00:00:01,000", "Hello")
            translator._translate_batch([test_entry])
            
            return True, "连接成功"
        except Exception as e:
            return False, str(e)
    
    # 使用 10 秒超时
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_test)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            return False, "连接超时 (10秒)"


def fetch_ollama_models(base_url: str) -> List[str]:
    """获取 Ollama 模型列表"""
    try:
        root_url = base_url.replace("/v1", "").rstrip("/")
        resp = requests.get(f"{root_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return [m['name'] for m in resp.json().get('models', [])]
    except Exception as e:
        print(f"[Settings] Failed to fetch Ollama models: {e}")
    return []


# ============================================================================
# 设置组件渲染
# ============================================================================

@st.dialog("设 置", width="large")
def render_settings_dialog():
    """渲染设置弹窗"""
    # 注入 CSS: 宽度 932px + 字体 20px + 减少标题间距
    st.markdown(
        """
        <style>
        div[role="dialog"][aria-modal="true"] {
            width: 932px !important;
            max-width: 932px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
        /* 减少标题和 Tab 之间的间距 */
        div[role="dialog"] .stTabs {
            margin-top: -15px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    config_manager = ConfigManager(get_db_connection)
    config = config_manager.load()
    
    # 初始化变更字典
    whisper_changes = {}
    model_changes = {}
    trans_changes = {}
    export_changes = {}
    
    # 调试模式 (已移除)
    # st.session_state['debug_mode'] = st.session_state.get('debug_mode', False)
    # remove spacer 

    # 创建 Tabs: 拆分 Whisper 设置 和 语音识别参数
    tab_whisper, tab_params, tab_model, tab_trans, tab_export = st.tabs([
        "Whisper 设置", 
        "语音识别参数",
        "翻译模型配置", 
        "翻译设置", 
        "字幕格式"
    ])
    
    # 1. Whisper 设置 (硬件/模型)
    with tab_whisper:
        st.subheader("模型与硬件")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            # 模型大小
            model_sizes = ["tiny", "base", "small", "medium", "large-v3"]
            model_size = st.selectbox(
                "Whisper 模型",
                model_sizes,
                index=model_sizes.index(config.whisper.model_size)
            )
            whisper_changes['whisper_model'] = model_size
            
            # 设备
            devices = ["cpu", "cuda", "mps"]
            curr_dev = config.whisper.device
            if curr_dev not in devices: 
                curr_dev = "cpu"
                
            device = st.selectbox(
                "运行设备",
                devices,
                index=devices.index(curr_dev)
            )
            whisper_changes['device'] = device
            
        with col_w2:
            # 计算类型
            compute_types = ["int8", "float16"]
            compute_type = st.selectbox(
                "计算精度",
                compute_types,
                index=compute_types.index(config.whisper.compute_type)
            )
            whisper_changes['compute_type'] = compute_type
            
    # 2. 语音识别参数 (逻辑参数)
    with tab_params:
        st.subheader("识别参数配置")
        
        # 内容类型
        content_type_options = {ct: get_content_type_display_name(ct) for ct in ContentType}
        content_type_keys = list(content_type_options.keys())
        current_ct_idx = content_type_keys.index(config.content_type) if config.content_type in content_type_keys else 0
        
        content_type = st.selectbox(
            "内容场景 (自动优化 VAD)",
            content_type_keys,
            format_func=lambda x: content_type_options[x],
            index=current_ct_idx
        )
        whisper_changes['content_type'] = content_type
        if content_type:
            st.caption(f"{get_content_type_description(content_type)}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 源语言（仅列出合法的 Whisper ISO 639-1 语言代码）
        lang_keys = list(WHISPER_SOURCE_LANG_MAP.keys())
        curr_source_lang = config.whisper.source_language
        # 兼容旧配置中可能保存的非法代码（如 'chs'、'eng'），回退到 'auto'
        if curr_source_lang not in lang_keys:
            curr_source_lang = 'auto'
        source_lang = st.selectbox(
            "视频原声语言",
            lang_keys,
            format_func=lambda x: WHISPER_SOURCE_LANG_MAP[x],
            index=lang_keys.index(curr_source_lang)
        )
        whisper_changes['source_language'] = source_lang

    # 2. 翻译模型配置
    with tab_model:
        st.subheader("LLM 服务商配置")
        
        # 服务商选择
        provider_keys = list(LLM_PROVIDERS.keys())
        try:
            default_prov_idx = provider_keys.index(config.current_provider)
        except ValueError:
            default_prov_idx = 0
            
        def on_provider_change():
            st.session_state._settings_provider_changed = True
            
        provider = st.selectbox(
            "选择 AI 服务商",
            provider_keys,
            index=default_prov_idx,
            key="settings_provider_select",
            on_change=on_provider_change
        )
        model_changes['provider'] = provider
        
        # 获取配置
        provider_cfg = config.provider_configs.get(provider)
        if not provider_cfg:
            default = LLM_PROVIDERS.get(provider, {})
            from core.models import ProviderConfig
            provider_cfg = ProviderConfig(
                api_key='',
                base_url=default.get('base_url', ''),
                model_name=default.get('model', '')
            )
            
        # 清除标记
        if '_settings_provider_changed' in st.session_state:
            del st.session_state['_settings_provider_changed']
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Base URL
        base_url = st.text_input(
            "Base URL",
            value=provider_cfg.base_url,
            help="API 请求地址",
            key=f"set_base_{provider}"
        )
        model_changes['base_url'] = base_url
        
        # Model Name & API Key
        if "Ollama" in provider:
            col_m1, col_m2 = st.columns([3, 1], vertical_alignment="bottom")
            with col_m1:
                ollama_models = fetch_ollama_models(base_url)
                if ollama_models:
                    try:
                        m_idx = ollama_models.index(provider_cfg.model_name)
                    except ValueError:
                        m_idx = 0
                    model_name = st.selectbox("选择模型", ollama_models, index=m_idx, key=f"set_model_{provider}")
                else:
                    st.warning("未检测到模型，请确保 Ollama 正在运行")
                    model_name = st.text_input("模型名称 (手动)", value=provider_cfg.model_name, key=f"set_model_man_{provider}")
            with col_m2:
                if st.button("刷新", key=f"set_ref_{provider}", use_container_width=True):
                    # 触发页面重新渲染，fetch_ollama_models 会重新请求
                    st.toast("模型列表已刷新")
                    st.rerun()
            api_key = ""
        else:
            model_name = st.text_input("模型名称", value=provider_cfg.model_name, key=f"set_model_{provider}")
            api_key = st.text_input(
                "API Key",
                value=provider_cfg.api_key,
                type="password",
                key=f"set_key_{provider}"
            )
            
        model_changes['model_name'] = model_name
        model_changes['api_key'] = api_key
        
        # 仅第三方模型显示测试连接按钮
        if "Ollama" not in provider:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("测试连接", use_container_width=True):
                if not api_key:
                    st.warning("请先填写 API Key")
                elif not model_name:
                    st.warning("请先填写模型名称")
                else:
                    with st.spinner("连接测试中..."):
                        ok, msg = test_api_connection(api_key, base_url, model_name)
                        if ok:
                            st.toast("连接成功！")
                        else:
                            st.error(f"连接失败: {msg}")

    # 3. 翻译设置
    with tab_trans:
        st.subheader("翻译流程控制")
        
        enable_trans = st.toggle("启用翻译功能", value=config.translation.enabled)
        trans_changes['enable_translation'] = enable_trans
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        target_lang = st.selectbox(
            "目标语言",
            TARGET_LANG_OPTIONS,
            format_func=lambda x: ISO_LANG_MAP.get(x, x),
            index=TARGET_LANG_OPTIONS.index(config.translation.target_language)
        )
        trans_changes['target_language'] = target_lang
        
        batch_size = st.number_input(
            "批处理行数 (长视频分批翻译)",
            min_value=50, max_value=5000, step=50,
            value=config.translation.max_lines_per_batch
        )
        trans_changes['max_lines_per_batch'] = batch_size

    # 4. 字幕格式
    with tab_export:
        st.subheader("导出格式选择")
        st.caption("选择生成的字幕文件格式（可多选）")
        st.markdown("<br>", unsafe_allow_html=True)
        
        export_formats = config.export.formats
        new_formats = []
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            # SRT
            if st.checkbox("SRT", value='srt' in export_formats): 
                new_formats.append('srt')
            st.caption("最通用，几乎所有播放器支持")
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            
            # VTT
            if st.checkbox("VTT", value='vtt' in export_formats): 
                new_formats.append('vtt')
            st.caption("Web/HTML5 播放器专用")
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            
            # ASS
            if st.checkbox("ASS", value='ass' in export_formats): 
                new_formats.append('ass')
            st.caption("支持丰富样式，动漫字幕常用")
            
        with col_e2:
            # SSA
            if st.checkbox("SSA", value='ssa' in export_formats): 
                new_formats.append('ssa')
            st.caption("ASS 的前身，兼容性更好")
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            
            # SUB
            if st.checkbox("SUB", value='sub' in export_formats): 
                new_formats.append('sub')
            st.caption("老式 DVD 播放器支持")
            
        if not new_formats:
            new_formats = ['srt'] # default fallback
            
        export_changes['export_formats'] = new_formats

    st.markdown("---")
    
    # 底部保存按钮
    if st.button("保存所有设置", type="primary", use_container_width=True):
        _save_full_config(config_manager, whisper_changes, model_changes, trans_changes, export_changes)
        st.rerun()


def _save_full_config(mgr, w_changes, m_changes, t_changes, e_changes):
    """保存逻辑"""
    config = mgr.load()
    
    # Whisper
    config.whisper.model_size = w_changes['whisper_model']
    config.whisper.compute_type = w_changes['compute_type']
    config.whisper.device = w_changes['device']
    config.whisper.source_language = w_changes['source_language']
    config.content_type = w_changes['content_type']
    
    # Models
    config.update_provider_config(
        m_changes['provider'],
        m_changes['api_key'],
        m_changes['base_url'],
        m_changes['model_name']
    )
    
    # Translation
    config.translation.enabled = t_changes['enable_translation']
    config.translation.target_language = t_changes['target_language']
    config.translation.max_lines_per_batch = t_changes['max_lines_per_batch']
    
    # Export
    config.export.formats = e_changes['export_formats']
    
    # Save
    if mgr.save(config):
        st.toast("配置已保存")
    else:
        st.toast("配置未变更")
