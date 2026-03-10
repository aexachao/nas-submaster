#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI 样式定义
集中管理所有 CSS 样式
"""

HERO_CSS = """
<style>
    /* Layout Styles (Theme-agnostic) */
    .block-container {
        max-width: 1280px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        margin: 0 auto;
    }
    
    h1 { font-size: 32px !important; font-weight: 700 !important; padding-bottom: 0.5rem; }
    h2, h3 { font-size: 16px !important; font-weight: 600 !important; }

    /* Hero Card - uses CSS variables for theme compatibility */
    .hero-card {
        border-radius: 6px;
        padding: 12px 16px;
        transition: border-color 0.2s;
        margin-bottom: 16px;
    }
    
    /* Status Chips - uses CSS variables */
    .status-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 6px;
        border: 1px solid;
    }
    .chip-gray { 
        background: rgba(128,128,128,0.2); 
        color: #71717a; 
        border-color: rgba(128,128,128,0.3); 
    }
    .chip-blue { 
        background: rgba(59,130,246,0.2); 
        color: #3b82f6; 
        border-color: rgba(59,130,246,0.4); 
    }
    .chip-green { 
        background: rgba(34,197,94,0.2); 
        color: #22c55e; 
        border-color: rgba(34,197,94,0.4); 
    }
    .chip-red { 
        background: rgba(239,68,68,0.2); 
        color: #ef4444; 
        border-color: rgba(239,68,68,0.4); 
    }

    /* Button Styles - minimal overrides, let Streamlit handle colors */
    .stButton button {
        border-radius: 6px !important;
        font-size: 13px !important;
        height: 32px !important;
        padding: 0 12px !important;
    }

    button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #2563eb !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    .stProgress > div > div > div > div { background-color: #2563eb; }

    .stCheckbox label {
        min-height: 0px !important;
        margin-bottom: 0px !important;
    }

    .task-card-wrapper {
        position: relative;
        margin-bottom: 24px;
    }

    /* Tab 字体调整 */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    /* 设置弹窗最大宽度 932px */
    div[data-testid="stDialog"] > div[role="dialog"] {
        max-width: 932px !important;
        width: 932px !important;
    }
</style>
"""