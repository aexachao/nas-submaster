# 🎬 NAS Subtitle Manager (NAS 字幕管家)

> **基于 Whisper + LLM 的全自动视频字幕提取与翻译工具**

[![Docker Image](https://img.shields.io/badge/Docker%20Image-aexachao%2Fnas--subtitle--manager-blue?logo=docker)](https://hub.docker.com/r/aexachao/nas-subtitle-manager)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NAS 字幕管家** 是一个专为家庭 NAS 用户设计的轻量级 Web 工具。它采用 **前后端分离** 的设计理念，能够自动扫描 NAS 媒体库，利用 **Faster-Whisper** 提取视频语音生成字幕，并调用 **大语言模型 (LLM)** 将其翻译成中文。

本项目已进行模块化重构，运行更稳定，维护更方便。

---

## ✨ 核心特性

*   **🎯 全流程自动化**：一键扫描媒体库 → 智能识别现有字幕 → 提取音频 → 语音转文字 → AI 翻译 → 原位保存。
*   **🧠 智能翻译引擎**：
    *   内置 `translator.py` 独立翻译模块。
    *   支持 **防复读**、**格式校验**、**智能断句**。
    *   内置“信达雅”级提示词（Prompt），告别机翻感。
*   **🤖 多模型支持**：
    *   **语音识别**：内置 Faster-Whisper，支持 `tiny` 到 `large-v3` 全系列模型。
    *   **AI 翻译**：完美支持 **Ollama (本地隐私)**、**DeepSeek (高性价比)**、**Google Gemini**、**OpenAI** 等主流接口。
*   **🎨 现代化 UI**：基于 Streamlit 构建的 **HeroUI** 风格界面，支持暗色模式，交互流畅。
*   **📊 任务队列系统**：支持批量添加任务、后台异步处理、实时进度监控、断点重试、暂停/恢复。
*   **🛡️ 隐私优先**：支持完全离线运行（Whisper 本地模型 + Ollama 本地 LLM），无需上传视频数据。

---

## 📸 界面预览

![Dashboard](docs/images/dashboard.png)

---

## 🚀 部署方式 (二选一)

### 方案一：使用 Docker Hub 镜像 (推荐)

最简单的方式，直接拉取构建好的镜像，无需下载源码。

1.  在 NAS 上创建一个文件夹（例如 `nas-subtitle`）。
2.  在该目录下创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  nas-subtitle:
    # 直接拉取 Docker Hub 上的最新镜像
    image: aexachao/nas-subtitle-manager:latest
    container_name: nas-subtitle
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data                       # 数据库和模型持久化目录 (自动生成)
      - /volume1/video:/media              # ⚠️ 修改这里：将你的 NAS 视频路径映射到容器内的 /media
    environment:
      - TZ=Asia/Shanghai
    extra_hosts:
      - "host.docker.internal:host-gateway" # 允许容器访问宿主机上的 Ollama

  # (可选) 本地大模型服务，如果不需要本地模型可删除此段
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_data:/root/.ollama
```

3.  启动服务：

```bash
docker compose up -d
```

---

### 方案二：本地源码构建

如果你需要修改代码或进行二次开发，可以使用此方法。

1.  **克隆项目**：

```bash
git clone https://github.com/aexachao/nas-subtitle-manager.git
cd nas-subtitle-manager
```

2.  **配置 `docker-compose.yml`**：
    (源码中已包含默认配置，请确保 `volumes` 映射路径正确)

3.  **构建并启动**：

```bash
# --build 参数强制重新构建镜像
docker compose up -d --build
```

---

## 📂 项目结构

```text
nas-subtitle-manager/
├── app.py                 # [入口] 主应用程序 (UI渲染、数据库管理、任务调度)
├── translator.py          # [核心] 独立翻译模块 (封装 LLM 调用、Prompt 优化、质量检测)
├── requirements.txt       # Python 依赖清单
├── Dockerfile             # 容器构建文件
├── docker-compose.yml     # 容器编排配置
└── data/                  # [自动生成] 存放 SQLite 数据库和 Whisper 模型文件
```

---

## 📖 使用指南

启动成功后，浏览器访问：`http://NAS_IP:8501`

### 1. 配置 Whisper (听写)
在左侧侧边栏 **"Whisper 设置"** 中配置：
*   **模型大小**：推荐 `small` (速度快) 或 `medium` (精度高，需较大内存)。
*   **计算类型**：NAS CPU 推荐使用 `int8`。
*   **视频原声**：建议手动指定语言（如“日语”），识别率远高于自动检测。

### 2. 配置翻译服务 (LLM)
在左侧侧边栏 **"翻译设置"** 中选择提供商。

#### 🏠 本地模型 (Ollama) - **隐私推荐**
1.  确保 Ollama 容器已运行。
2.  下载推荐模型（通义千问 Qwen2.5）：
    ```bash
    docker exec -it ollama ollama pull qwen2.5:7b
    ```
3.  在网页端选择 **"Ollama (本地模型)"**，点击刷新列表选择模型。

#### ☁️ 云端 API - **质量推荐**
*   **DeepSeek**：国内首选，甚至比 GPT-4 更懂中文俚语。
*   **Google Gemini**：`gemini-1.5-flash` 免费且速度极快。
*   **配置**：选择厂商 -> 填入 API Key -> 点击“测试连接”。

### 3. 批量任务管理
1.  **刷新媒体库**：扫描 `/media` 下的所有视频文件。
2.  **筛选与选择**：利用顶部的 Radio 筛选“无字幕”视频，支持全选。
3.  **开始处理**：点击“开始处理”，任务将进入后台队列。
4.  **监控**：在“任务队列”标签页查看实时进度。
    *   **重试**：失败的任务可一键重试。

---

## 🤝 常见问题 (FAQ)

**Q: 为什么一直卡在“加载模型...”？**
A: 首次运行需要下载 Whisper 模型。如果网络不通，请检查 NAS 网络设置，或手动下载模型文件放置于 `./data/models` 目录下。

**Q: 翻译进度条不动了？**
A: 可能是 API 超时或并发限制。程序内置了重试机制，点击任务卡片上的“重试”按钮即可。

**Q: 怎么更新到最新版本？**
A:
*   **Docker Hub 用户**：`docker compose pull && docker compose up -d`
*   **源码用户**：`git pull && docker compose up -d --build`

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源。

---

**⭐ 如果这个项目对你有帮助，欢迎在 GitHub 上点个 Star！**
