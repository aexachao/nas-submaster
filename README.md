<div align="center">

<img src="assets/logo.png" width="20%" />

<!-- 关键：添加 border-bottom: none; 去掉标题下的横线 -->
<h1 style="margin-top: 10px; margin-bottom: 0; border-bottom: none;">
NAS SubMaster (NAS 字幕管家)
</h1>

<!-- 关键：添加 margin-bottom: 5px; 缩小与下方徽章的距离 -->
<p style="font-size: 16px; font-weight: bold; margin-top: 5px; margin-bottom: 5px;">
基于 Whisper + LLM 的全自动视频字幕提取与翻译工具
</p>

<!-- 移除了 <br> 标签，让徽章紧贴着说明文字 -->
[![Docker Pulls](https://img.shields.io/docker/pulls/aexachao/nas-subtitle-manager.svg?logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/aexachao/nas-subtitle-manager)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

<hr>

</div>


**NAS 字幕管家** 是一个专为家庭 NAS 用户设计的智能化字幕工具。本项目已完成 **深度代码重构**，采用 **UI 与业务逻辑分离** 的模块化架构，运行更稳定，扩展性更强。

它能够自动扫描 NAS 媒体库，支持 **3级子目录精确筛选**，利用 **Faster-Whisper** 提取语音，并调用 **大语言模型 (LLM)** 生成高质量的中文字幕。

---

## ✨ 核心特性

* **🏗️ 模块化架构 (New)**：代码重构为 Service/UI 分层模式。`services` 层处理核心计算，`ui` 层负责交互渲染，逻辑更清晰，维护更方便。
* **📂 精确目录扫描 (New)**：告别全盘漫长扫描！新增 **3级目录选择器**，支持精确指定扫描媒体库下的某个子目录（如 `/media/电影/2024/科幻`），只处理你关心的文件夹。
* **🎯 全流程自动化**：一键扫描 → 智能识别缺失字幕 → 提取音频 → 语音转文字 → AI 翻译 → 原位保存。
* **🧠 智能翻译引擎**：
    * 内置 `translator` 服务，支持 **防复读**、**格式校验**、**智能断句**。
    * 内置“信达雅”级 Prompt，拒绝生硬机翻。
* **🤖 多模型支持**：
    * **语音识别**：内置 Faster-Whisper，支持 `tiny` 到 `large-v3` 全系列模型。
    * **AI 翻译**：完美支持 **Ollama (本地隐私)**、**DeepSeek (高性价比)**、**Google Gemini**、**OpenAI** 等主流接口。
* **📊 任务队列系统**：支持批量添加任务、后台异步处理、实时进度监控、断点重试。

---

## 📸 界面预览

![Dashboard](docs/images/dashboard.png)

---

## 🚀 部署方式 (二选一)

### 方案一：使用 Docker Hub 镜像 (推荐)

1.  在 NAS 上创建一个文件夹（例如 `nas-subtitle`）。
2.  在该目录下创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  nas-subtitle:
    image: aexachao/nas-subtitle-manager:latest
    container_name: nas-subtitle
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data                       # 数据库、配置和模型持久化目录
      - /volume1/video:/media              # ⚠️ 视频根目录，映射后可在 Web 端选择具体子目录
    environment:
      - TZ=Asia/Shanghai
    extra_hosts:
      - "host.docker.internal:host-gateway" # 允许容器访问宿主机上的 Ollama

  # (可选) 本地大模型服务
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

如果您需要二次开发，请参考最新的模块化目录结构。

1.  **克隆项目**：
    ```bash
    git clone [https://github.com/aexachao/nas-subtitle-manager.git](https://github.com/aexachao/nas-subtitle-manager.git)
    cd nas-subtitle-manager
    ```

2.  **构建并启动**：
    ```bash
    docker compose up -d --build
    ```

---

## 📂 项目结构

本项目采用了清晰的分层架构：

```text
nas-submaster/
├── app.py
├── assets
│   └── logo.png
├── core
│   ├── config.py
│   ├── models.py
│   └── worker.py
├── database
│   ├── connection.py
│   ├── media_dao.py
│   └── task_dao.py
├── Dockerfile
├── requirements.txt
├── services
│   ├── media_scanner.py
│   ├── subtitle_converter.py
│   ├── translator.py
│   └── whisper_service.py
├── ui
│   ├── components.py
│   ├── pages
│   │   ├── media_library.py
│   │   └── task_queue.py
│   ├── settings_modal.py
│   └── styles.py
└── utils
    ├── format_utils.py
    └── lang_detection.py
```

---

## 📖 使用指南

启动成功后，浏览器访问：`http://localhost:8501`

### 1. 媒体库扫描 (Sub-folder Scanning)
在首页 **"媒体库"** 区域：
* **根路径**：默认为 Docker 映射的 `/media`。
* **目录选择**：点击下拉菜单，系统会动态加载 `/media` 下的文件夹。
    * ✅ 支持 **3级深度** 的子文件夹浏览。
    * 例如：你可以直接选择 `Movie > 2024 > 动作片` 进行扫描，而无需扫描整个媒体库。
* **执行扫描**：选中目标文件夹后，点击“扫描当前目录”。

### 2. 批量任务管理
1.  **筛选**：扫描完成后，列表展示该目录下的视频文件，勾选需要处理的文件。
2.  **提交**：点击“添加到队列”，任务将被发送到 `core/worker.py` 进行后台处理。
3.  **监控**：切换到 **"任务队列"** 页面查看实时日志。

### 3. 配置建议
* **Whisper**：NAS 若无显卡，建议使用 `tiny` 或 `small` 模型 + `int8` 量化。
* **翻译 API**：
    * **DeepSeek**：推荐用于高性价比翻译。
    * **Ollama**：推荐 `qwen2.5` 模型用于本地离线翻译。

---

## 🤝 常见问题 (FAQ)

**Q: 目录选择器为什么只能看到 3 级？**
A: 为了保证 Web 界面的响应速度，我们在 `media_scanner.py` 中限制了遍历深度。如果您的文件层级极深，建议调整 NAS 的目录挂载方式，将更深层的目录直接映射到容器的 `/media` 下。

**Q: 之前的数据库还能用吗？**
A: 本次重构优化了数据库结构，旧版 `data/database.db` 可能无法直接兼容。建议备份旧数据后，让程序自动生成新的数据库文件。

**Q: 如何查看报错日志？**
A: 可以通过 `docker logs -f nas-subtitle` 查看后端详细运行日志。

---

## 📄 开源协议

本项目采用 [AGPL-3,0](LICENSE) 协议开源。

**简单来说**：你可以自由使用、修改和商用本项目，但如果你把它做成网站或服务给别人用，需要公开你的源代码。

---

**⭐ 如果这个项目对你有帮助，欢迎在 GitHub 上点个 Star！**
