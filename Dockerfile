# 使用 NVIDIA CUDA 基础镜像（libcublas.so.12 等 CUDA 库已内置在镜像中）
# 启用 GPU 时无需在宿主机挂载 NVIDIA 驱动目录
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    tzdata \
    curl \
    wget \
    iputils-ping \
    net-tools \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 让 `python` / `pip` 命令指向 python3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 复制依赖文件
COPY requirements.txt .

# 使用 pip 默认源安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /data/models

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动命令
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.runOnSave=false"]
