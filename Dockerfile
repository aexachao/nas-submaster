# 使用 Python 3.10 基础镜像
FROM python:3.10-slim

# 设置环境变量
# 1. 不生成 .pyc 文件
# 2. 也是不缓冲输出
# 3. 设置时区
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# 注意：GitHub Actions 在海外，直接用官方源速度最快，不需要换源
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tzdata \
    curl \
    # 保留 ping 和 netstat 以便你在 NAS 上排查网络问题
    iputils-ping \
    net-tools \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 【重要修改】去掉了清华源 (-i https://pypi.tuna...), 使用官方源
# GitHub Actions 服务器连接官方 PyPI 速度非常快
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# 【重要修改】复制当前目录下的所有文件到容器
# 这样以后你增加 .py 文件或 assets 文件夹，不需要改 Dockerfile
# ============================================================================
COPY . .

# 创建数据目录
RUN mkdir -p /data/models

# 暴露端口
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