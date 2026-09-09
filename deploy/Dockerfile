# AtuoIPDR — Python + R multi-stage build
# 目标镜像 < 1.5GB，适配 4核/4GB 服务器
FROM python:3.10-slim AS base

# 系统依赖（OpenCV + R 编译环境）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libgomp1 \
    r-base-core \
    && rm -rf /var/lib/apt/lists/*

# R 包（预编译 CRAN 二进制，避免 OOM）
RUN Rscript -e "install.packages(c('IPDfromKM','survival','jsonlite'), repos='https://cran.r-project.org', Ncpus=1)"

# Python 依赖
COPY requirements_server.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 代码
WORKDIR /app
COPY . /app/

# 创建工作目录
RUN mkdir -p /data/uploads /data/results /data/tmp

# Flask 端口
EXPOSE 5000

# 内存限制（4GB 服务器安全余量）
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=webapp/app.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "600", "--preload", "webapp.app:app"]
