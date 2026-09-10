# AutoIPD — Python + R
# 目标镜像 < 1.5GB，适配 4核/4GB 服务器
# 固定 bookworm：与下方 Posit 预编译二进制仓库路径匹配
FROM python:3.10-slim-bookworm

# 系统依赖（OpenCV 运行库 + R 运行时）；apt 走腾讯云内网镜像
RUN sed -i 's@deb.debian.org@mirrors.cloud.tencent.com@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 curl \
    r-base-core \
    && rm -rf /var/lib/apt/lists/*

# R 包：优先 Posit 预编译 Linux 二进制（不编译、防 OOM），失败回退腾讯 CRAN 源码
RUN Rscript -e "options(HTTPUserAgent=sprintf('R/%s R (%s)', getRversion(), paste(getRversion(), R.version\$platform, R.version\$arch, R.version\$os))); pkgs <- c('IPDfromKM','survival','jsonlite'); try(install.packages(pkgs, repos='https://packagemanager.posit.co/cran/__linux__/bookworm/latest', Ncpus=1), silent=TRUE); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(miss)) install.packages(miss, repos='https://mirrors.cloud.tencent.com/CRAN/', Ncpus=1)"

# Python 依赖：腾讯云 PyPI 镜像（注意 context 为仓库根，路径是 deploy/ 下）
COPY deploy/requirements_server.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r /tmp/requirements.txt

# 代码
WORKDIR /app
COPY . /app/

# 创建工作目录
RUN mkdir -p /data/uploads /data/results /data/tmp

EXPOSE 5000

ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=webapp/app.py

# 单 worker：任务状态存内存（JOBS），多 worker 会导致状态查询落在不同进程
CMD ["gunicorn", "--pythonpath", "/app/deploy", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "900", "webapp.app:app"]
