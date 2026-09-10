#!/bin/bash
# AtuoIPDR 服务器部署脚本
# 用法: bash deploy.sh（在服务器上以 root 运行）
set -e

REPO="https://github.com/wenhan9739/AtuoIPDR.git"
APP_DIR="/opt/atuoipdr"
# 密钥不要写进仓库（public！）：export MINERU_API_KEY=... 后再运行
MINERU_KEY="${MINERU_API_KEY:?请先 export MINERU_API_KEY=你的密钥}"

echo "=== AtuoIPDR 部署 ==="

# 1. 停旧项目
echo "[1/6] 停止旧项目..."
docker ps -q | xargs -r docker stop 2>/dev/null || true
docker ps -aq | xargs -r docker rm 2>/dev/null || true
systemctl stop nginx 2>/dev/null || true

# 2. 克隆/更新代码
echo "[2/6] 拉取代码..."
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull origin master
else
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

# 3. Docker 构建 + 启动
echo "[3/6] Docker 构建..."
export MINERU_API_KEY="$MINERU_KEY"
docker-compose -f deploy/docker-compose.yml up --build -d

# 4. 健康检查
echo "[4/6] 健康检查..."
sleep 5
for i in $(seq 1 10); do
    if curl -sf http://localhost:5000/health > /dev/null; then
        echo "  ✓ Flask OK"
        break
    fi
    echo "  等待中... ($i/10)"
    sleep 5
done

# 5. 防火墙
echo "[5/6] 防火墙..."
# 宝塔面板通常自行管理端口，此处仅确保 80 端口开放
# iptables -I INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || true

# 6. 完成
echo "[6/6] 部署完成！"
echo ""
echo "  访问地址: http://www.magpieagent.online"
echo "  健康检查: http://www.magpieagent.online/health"
echo "  日志: docker logs atuoipdr"
