#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/pi/home-server/pi-ai-gateway"
SYSTEMD_DIR="/etc/systemd/system"
NGINX_CONF_DIR="${HOME}/home-server/nginx/conf.d"

echo "=== Pi AI Gateway Installer ==="

# 1. Check secrets
echo "[1/5] Checking secrets..."
cd "${PROJECT_DIR}"
if [ -f .env ]; then
    echo "  .env found"
elif command -v op &> /dev/null; then
    op inject -i .env.tpl -o .env --force
    echo "  Secrets injected from 1Password"
else
    echo "  ERROR: No .env file found. Create one with:"
    echo "    OPENROUTER_API_KEY=your-key"
    echo "    GATEWAY_API_KEY="
    exit 1
fi

# 2. Create Python venv for model-bridge
echo "[2/5] Setting up model-bridge venv..."
if [ ! -d "${PROJECT_DIR}/.venv" ]; then
    python3 -m venv "${PROJECT_DIR}/.venv"
fi
"${PROJECT_DIR}/.venv/bin/pip" install -q -e .

# 3. Install systemd unit
echo "[3/5] Installing model-bridge systemd service..."
sudo cp "${PROJECT_DIR}/deploy/model-bridge.service" "${SYSTEMD_DIR}/"
sudo systemctl daemon-reload
sudo systemctl enable model-bridge.service
sudo systemctl restart model-bridge.service

# 4. Copy nginx config
echo "[4/5] Updating nginx configuration..."
if [ -d "${NGINX_CONF_DIR}" ]; then
    cp "${PROJECT_DIR}/deploy/nginx-ai.conf" "${NGINX_CONF_DIR}/ai-gateway.conf"
    docker exec nginx-proxy nginx -s reload 2>/dev/null || echo "  nginx reload skipped (container not running?)"
else
    echo "  WARNING: ${NGINX_CONF_DIR} not found. Copy deploy/nginx-ai.conf manually."
fi

# 5. Show docker-compose snippet
echo "[5/5] Docker service ready."
echo ""
echo "Add this to your ~/home-server/docker-compose.yml services section:"
echo ""
echo "  ai-gateway:"
echo "    build:"
echo "      context: /home/pi/pi-ai-gateway"
echo "      dockerfile: deploy/Dockerfile"
echo "    container_name: ai-gateway"
echo "    restart: unless-stopped"
echo "    ports:"
echo "      - \"8080:8080\""
echo "    env_file:"
echo "      - /home/pi/pi-ai-gateway/.env"
echo "    environment:"
echo "      - GATEWAY_MODEL_BRIDGE_URL=http://host.docker.internal:9099"
echo "    extra_hosts:"
echo "      - \"host.docker.internal:host-gateway\""
echo "    networks:"
echo "      - vpn_net"
echo "    mem_limit: 128m"
echo "    cpus: \"0.5\""
echo ""
echo "Then run: docker compose up -d ai-gateway"
echo ""

# Verify
echo "=== Verification ==="
sleep 2
echo -n "model-bridge: "
curl -sf http://localhost:9099/health && echo "OK" || echo "FAILED"
echo ""
echo "Done. Next: add the ai-gateway service to docker-compose.yml and run 'docker compose up -d ai-gateway'"
