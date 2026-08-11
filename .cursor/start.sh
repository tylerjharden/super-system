#!/usr/bin/env bash
# Per-boot runtime initialization. Brings up the Docker daemon (required every
# boot; there is no systemd in this VM) and, best-effort, one headless Factorio
# instance for interactive lab-play. Must tolerate restarts and return; it never
# fails the environment on a best-effort cluster start.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Starting Docker daemon"
sudo service docker start >/dev/null 2>&1 || sudo systemctl start docker >/dev/null 2>&1 || true

ready=0
for _ in $(seq 1 30); do
  if docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[ "$ready" = "1" ] && echo "Docker daemon ready" || echo "WARN: Docker daemon not ready" >&2

echo "==> Best-effort: start one Factorio instance"
cd "$REPO_DIR"
if [ "$ready" = "1" ] && [ -x .venv/bin/fle ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
  [ -f .env ] && { set -a; . ./.env; set +a; }
  fle cluster start -n 1 || echo "WARN: 'fle cluster start' failed (non-fatal)" >&2
else
  echo "Skipping cluster start (docker not ready or fle not installed)"
fi

echo "start.sh complete"
