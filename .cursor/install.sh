#!/usr/bin/env bash
# Idempotent repository bootstrap for the REI Adapt-1 x Factorio Learning
# Environment (FLE) research preview. Safe to re-run: every step is guarded or
# naturally convergent. Must terminate (no long-running processes here).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTORIO_IMAGE="factoriotools/factorio:2.0.73"
OLLAMA_MODEL="${ADAPT1_FLE_OLLAMA_MODEL:-qwen2.5-coder:7b}"

echo "==> [1/10] System packages"
# Keep apt fully non-interactive: some packages (e.g. fuse3) ship conffiles that
# otherwise trigger an interactive prompt and break unattended install.
APT_OPTS=(-y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)
sudo apt-get update -y
sudo apt-get install "${APT_OPTS[@]}" \
  ca-certificates curl git jq build-essential \
  python3 python3-venv python3-dev \
  libpq-dev netcat-openbsd fuse-overlayfs zstd

echo "==> [2/10] uv (Astral)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if ! grep -qs '.local/bin' "$HOME/.bashrc"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

echo "==> [3/10] Docker Engine + Compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
# Allow the agent user to use Docker without sudo. Group membership becomes
# effective on the next login session / boot (i.e. on every fresh agent pod).
sudo usermod -aG docker "$USER" || true

echo "==> [4/10] Docker daemon config for nested VM"
# Cloud Agent VMs run with /var/lib/docker on an overlay filesystem, so the
# default overlay(fs) storage driver fails to mount images (overlay-on-overlay
# -> 'invalid argument'). fuse-overlayfs avoids this. We also disable the
# containerd image store so the classic graph driver honours storage-driver.
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "features": { "containerd-snapshotter": false },
  "storage-driver": "fuse-overlayfs"
}
JSON

echo "==> [5/10] Start Docker daemon (no systemd in this VM)"
sudo service docker restart || sudo service docker start || true
for _ in $(seq 1 30); do
  if sudo docker info >/dev/null 2>&1; then break; fi
  sleep 1
done
sudo docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not ready" >&2; exit 1; }

echo "==> [6/10] Pre-pull Factorio image ($FACTORIO_IMAGE)"
# Pull into the fuse-overlayfs store so the first 'fle cluster start' is fast.
sudo docker pull "$FACTORIO_IMAGE"

echo "==> [7/10] Install Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> [8/10] Pre-pull local code model ($OLLAMA_MODEL)"
OLLAMA_BOOTSTRAP_PID=""
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ollama serve >/tmp/adapt1-fle-ollama-install.log 2>&1 &
  OLLAMA_BOOTSTRAP_PID="$!"
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
ollama pull "$OLLAMA_MODEL"
if [ -n "$OLLAMA_BOOTSTRAP_PID" ]; then
  kill "$OLLAMA_BOOTSTRAP_PID"
  wait "$OLLAMA_BOOTSTRAP_PID" 2>/dev/null || true
fi

echo "==> [9/10] Python toolchain + FLE"
cd "$REPO_DIR"
uv python install 3.12
# Rebuild the venv if it is missing or incomplete. A fresh git checkout (each
# build/boot) can leave an empty .venv directory behind, so guard on the actual
# activate script rather than the directory.
if [ ! -f .venv/bin/activate ]; then
  # Clear any stale/incomplete .venv before recreating. sudo handles root-owned
  # files that 'fle cluster start' can leave via container bind mounts; --clear
  # --force lets uv recreate over any residual non-venv directory.
  sudo rm -rf .venv 2>/dev/null || rm -rf .venv 2>/dev/null || true
  uv venv --python 3.12 --clear --force .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -U pip
# Project metadata owns the FLE compatibility pin and development tooling.
uv pip install -e "$REPO_DIR[dev]"

echo "==> [10/10] Local config / data dirs"
mkdir -p .fle configs
if [ ! -f .env ]; then
  cat > .env <<'ENVV'
ADAPT1_BASE_URL=https://rei-neuroadapt-api.reilabs.org
ADAPT1_FLE_MODEL=ollama-qwen2.5-coder:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
FLE_DB_TYPE=sqlite
SQLITE_DB_FILE=.fle/data.db
PORT_OFFSET=0
ENVV
fi

echo "install.sh complete"
