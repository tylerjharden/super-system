#!/usr/bin/env bash
# Per-boot runtime initialization. Brings up the Docker daemon (required every
# boot; there is no systemd in this VM) and, best-effort, one headless Factorio
# instance for interactive lab-play. Must tolerate restarts and return; it never
# fails the environment on a best-effort cluster start.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_MODEL="${ADAPT1_FLE_OLLAMA_MODEL:-qwen2.5-coder:7b}"

echo "==> Starting local Ollama model server"
if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    nohup ollama serve >/tmp/adapt1-fle-ollama.log 2>&1 &
    echo "$!" >/tmp/adapt1-fle-ollama.pid
  fi
  ollama_ready=0
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      ollama_ready=1
      break
    fi
    sleep 1
  done
  if [ "$ollama_ready" = "1" ]; then
    ollama list | grep -Fq "$OLLAMA_MODEL" \
      && echo "Ollama ready with $OLLAMA_MODEL" \
      || echo "WARN: Ollama model $OLLAMA_MODEL is not installed" >&2
  else
    echo "WARN: Ollama API not ready" >&2
  fi
else
  echo "WARN: Ollama is not installed" >&2
fi

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
