#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${ARTIFACT_ROOT:-/root/artifacts/new_experiment_1.2_tpu}"
PORT="${TENSORBOARD_PORT:-16006}"
RUNTIME_DIR="$LOG_ROOT/tunnel"
mkdir -p "$RUNTIME_DIR"

if ! curl -fsS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
  tensorboard --logdir "$LOG_ROOT/runs" --host 0.0.0.0 --port "$PORT" \
    --reload_multifile=true >"$RUNTIME_DIR/tensorboard.log" 2>&1 &
  echo $! >"$RUNTIME_DIR/tensorboard.pid"
fi

if ! curl -fsS --retry 30 --retry-delay 1 --retry-connrefused \
  "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
  cat "$RUNTIME_DIR/tensorboard.log" >&2
  echo "TensorBoard did not become healthy; inspect $RUNTIME_DIR/tensorboard.log" >&2
  exit 1
fi

if [[ ! -x /usr/local/bin/cloudflared ]]; then
  echo "cloudflared is not installed; run scripts/install_cloudflared.sh first" >&2
  exit 1
fi

if [[ -f "$RUNTIME_DIR/cloudflared.pid" ]] \
   && kill -0 "$(cat "$RUNTIME_DIR/cloudflared.pid")" 2>/dev/null; then
  echo "Cloudflare tunnel is already running."
else
  cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
    >"$RUNTIME_DIR/cloudflared.log" 2>&1 &
  echo $! >"$RUNTIME_DIR/cloudflared.pid"
fi

URL=$(timeout 30 tail --pid="$(cat "$RUNTIME_DIR/cloudflared.pid")" -n +1 -F \
  "$RUNTIME_DIR/cloudflared.log" 2>/dev/null \
  | grep -m1 -o 'https://[-a-z0-9]*\.trycloudflare\.com' || true)
if [[ -n "$URL" ]]; then
  echo "TensorBoard: $URL"
  exit 0
fi

echo "Tunnel started, but its URL is not ready yet. Inspect:"
echo "  $RUNTIME_DIR/cloudflared.log"