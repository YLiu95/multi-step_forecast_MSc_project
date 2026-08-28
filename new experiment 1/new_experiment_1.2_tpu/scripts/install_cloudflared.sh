#!/usr/bin/env bash
set -euo pipefail

URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
curl -fL "$URL" -o /usr/local/bin/cloudflared
chmod 0755 /usr/local/bin/cloudflared
cloudflared --version