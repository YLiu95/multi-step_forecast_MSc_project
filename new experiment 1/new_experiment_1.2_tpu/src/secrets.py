from __future__ import annotations

import os
from pathlib import Path


SECRET_DIR = Path("/root/.secrets")


def get_token(environment_name: str) -> str | None:
    if token := os.environ.get(environment_name):
        return token
    cache = SECRET_DIR / environment_name.lower()
    if cache.exists():
        return cache.read_text().strip()
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(environment_name)
    except Exception:
        return None
    if token:
        SECRET_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache.write_text(token)
        cache.chmod(0o600)
    return token