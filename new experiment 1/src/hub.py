"""Backups: Hugging Face for checkpoints, GitHub for code and history.

Design rule: **a backup must never be able to kill the training run.** Every
function here swallows its exceptions and returns a status string. A flaky
network on a Kaggle box should cost you one upload, not twelve hours of GPU
time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

SECRET_DIR = Path("/root/.secrets")


# --------------------------------------------------------------------------- #
#  Tokens
# --------------------------------------------------------------------------- #
def get_token(which: str) -> str | None:
    """Read a token from the env, then the local cache, then Kaggle secrets."""
    env_key = {"hf": "HF_TOKEN", "gh": "GITHUB_TOKEN"}[which]
    if os.environ.get(env_key):
        return os.environ[env_key]
    cached = SECRET_DIR / which
    if cached.exists():
        return cached.read_text().strip()
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(env_key)
        SECRET_DIR.mkdir(exist_ok=True, mode=0o700)
        cached.write_text(tok)
        os.chmod(cached, 0o600)
        return tok
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Hugging Face
# --------------------------------------------------------------------------- #
class HFBackup:
    def __init__(self, repo_id: str, enabled: bool = True):
        self.repo_id = repo_id
        self.enabled = enabled
        self.api = None
        self.token = get_token("hf")
        if not (enabled and self.token):
            self.enabled = False
            return
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True,
                                 private=False)
        except Exception as exc:
            print(f"[hf] disabled: {type(exc).__name__}: {exc}")
            self.enabled = False

    def upload(self, local: str | Path, path_in_repo: str,
               message: str = "update") -> str:
        if not self.enabled:
            return "hf disabled"
        local = Path(local)
        t0 = time.time()
        try:
            self.api.upload_file(path_or_fileobj=str(local),
                                 path_in_repo=path_in_repo,
                                 repo_id=self.repo_id, repo_type="model",
                                 commit_message=message)
            mb = local.stat().st_size / 1e6
            return (f"hf ok  {path_in_repo}  {mb:,.0f} MB in "
                    f"{time.time() - t0:.0f}s")
        except Exception as exc:
            return f"hf FAILED {path_in_repo}: {type(exc).__name__}: {exc}"

    def upload_dir(self, local: str | Path, path_in_repo: str,
                   message: str = "update") -> str:
        if not self.enabled:
            return "hf disabled"
        try:
            self.api.upload_folder(folder_path=str(local),
                                   path_in_repo=path_in_repo,
                                   repo_id=self.repo_id, repo_type="model",
                                   commit_message=message)
            return f"hf ok  dir {path_in_repo}"
        except Exception as exc:
            return f"hf FAILED dir {path_in_repo}: {type(exc).__name__}: {exc}"

    def list_files(self, prefix: str = "") -> list[str]:
        if not self.enabled:
            return []
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="model")
            return sorted(f for f in files if f.startswith(prefix))
        except Exception:
            return []

    def download(self, path_in_repo: str, dest_dir: str | Path) -> Path | None:
        if not self.enabled:
            return None
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo_id, path_in_repo, repo_type="model",
                                token=self.token)
            dest = Path(dest_dir) / Path(path_in_repo).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            return dest
        except Exception as exc:
            print(f"[hf] download failed {path_in_repo}: {exc}")
            return None


# --------------------------------------------------------------------------- #
#  GitHub
# --------------------------------------------------------------------------- #
class GitBackup:
    """Commit and push the working tree. Never raises."""

    def __init__(self, repo_dir: str | Path, remote: str, enabled: bool = True):
        self.dir = Path(repo_dir)
        self.remote = remote
        self.token = get_token("gh")
        self.enabled = enabled and self.token is not None and (self.dir / ".git").exists()

    def _run(self, *args: str, timeout: int = 300) -> tuple[int, str]:
        try:
            p = subprocess.run(["git", *args], cwd=self.dir, capture_output=True,
                               text=True, timeout=timeout)
            out = (p.stdout + p.stderr)
            if self.token:                       # never let a token reach a log
                out = out.replace(self.token, "***")
            return p.returncode, out.strip()
        except Exception as exc:
            return 1, f"{type(exc).__name__}: {exc}"

    def push(self, message: str, paths: list[str] | None = None) -> str:
        if not self.enabled:
            return "git disabled"
        url = f"https://x-access-token:{self.token}@github.com/{self.remote}.git"
        self._run("remote", "set-url", "origin", url)
        self._run("add", "-A", *(paths or ["."]))
        code, out = self._run("commit", "-m", message)
        if code != 0 and "nothing to commit" in out:
            return "git: nothing to commit"
        code, out = self._run("push", "origin", "HEAD:main")
        self._run("remote", "set-url", "origin",
                  f"https://github.com/{self.remote}.git")
        return "git ok" if code == 0 else f"git FAILED: {out[-400:]}"
