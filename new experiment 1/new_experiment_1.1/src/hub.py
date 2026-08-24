from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


SECRET_DIR = Path("/root/.secrets")


def get_token(kind: str) -> str | None:
    environment_name = {"hf": "HF_TOKEN", "gh": "GITHUB_TOKEN"}[kind]
    if os.environ.get(environment_name):
        return os.environ[environment_name]
    cached = SECRET_DIR / kind
    if cached.exists():
        return cached.read_text().strip()
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(environment_name)
        SECRET_DIR.mkdir(mode=0o700, exist_ok=True)
        cached.write_text(token)
        cached.chmod(0o600)
        return token
    except Exception:
        return None


class HFBackup:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.token = get_token("hf")
        self.api = None
        if not self.token:
            return
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.token)
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as exc:
            print(f"[hf] disabled: {type(exc).__name__}: {exc}")
            self.api = None

    @property
    def enabled(self) -> bool:
        return self.api is not None

    def upload(self, local: str | Path, remote: str, message: str) -> str:
        if not self.enabled:
            return "Hugging Face backup disabled (token unavailable)"
        started = time.time()
        try:
            self.api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                 repo_id=self.repo_id, repo_type="model",
                                 commit_message=message)
            return f"Hugging Face uploaded {remote} in {time.time() - started:.0f}s"
        except Exception as exc:
            return f"Hugging Face upload failed for {remote}: {type(exc).__name__}: {exc}"

    def rotate(self, prefix: str, keep: int) -> list[str]:
        if not self.enabled:
            return []
        messages = []
        try:
            files = sorted(path for path in self.api.list_repo_files(
                self.repo_id, repo_type="model")
                           if path.startswith(prefix) and path.endswith(".pt")
                           and "latest.pt" not in path)
            for old in files[:-keep]:
                self.api.delete_file(old, repo_id=self.repo_id, repo_type="model",
                                     commit_message=f"rotate old checkpoint {old}")
                messages.append(f"Hugging Face removed {old}")
        except Exception as exc:
            messages.append(f"Hugging Face rotation failed: {type(exc).__name__}: {exc}")
        return messages


class GitBackup:
    def __init__(self, repo_dir: str | Path, remote: str):
        self.repo_dir = Path(repo_dir)
        self.remote = remote
        self.token = get_token("gh")

    @property
    def enabled(self) -> bool:
        return bool(self.token and (self.repo_dir / ".git").exists())

    def _run(self, *arguments: str) -> tuple[int, str]:
        try:
            result = subprocess.run(["git", *arguments], cwd=self.repo_dir,
                                    text=True, capture_output=True, timeout=300)
            output = result.stdout + result.stderr
            if self.token:
                output = output.replace(self.token, "***")
            return result.returncode, output.strip()
        except Exception as exc:
            return 1, f"{type(exc).__name__}: {exc}"

    def push(self, message: str, path: str) -> str:
        if not self.enabled:
            return "GitHub backup disabled (token unavailable)"
        authenticated = f"https://x-access-token:{self.token}@github.com/{self.remote}.git"
        public = f"https://github.com/{self.remote}.git"
        self._run("remote", "set-url", "origin", authenticated)
        try:
            name_code, _ = self._run("config", "--get", "user.name")
            email_code, _ = self._run("config", "--get", "user.email")
            if name_code:
                _, author_name = self._run("log", "-1", "--format=%an")
                self._run("config", "user.name", author_name or "YLiu95")
            if email_code:
                _, author_email = self._run("log", "-1", "--format=%ae")
                self._run("config", "user.email",
                          author_email or "yliu95@users.noreply.github.com")
            self._run("add", "--", path)
            code, output = self._run("commit", "-m", message)
            if code and "nothing to commit" in output:
                return "GitHub: nothing new to commit"
            if code:
                return f"GitHub commit failed: {output[-500:]}"
            code, output = self._run("push", "origin", "HEAD:main")
            return "GitHub backup uploaded" if code == 0 else f"GitHub push failed: {output[-500:]}"
        finally:
            self._run("remote", "set-url", "origin", public)