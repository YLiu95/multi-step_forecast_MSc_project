from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .config import EXPERIMENT_ROOT
from .secrets import get_token


class HFBackup:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.token = get_token("HF_TOKEN")
        self.api = None
        if not self.token:
            return
        try:
            from huggingface_hub import HfApi

            self.api = HfApi(token=self.token)
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
        except Exception as error:
            print(f"[hf] disabled: {type(error).__name__}: {error}", flush=True)
            self.api = None

    @property
    def enabled(self) -> bool:
        return self.api is not None

    def upload_folder(self, local: str | Path, remote: str, message: str) -> str:
        if not self.enabled:
            return "Hugging Face backup disabled (token unavailable)"
        started = time.time()
        try:
            self.api.upload_folder(
                folder_path=str(local),
                path_in_repo=remote,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=message,
            )
            return f"Hugging Face uploaded {remote} in {time.time() - started:.0f}s"
        except Exception as error:
            return f"Hugging Face upload failed: {type(error).__name__}: {error}"

    def rotate_checkpoints(self, keep: int) -> list[str]:
        if not self.enabled:
            return []
        messages = []
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="model")
            epochs = sorted({
                path.split("/")[1]
                for path in files
                if path.startswith("checkpoints/epoch_") and len(path.split("/")) > 2
            })
            for epoch in epochs[:-keep]:
                remote = f"checkpoints/{epoch}"
                self.api.delete_folder(
                    remote,
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Rotate old checkpoint {epoch}",
                )
                messages.append(f"Hugging Face removed {remote}")
        except Exception as error:
            messages.append(f"Hugging Face rotation failed: {type(error).__name__}: {error}")
        return messages

    def download_latest(self, destination: str | Path) -> Path | None:
        if not self.enabled:
            return None
        try:
            from huggingface_hub import hf_hub_download

            files = self.api.list_repo_files(self.repo_id, repo_type="model")
            epochs = sorted({
                path.split("/")[1]
                for path in files
                if path.startswith("checkpoints/epoch_") and path.endswith("state.msgpack")
            })
            if not epochs:
                return None
            epoch = epochs[-1]
            target = Path(destination) / epoch
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("state.msgpack", "metadata.json"):
                downloaded = hf_hub_download(
                    self.repo_id,
                    f"checkpoints/{epoch}/{filename}",
                    repo_type="model",
                    token=self.token,
                )
                shutil.copy2(downloaded, target / filename)
            return target
        except Exception as error:
            print(f"[hf] resume download failed: {type(error).__name__}: {error}", flush=True)
            return None

    def download_best(self, destination: str | Path) -> Path | None:
        if not self.enabled:
            return None
        try:
            from huggingface_hub import hf_hub_download

            target = Path(destination)
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("params.msgpack", "metadata.json"):
                downloaded = hf_hub_download(
                    self.repo_id,
                    f"best model/{filename}",
                    repo_type="model",
                    token=self.token,
                )
                shutil.copy2(downloaded, target / filename)
            return target
        except Exception as error:
            print(f"[hf] best-model download failed: {type(error).__name__}: {error}", flush=True)
            return None


class GitBackup:
    def __init__(self, repo_id: str, clone_dir: str | Path = "/root/repo_msc"):
        self.repo_id = repo_id
        self.clone_dir = Path(clone_dir)
        self.token = get_token("GITHUB_TOKEN")

    def _run(self, *arguments: str, cwd: Path | None = None) -> tuple[int, str]:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=cwd or self.clone_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            output = result.stdout + result.stderr
            if self.token:
                output = output.replace(self.token, "***")
            return result.returncode, output.strip()
        except Exception as error:
            return 1, f"{type(error).__name__}: {error}"

    def ensure_clone(self) -> str:
        if (self.clone_dir / ".git").exists():
            return "GitHub clone ready"
        if not self.token:
            return "GitHub backup disabled (token unavailable)"
        authenticated = f"https://x-access-token:{self.token}@github.com/{self.repo_id}.git"
        code, output = self._run(
            "clone", "--depth", "1", authenticated, str(self.clone_dir), cwd=Path("/root")
        )
        if code:
            return f"GitHub clone failed: {output[-500:]}"
        self._run("remote", "set-url", "origin", f"https://github.com/{self.repo_id}.git")
        return "GitHub clone created"

    def push(self, subdir: str, message: str) -> str:
        ready = self.ensure_clone()
        if "failed" in ready or "disabled" in ready:
            return ready
        destination = self.clone_dir / subdir
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            EXPERIMENT_ROOT,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", ".pytest_cache", "*.pyc", ".git", "logs"
            ),
        )
        source_logs = EXPERIMENT_ROOT / "logs"
        if source_logs.exists():
            shutil.copytree(source_logs, destination / "logs", dirs_exist_ok=True)
        authenticated = f"https://x-access-token:{self.token}@github.com/{self.repo_id}.git"
        public = f"https://github.com/{self.repo_id}.git"
        self._run("remote", "set-url", "origin", authenticated)
        try:
            self._run("config", "user.name", "YLiu95")
            self._run("config", "user.email", "yliu95@users.noreply.github.com")
            self._run("add", "--", subdir)
            code, output = self._run("commit", "-m", message)
            if code and "nothing to commit" in output:
                return "GitHub: nothing new to commit"
            if code:
                return f"GitHub commit failed: {output[-500:]}"
            code, output = self._run("push", "origin", "HEAD:main")
            return "GitHub backup uploaded" if code == 0 else f"GitHub push failed: {output[-500:]}"
        finally:
            self._run("remote", "set-url", "origin", public)