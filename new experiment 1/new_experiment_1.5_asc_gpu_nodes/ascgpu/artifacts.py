from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
import torch

from .data import write_json

REPOSITORY = "YL95/experiment-1.5-asc-gpu-nodes"


def authenticated_api() -> HfApi:
    load_dotenv(Path.home() / ".env", override=False)
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is missing")
    return HfApi(token=os.environ["HF_TOKEN"])


def probe(root: Path) -> dict:
    directory = root / "transfer_probe"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    generator = torch.Generator().manual_seed(time.time_ns() % (2**63 - 1))
    for shard in range(8):
        path = directory / f"synthetic-{shard}.pt"
        torch.save({"synthetic_transfer_probe_not_a_model": torch.randn(16 * 1024 * 1024, generator=generator)}, path)
        paths.append(path)
    api = authenticated_api()
    operations = [CommitOperationAdd(path_in_repo=f"readiness/{path.name}", path_or_fileobj=str(path)) for path in paths]
    started = time.monotonic()
    commit = api.create_commit(repo_id=REPOSITORY, operations=operations, num_threads=8,
                               commit_message="Measure temporary eight-shard checkpoint transfer")
    elapsed = time.monotonic() - started
    entries = {entry.rfilename: entry for entry in api.model_info(REPOSITORY, revision=commit.oid, files_metadata=True).siblings}
    for operation in operations:
        entry = entries[operation.path_in_repo]
        if entry.lfs.sha256 != operation.upload_info.sha256.hex():
            raise RuntimeError("Upload probe digest did not match")
    total = sum(path.stat().st_size for path in paths)
    result = {"bytes": total, "seconds": elapsed, "shards": 8,
              "effective_bytes_per_second": total / elapsed, "commit": commit.oid,
              "remote_digests_verified": True, "includes_hashing_overhead": False}
    api.create_commit(repo_id=REPOSITORY,
                      operations=[CommitOperationDelete(path_in_repo=operation.path_in_repo) for operation in operations],
                      commit_message="Remove temporary transfer probes; retain measurement report")
    for path in paths:
        path.unlink()
    write_json(root / "reports" / "upload_probe.json", result)
    print(json.dumps(result), flush=True)
    return result


def checkpoint_directory(root: Path, label: str) -> Path:
    pointer = json.loads((root / f"{label}.json").read_text())
    directory = (root / pointer["directory"]).resolve()
    directory.relative_to((root / "checkpoints").resolve())
    if not (directory / "COMPLETE").exists():
        raise ValueError(f"Incomplete {label} checkpoint")
    return directory


def final_files(root: Path) -> dict[str, Path]:
    result = {}
    for label, pattern in (("latest", "tp-*.pt"), ("best", "weights-*.safetensors")):
        directory = checkpoint_directory(root, label)
        state = json.loads((directory / "state.json").read_text())
        shards = sorted(directory.glob(pattern))
        if len(shards) != state["tp"]:
            raise ValueError(f"{label} checkpoint is missing tensor-parallel shards")
        for path in shards:
            result[f"{label}/{path.name}"] = path
        for name in ("state.json", "config.json", "partitions.json"):
            result[f"{label}/{name}"] = directory / name
    for path in (root / "reports").glob("*.json"):
        result[f"reports/{path.name}"] = path
    for path in (root / "runs").rglob("events.out.tfevents.*"):
        result[path.relative_to(root).as_posix()] = path
    for name in ("config.json", "README.md"):
        path = root / name
        if path.exists():
            result[name] = path
    history = root / "history.jsonl"
    if history.exists():
        result["reports/history.jsonl"] = history
    vocabulary = root / "panel" / "vocabulary.json"
    if vocabulary.exists():
        result["vocabulary.json"] = vocabulary
    if not any(name.startswith("runs/") for name in result):
        raise ValueError("No TensorBoard event files found")
    if any(path.name == ".env" or "cache" in path.relative_to(root).parts for path in result.values()):
        raise ValueError("A private file escaped the artifact allowlist")
    return result


def git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(root: Path) -> dict:
    template = Path(__file__).resolve().parents[1] / "MODEL_CARD.md"
    if template.exists():
        (root / "README.md").write_text(template.read_text())
    paths = final_files(root)
    api = authenticated_api()
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in paths.items()]
    total = sum(path.stat().st_size for path in paths.values())
    print(json.dumps({"stage": "uploading_final", "files": len(paths), "bytes": total}), flush=True)
    started = time.monotonic()
    commit = api.create_commit(repo_id=REPOSITORY, operations=operations,
                               commit_message="Back up pilot best, latest resumable state, metrics and provenance",
                               num_threads=8)
    remote = {entry.rfilename: entry for entry in api.model_info(REPOSITORY, revision=commit.oid, files_metadata=True).siblings}
    verified = []
    for operation in operations:
        name = operation.path_in_repo
        entry = remote.get(name)
        path = paths[name]
        if entry is None or entry.size != path.stat().st_size:
            raise RuntimeError(f"Remote size verification failed for {name}")
        if entry.lfs is not None:
            expected = operation.upload_info.sha256.hex()
            actual = entry.lfs.sha256
        else:
            expected = git_blob_hash(path)
            actual = entry.blob_id
        if actual != expected:
            raise RuntimeError(f"Remote digest verification failed for {name}")
        verified.append({"path": name, "bytes": entry.size, "digest": actual})
    result = {"repository": REPOSITORY, "commit": commit.oid, "bytes": total,
              "seconds": time.monotonic() - started, "verified_files": verified,
              "verified_at_unix": time.time(), "status": "verified"}
    write_json(root / "reports" / "backup_verification.json", result)
    api.upload_file(repo_id=REPOSITORY, path_or_fileobj=str(root / "reports" / "backup_verification.json"),
                    path_in_repo="reports/backup_verification.json", commit_message="Record verified checkpoint and event-file digests")
    print(json.dumps({"stage": "backup_verified", "commit": commit.oid, "files": len(verified), "bytes": total}), flush=True)
    return result


def publish_logs(root: Path) -> None:
    paths = list((root / "runs").rglob("events.out.tfevents.*"))
    if not paths:
        return
    staging = root / "log_upload_staging"
    staging.mkdir(exist_ok=True)
    operations = []
    for path in paths:
        relative = path.relative_to(root)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        operations.append(CommitOperationAdd(path_in_repo=relative.as_posix(), path_or_fileobj=str(destination)))
    authenticated_api().create_commit(repo_id=REPOSITORY, operations=operations,
                                      commit_message="Back up live aggregate TensorBoard metrics")


def watch(root: Path) -> None:
    event = threading.Event()
    previous_upload = 0.0
    print("Artifact watcher ready; waiting for checkpoint completion markers", flush=True)
    while not (root / "TRAINING_DONE.json").exists():
        if time.monotonic() - previous_upload >= 120:
            try:
                publish_logs(root)
                previous_upload = time.monotonic()
            except Exception as error:
                print(json.dumps({"stage": "log_upload_retry", "error_type": type(error).__name__}), flush=True)
        event.wait(10)
    publish(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe", "publish", "watch"])
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    {"probe": probe, "publish": publish, "watch": watch}[arguments.action](arguments.root)