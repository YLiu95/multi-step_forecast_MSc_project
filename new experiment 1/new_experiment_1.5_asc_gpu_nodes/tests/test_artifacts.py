import pytest

from ascgpu.artifacts import checkpoint_directory, final_files, git_blob_hash
from ascgpu.data import write_json


def test_only_complete_checkpoints_and_allowed_artifacts(tmp_path):
    directory = tmp_path / "checkpoints" / "step_1"
    directory.mkdir(parents=True)
    write_json(directory / "state.json", {"tp": 1})
    write_json(directory / "config.json", {})
    write_json(directory / "partitions.json", {})
    (directory / "tp-00.pt").write_bytes(b"resumable")
    (directory / "weights-00.safetensors").write_bytes(b"weights")
    for label in ("best", "latest"):
        write_json(tmp_path / f"{label}.json", {"directory": "checkpoints/step_1"})
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint_directory(tmp_path, "latest")
    (directory / "COMPLETE").touch()
    log = tmp_path / "runs" / "pilot" / "events.out.tfevents.test"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"events")
    (tmp_path / "reports").mkdir()
    write_json(tmp_path / "reports" / "backup_verification.json", {"previous_verification": True})
    (tmp_path / ".env").write_text("not-a-real-secret")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "private.parquet").write_bytes(b"private")
    files = final_files(tmp_path)
    assert set(files) == {"latest/tp-00.pt", "best/weights-00.safetensors", "latest/state.json", "best/state.json",
                          "latest/config.json", "best/config.json", "latest/partitions.json", "best/partitions.json",
                          "runs/pilot/events.out.tfevents.test"}
    assert len(git_blob_hash(log)) == 40


def test_checkpoint_pointer_cannot_escape_checkpoint_directory(tmp_path):
    write_json(tmp_path / "latest.json", {"directory": "../elsewhere"})
    with pytest.raises(ValueError):
        checkpoint_directory(tmp_path, "latest")