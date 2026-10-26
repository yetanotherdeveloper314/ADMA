import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import train


def test_merge_all_datasets_uses_local_paths_and_rebuilds_splits(tmp_path, monkeypatch):
    source = tmp_path / "source"
    train_images = source / "train" / "images"
    train_labels = source / "train" / "labels"
    val_images = source / "valid" / "images"
    val_labels = source / "valid" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True)

    label_content = (
        "0 0.5 0.5 0.25 0.25\n"
        "1 0.5 0.5 0.25 0.25\n"
        "2 0.5 0.5 0.25 0.25\n"
        "3 0.5 0.5 0.25 0.25\n"
        "4 0.5 0.5 0.25 0.25\n"
        "5 0.5 0.5 0.25 0.25\n"
    )
    for image_dir, label_dir, stem in (
        (train_images, train_labels, "train_sample"),
        (val_images, val_labels, "val_sample"),
    ):
        (image_dir / f"{stem}.jpg").write_bytes(b"image")
        (label_dir / f"{stem}.txt").write_text(label_content)

    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "C:/old/checkout/source",
                "train": "train/images",
                "val": "valid/images",
                "names": ["Aircraft", "Drone", "Person", "Tank", "Warship", "Weapons"],
            }
        )
    )

    output = tmp_path / "combined"
    monkeypatch.setattr(train, "DATASETS", {"military_objects": {"location": str(source)}})
    monkeypatch.setattr(train, "COMBINED_DIR", output)

    result = train._merge_all_datasets()

    assert result == output / "data.yaml"
    assert sorted(
        path.name for path in (output / "images" / "train" / "military_objects").iterdir()
    ) == ["000000.jpg"]
    assert sorted(
        path.name for path in (output / "images" / "val" / "military_objects").iterdir()
    ) == ["000000.jpg"]
    assert (output / "labels" / "train" / "military_objects" / "000000.txt").read_text() == (
        "1 0.5 0.5 0.25 0.25\n0 0.5 0.5 0.25 0.25\n2 0.5 0.5 0.25 0.25\n3 0.5 0.5 0.25 0.25\n"
    )


def test_io_path_adds_windows_long_path_prefix(monkeypatch):
    monkeypatch.setattr(train.sys, "platform", "win32")
    long_path = train.Path("C:/") / ("nested" * 45) / "labels.txt"

    assert train._io_path(long_path).startswith("\\\\?\\")


def _prepare_training(tmp_path, monkeypatch, evaluator_runner=None):
    events = []
    active = {"run": None}

    class FakeModel:
        def __init__(self, weights):
            events.append(("model", weights))
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def train(self, **kwargs):
            events.append(("train", kwargs))
            assert train.os.environ["MLFLOW_KEEP_RUN_ACTIVE"] == "True"
            active["run"] = object()
            for callback in self.callbacks["on_train_start"]:
                callback(object())

    mlflow = SimpleNamespace(
        active_run=lambda: active["run"],
        log_params=lambda params: events.append(("params", params)),
        log_artifact=lambda path, artifact_path: events.append(("artifact", path, artifact_path)),
        end_run=lambda: events.append(("end_run",)),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", mlflow)
    monkeypatch.setattr(train, "YOLO", FakeModel)
    monkeypatch.setattr(train, "_git_commit", lambda: "abc123")

    run_best = tmp_path / "run" / "weights" / "best.pt"
    run_best.parent.mkdir(parents=True)
    run_best.write_bytes(b"checkpoint")
    monkeypatch.setattr(train, "_find_best_pt", lambda project, name: run_best)
    monkeypatch.setattr(train, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train, "MODELS_DIR", str(tmp_path / "models"))

    def run_evaluator(command, **kwargs):
        if evaluator_runner:
            return evaluator_runner(command, **kwargs)
        events.append(("evaluator", command))
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train.subprocess, "run", run_evaluator)
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir()
    data_yaml.write_text("names: [tank]\n")
    return events, data_yaml, run_best


def test_train_logs_metadata_evaluates_checkpoint_and_closes_mlflow(tmp_path, monkeypatch):
    events, data_yaml, run_best = _prepare_training(tmp_path, monkeypatch)

    result = train.train(data_yaml, "experiment", seed=123)

    train_call = next(value for key, value in events if key == "train")
    metadata = next(value for key, value in events if key == "params")
    artifact = next(value for value in events if value[0] == "artifact")
    evaluator_command = next(value for key, value in events if key == "evaluator")
    assert train_call["seed"] == 123
    assert train_call["data"] == str(data_yaml.resolve())
    assert metadata == {
        "data_yaml": str(data_yaml.resolve()),
        "seed": 123,
        "git_commit": "abc123",
    }
    assert evaluator_command[evaluator_command.index("--data") + 1] == str(data_yaml.resolve())
    assert evaluator_command[evaluator_command.index("--model") + 1] == str(run_best.resolve())
    assert Path(evaluator_command[evaluator_command.index("--output-dir") + 1]) == (
        tmp_path / "runs" / "eval" / "experiment"
    )
    assert artifact[2] == "evaluation"
    assert events[-1] == ("end_run",)
    assert result == Path(tmp_path / "models" / "experiment" / "best.pt")


def test_train_closes_mlflow_when_evaluator_fails(tmp_path, monkeypatch):
    def fail_evaluator(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    events, data_yaml, _ = _prepare_training(tmp_path, monkeypatch, evaluator_runner=fail_evaluator)

    with pytest.raises(subprocess.CalledProcessError):
        train.train(data_yaml, "experiment")

    assert events[-1] == ("end_run",)
