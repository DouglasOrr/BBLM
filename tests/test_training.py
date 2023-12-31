import json
from pathlib import Path
from typing import Any

import pytest
import torch

import bblm as B


def _run_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> B.training.ExperimentSettings:
    monkeypatch.setenv("WANDB_MODE", "disabled")  # for speed (otherwise, "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    settings = B.training.ExperimentSettings(
        "test",
        model=B.models.Model.Settings(
            hidden_size=64,
            n_heads=2,
            head_size=32,
            sequence_length=128,
            ffn_size=256,
            depth=3,
        ),
        training=B.training.TrainingSettings(
            lr=1e-2,
            batch_size=4,
            sequence_length=128,
            steps=12,
            valid_interval=6,
            valid_batches=2,
        ),
        execution=B.training.ExecutionSettings(
            device="cpu",
            dtype="float32",
            log_interval=2,
        ),
        logging=B.training.LoggingSettings(
            stderr=True,
            local=str(tmp_path / "log.jsonl"),
            checkpoint=str(tmp_path / "model.pt"),
            wandb="test-bblm",
        ),
    )
    data_path = Path(__file__).parent / "wikitext_tiny"
    B.training.run_experiment(settings, str(data_path))
    return settings


def test_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _run_experiment(tmp_path, monkeypatch)

    # Log file looks sensible
    header, *log = [json.loads(line) for line in (tmp_path / "log.jsonl").open("r")]

    assert header["model"]["hidden_size"] == 64
    assert "logging" not in header

    assert [x["step"] for x in log] == [0, 2, 4, 6, 8, 10, 12]
    assert all("lr" in x for x in log)
    assert all("duration" in x for x in log)
    assert all("loss" in x for x in log[:-1])
    assert log[6]["valid_loss"] < log[3]["valid_loss"] < log[0]["valid_loss"]

    # Model checkpoint is loadable
    model = B.models.Model(settings.model)
    model.load_state_dict(torch.load(tmp_path / "model.pt"))


def test_experiment_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _forward_with_error(*args: Any, **kwargs: Any) -> None:
        raise ValueError("this is a test")

    monkeypatch.setattr("bblm.models.Model.forward", _forward_with_error)
    with pytest.raises(ValueError):
        _run_experiment(tmp_path, monkeypatch)

    log = [json.loads(line) for line in (tmp_path / "log.jsonl").open("r")]

    assert log[-1]["error"] == repr(ValueError("this is a test"))
    assert "_forward_with_error" in log[-1]["error_trace"]
