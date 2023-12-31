"""Training loop."""

import contextlib
import dataclasses
import datetime
import json
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import torch
from torch import nn

from . import datasets, models

# Training API


@dataclass
class TrainingSettings:
    lr: float
    batch_size: int
    sequence_length: int
    steps: int
    valid_interval: int
    valid_batches: int


def train(
    model: nn.Module, data: datasets.Data, s: TrainingSettings
) -> Iterable[Dict[str, Any]]:
    """A full training run, lazily yielding a log describing progress."""

    opt = torch.optim.Adam(model.parameters(), lr=s.lr)
    schedule = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=s.steps)
    (device,) = {p.device for p in model.parameters()}
    valid_batches = list(
        islice(
            data.batches("valid", s.batch_size, s.sequence_length, seed=7395223495),
            s.valid_batches,
        )
    )
    train_batches = islice(
        data.batches("train", s.batch_size, s.sequence_length, seed=None), s.steps
    )
    t_last = time.time()
    for step, batch in enumerate(chain(train_batches, [None])):
        log: Dict[str, Any] = dict(step=step)
        if step % s.valid_interval == 0 or batch is None:
            with torch.no_grad():
                model.eval()
                valid_loss = torch.stack(
                    [model(batch.to(device)) for batch in valid_batches]
                ).mean()
                model.train()
            log.update(valid_loss=float(valid_loss))
        if batch is not None:
            opt.zero_grad()
            loss = model(batch.to(device))
            loss.backward()
            opt.step()
            log.update(loss=float(loss))
        t = time.time()
        log.update(duration=t - t_last, lr=schedule.get_last_lr()[0])
        yield log
        t_last = t
        schedule.step()


# Experiment API


def aggregate_log(
    log: Iterable[Dict[str, Any]], count: int
) -> Iterable[Dict[str, Any]]:
    """Aggregate the log over a window, using:

    step -- first step index in window
    valid_loss -- first (only) in window
    loss -- mean over window
    duration -- sum over window
    lr -- first in window
    """
    agg, agg_count = None, 0
    for item in log:
        if agg is None:
            agg, agg_count = item.copy(), 1
        else:
            assert "valid_loss" not in item
            # Only aggregate: training loss (mean) and duration (sum)
            if "loss" in item:
                agg["loss"] = (agg_count * agg["loss"] + item["loss"]) / (agg_count + 1)
            agg["duration"] += item.get("duration", 0)
            agg_count += 1
        if agg_count == count:
            yield agg
            agg, agg_count = None, 0
    if agg is not None:
        yield agg


@dataclass
class ExecutionSettings:
    device: str
    dtype: str
    log_interval: int


@dataclass
class LoggingSettings:
    stderr: bool
    local: Optional[str]
    checkpoint: Optional[str]
    wandb: Optional[str]  # project name


@dataclass
class ExperimentSettings:
    experiment: str
    model: models.Model.Settings
    training: TrainingSettings
    execution: ExecutionSettings
    logging: LoggingSettings


def get_stats(model: nn.Module) -> Dict[str, Any]:
    """Get auxilliary statistics (measured, not configured) at the start of training."""
    n_parameters = sum(p.nelement() for p in model.parameters())
    (device,) = {p.device for p in model.parameters()}
    device_name = (
        torch.cuda.get_device_name()
        if device.type == "cuda"
        else f"cpu-{torch.get_num_threads()}"
    )
    return dict(
        n_parameters=n_parameters,
        device_name=device_name,
    )


_Logger = Callable[[Dict[str, Any]], None]


def _stderr_logger(config: Dict[str, Any], stats: Dict[str, Any]) -> _Logger:
    print(config, file=sys.stderr)
    print(stats, file=sys.stderr)
    return lambda log: print(log, file=sys.stderr)


@contextlib.contextmanager
def _jsonl_logger(
    path: Path, config: Dict[str, Any], stats: Dict[str, Any]
) -> Iterator[_Logger]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:

        def _log(log: Dict[str, Any]) -> None:
            f.write(json.dumps(log) + "\n")

        _log(dict(**config, **stats, start_time=datetime.datetime.now().isoformat()))
        try:
            yield _log
        except Exception as error:
            _log(dict(error=repr(error), error_trace=traceback.format_exc()))
            raise error


@contextlib.contextmanager
def _wandb_logger(
    project: str, config: Dict[str, Any], stats: Dict[str, Any]
) -> Iterator[_Logger]:
    import wandb  # pylint:disable=import-outside-toplevel

    wandb.init(project=project, config=config, reinit=True)
    assert wandb.run
    wandb.run.summary.update(stats)
    try:
        yield wandb.log
        wandb.finish(0)
    except Exception as error:
        wandb.run.summary.update(
            dict(error=repr(error), error_trace=traceback.format_exc())
        )
        wandb.finish(1)
        raise error


def run_experiment(s: ExperimentSettings, data_path: str) -> None:
    """High-level single-experiment runner."""
    data = datasets.load_wikitext_103_raw(data_path)
    model = models.Model(s.model).to(
        device=torch.device(s.execution.device),
        dtype=getattr(torch, s.execution.dtype),
    )
    config = dataclasses.asdict(s)
    del config["logging"]
    stats = get_stats(model)
    with contextlib.ExitStack() as stack:
        loggers = []
        if s.logging.stderr:
            loggers.append(_stderr_logger(config, stats))
        if s.logging.local:
            loggers.append(
                stack.enter_context(_jsonl_logger(Path(s.logging.local), config, stats))
            )
        if s.logging.wandb:
            loggers.append(
                stack.enter_context(_wandb_logger(s.logging.wandb, config, stats))
            )

        # Run training
        for log in aggregate_log(
            train(model, data, s.training), s.execution.log_interval
        ):
            for logger in loggers:
                logger(log)
            if "valid_loss" in log and s.logging.checkpoint:
                Path(s.logging.checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), s.logging.checkpoint)
