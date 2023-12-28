"""Training loop."""

import time
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Dict, Iterable

import torch

from .data import Data
from .model import Model

# Training API


@dataclass
class TrainingSettings:
    lr: float
    batch_size: int
    sequence_length: int
    steps: int
    valid_interval: int
    valid_batches: int


def train(model: Model, data: Data, s: TrainingSettings) -> Iterable[Dict[str, Any]]:
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


# Experiment API (WIP)


@dataclass
class LoggingSettings:
    interval: int
    local: bool
    local_checkpoint: bool
    wandb: bool


@dataclass
class ExperimentSettings:
    experiment: str
    model: Model.Settings
    training: TrainingSettings
    logging: LoggingSettings
