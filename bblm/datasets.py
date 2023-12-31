"""Data loading and sampling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch
from torch import Tensor


@dataclass
class Data:
    """A dataset that has been split into "train", "valid", "test".

    parts -- {train|valid|test: tensor(N; uint8)} -- raw utf8 byte data
    """

    parts: Dict[str, Tensor]

    def batches(
        self, part: str, batch_size: int, sequence_length: int, seed: Optional[int]
    ) -> Iterable[Tensor]:
        """Sample an infinite stream of batches (with replacement)."""
        data = self.parts[part]
        generator = torch.Generator()
        if seed:
            generator.manual_seed(seed)
        else:
            generator.seed()
        while True:
            yield torch.stack(
                [
                    data[i : i + sequence_length]
                    for i in torch.randint(
                        len(data) - sequence_length, (batch_size,), generator=generator
                    )
                ]
            )


def load_wikitext_103_raw(path: Union[str, Path]) -> Data:
    """Load WikiText-103-raw into memory, for byte-language-modelling."""
    return Data(
        {
            k: torch.frombuffer(
                bytearray((Path(path) / f"{k}.txt").read_bytes()), dtype=torch.uint8
            )
            for k in ["train", "valid", "test"]
        }
    )
