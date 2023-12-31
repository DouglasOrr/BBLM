from pathlib import Path

import torch

import bblm.datasets as D


def test_data() -> None:
    root = Path(__file__).parent / "wikitext_tiny"
    data = D.load_wikitext_103_raw(root)
    for part in ["train", "valid", "test"]:
        expected = (root / f"{part}.txt").read_bytes()
        actual = bytes(data.parts[part].tolist())
        assert actual == expected

    seed = 45892523
    batch = next(iter(data.batches("valid", 3, 64, seed=seed)))
    assert batch.shape == (3, 64)
    assert batch.dtype == torch.uint8
    valid_str = (root / "valid.txt").read_bytes()
    for sequence in batch:
        assert bytes(sequence) in valid_str

    # Same seed -> same batch, different seed -> (probably) different batch
    assert torch.equal(batch, next(iter(data.batches("valid", 3, 64, seed=seed))))
    assert not torch.equal(
        batch, next(iter(data.batches("valid", 3, 64, seed=seed + 1)))
    )

    # No seed
    batch = next(iter(data.batches("valid", 7, 64, seed=None)))
    assert batch.shape == (7, 64)
