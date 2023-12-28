from pathlib import Path

import bblm as B


def test_training() -> None:
    data = B.data.load_wikitext_103_raw(Path(__file__).parent / "wikitext_tiny")
    model = B.model.Model(
        B.model.Model.Settings(
            hidden_size=64,
            n_heads=2,
            head_size=32,
            sequence_length=128,
            ffn_size=256,
            depth=3,
        )
    )
    log = list(
        B.training.train(
            model,
            data,
            B.training.TrainingSettings(
                lr=1e-2,
                batch_size=4,
                sequence_length=128,
                steps=10,
                valid_interval=5,
                valid_batches=2,
            ),
        )
    )
    assert [x["step"] for x in log] == list(range(11))
    assert all("lr" in x for x in log)
    assert all("duration" in x for x in log)
    assert all("loss" in x for x in log[:-1])
    assert log[10]["valid_loss"] < log[5]["valid_loss"] < log[0]["valid_loss"]
