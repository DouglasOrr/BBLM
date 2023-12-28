import torch

import bblm.model as M


def test_model() -> None:
    model = M.Model(
        M.Model.Settings(
            hidden_size=32,
            n_heads=5,
            head_size=8,
            sequence_length=80,
            ffn_size=64,
            depth=3,
        )
    )
    opt = torch.optim.SGD(model.parameters(), 1e-4)
    batch = torch.randint(0, 256, (7, 29), dtype=torch.uint8)
    loss_0 = model(batch)
    assert 0 < loss_0 < torch.log(torch.tensor(4 * 256))

    # Gradients work, we can step to decrease the loss
    loss_0.backward()
    opt.step()
    loss_1 = model(batch)
    assert loss_1 < loss_0
