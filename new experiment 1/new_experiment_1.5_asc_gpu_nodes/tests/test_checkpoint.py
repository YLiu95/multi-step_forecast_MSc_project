import torch

from ascgpu.model import ForecastModel, ModelConfig, ParallelContext
from ascgpu.train import restore_checkpoint, save_checkpoint


def test_full_optimizer_state_can_be_restored(tmp_path):
    config = ModelConfig(width=16, heads=4, feedforward=32, temporal_depth=1, cross_depth=1,
                         window=8, patch=4, basket=3, tickers=7, rematerialize=False)
    model = ForecastModel(config, ParallelContext())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3, 8)
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    positions = torch.tensor([0, 2])
    model(inputs, ids, positions).square().mean().backward()
    optimizer.step()
    state = {"step": 1, "samples": 2, "tp": 1, "dp": 1}
    save_checkpoint(tmp_path, model, optimizer, state, True)
    expected = model(inputs, ids, positions).detach()
    replacement = ForecastModel(config, ParallelContext())
    replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3)
    restored = restore_checkpoint(tmp_path / "checkpoints" / "step_00000001", replacement, replacement_optimizer)
    assert restored == state
    torch.testing.assert_close(replacement(inputs, ids, positions), expected)
    assert len(replacement_optimizer.state) == len(optimizer.state)