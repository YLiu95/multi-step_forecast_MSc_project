from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn import functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.multiprocessing.errors import record

from .data import write_json
from .model import ForecastModel, ModelConfig, ParallelContext, merge_shards


@record
def check(tp: int, device: str):
    context = ParallelContext.initialize(tp, device)
    config = ModelConfig(width=32, heads=8, feedforward=64, temporal_depth=1, cross_depth=1,
                         window=16, patch=4, basket=4, tickers=19, rematerialize=False)
    torch.manual_seed(41)
    model = ForecastModel(config, context)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    gathered = [None] * tp
    dist.all_gather_object(gathered, state, group=context.tp_group)
    reference = ForecastModel(config, ParallelContext(device=context.device))
    reference.load_state_dict(merge_shards(gathered, model.partition_metadata()))
    inputs = torch.randn(2, 4, 16, device=context.device)
    dist.broadcast(inputs, src=context.dp_rank * tp, group=context.tp_group)
    tickers = torch.tensor([[1, 5, 7, 2], [8, 3, 6, 9]], device=context.device)
    positions = torch.tensor([2, 0], device=context.device)
    labels = torch.tensor([-1.0, 0.5], device=context.device)
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with precision:
        output = model(inputs, tickers, positions)
        expected = reference(inputs, tickers, positions)
        loss = functional.huber_loss(output, labels)
        reference_loss = functional.huber_loss(expected, labels)
    tolerance = 0.04 if device == "cuda" else 2e-5
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    loss.backward()
    reference_loss.backward()
    gradients = {name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()}
    dist.all_gather_object(gathered, gradients, group=context.tp_group)
    merged = merge_shards(gathered, model.partition_metadata())
    largest_error = 0.0
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(merged[name], parameter.grad.cpu(), rtol=tolerance, atol=tolerance)
        largest_error = max(largest_error, float((merged[name] - parameter.grad.cpu()).abs().max()))
    permutation = torch.tensor([2, 0, 3, 1], device=context.device)
    inverse = torch.argsort(permutation)
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()):
        permuted = model(inputs[:, permutation], tickers[:, permutation], inverse[positions])
    torch.testing.assert_close(output, permuted, rtol=tolerance, atol=tolerance)
    reduced = torch.tensor(float(context.dp_rank + 1), device=context.device)
    dist.all_reduce(reduced, group=context.dp_group)
    assert reduced.item() == context.dp_size * (context.dp_size + 1) / 2
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    wrapped = DistributedDataParallel(model, process_group=context.dp_group,
                                      device_ids=[context.local_rank] if device == "cuda" else None,
                                      broadcast_buffers=False, gradient_as_bucket_view=True)
    replica_inputs = inputs + context.dp_rank * 0.1
    with torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        distributed_loss = functional.huber_loss(wrapped(replica_inputs, tickers, positions), labels)
        expected_loss = functional.huber_loss(reference(replica_inputs, tickers, positions), labels)
    distributed_loss.backward()
    expected_loss.backward()
    for parameter in reference.parameters():
        dist.all_reduce(parameter.grad, group=context.dp_group)
        parameter.grad.div_(context.dp_size)
    gradients = {name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()}
    dist.all_gather_object(gathered, gradients, group=context.tp_group)
    merged = merge_shards(gathered, model.partition_metadata())
    ddp_error = 0.0
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(merged[name], parameter.grad.cpu(), rtol=tolerance, atol=tolerance)
        ddp_error = max(ddp_error, float((merged[name] - parameter.grad.cpu()).abs().max()))
    if context.rank == 0:
        result = {"status": "passed", "tp": tp, "dp": context.dp_size, "device": device,
                  "forward_max_error": float((output - expected).abs().max()),
                  "gradient_max_error": largest_error, "parameters": model.parameter_count(),
                  "ddp_gradient_max_error": ddp_error,
                  "config": asdict(config), "job_id": os.environ.get("SLURM_JOB_ID")}
    context.close()
    if context.rank == 0:
        result["clean_shutdown"] = True
        if os.environ.get("ARTIFACT_ROOT"):
            write_json(Path(os.environ["ARTIFACT_ROOT"]) / "reports" / f"parallel-check-{device}-tp{tp}-dp{context.dp_size}.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    arguments = parser.parse_args()
    check(arguments.tp, arguments.device)