from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import os

import torch
from torch import distributed as dist, nn
from torch.nn import functional as functional
from torch.utils.checkpoint import checkpoint


@dataclass
class ParallelContext:
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    rank: int = 0
    local_rank: int = 0
    tp_group: object = None
    dp_group: object = None
    device: torch.device = torch.device("cpu")
    owns_process_group: bool = False

    @classmethod
    def initialize(cls, tp_size: int = 8, device: str = "cuda") -> "ParallelContext":
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world % tp_size:
            raise ValueError("World size must be divisible by tensor parallel size")
        selected = torch.device(device, local_rank) if device == "cuda" else torch.device("cpu")
        if device == "cuda":
            torch.cuda.set_device(selected)
        dist.init_process_group("nccl" if device == "cuda" else "gloo", timeout=timedelta(minutes=10))
        context = cls(tp_size, rank % tp_size, world // tp_size, rank // tp_size, rank, local_rank, device=selected)
        context.owns_process_group = True
        for replica in range(context.dp_size):
            ranks = list(range(replica * tp_size, (replica + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                context.tp_group = group
        for shard in range(tp_size):
            ranks = list(range(shard, world, tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                context.dp_group = group
        return context

    def close(self):
        if not self.owns_process_group or not dist.is_initialized():
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        dist.barrier()
        for group in (self.dp_group, self.tp_group):
            if group is not None:
                dist.destroy_process_group(group)
            dist.barrier()
        dist.destroy_process_group()
        self.owns_process_group = False


class CopyToTensorGroup(torch.autograd.Function):
    @staticmethod
    def forward(context, values, group, size):
        context.group, context.size = group, size
        return values

    @staticmethod
    def backward(context, gradient):
        result = gradient.contiguous()
        if context.size > 1:
            result = result.clone()
            dist.all_reduce(result, group=context.group)
        return result, None, None


class ReduceFromTensorGroup(torch.autograd.Function):
    @staticmethod
    def forward(context, values, group, size):
        result = values.contiguous()
        if size > 1:
            result = result.clone()
            dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(context, gradient):
        return gradient, None, None


class GatherFromTensorGroup(torch.autograd.Function):
    @staticmethod
    def forward(context, values, group, size, rank):
        context.size, context.rank = size, rank
        if size == 1:
            return values
        pieces = [torch.empty_like(values) for _ in range(size)]
        dist.all_gather(pieces, values.contiguous(), group=group)
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def backward(context, gradient):
        return gradient.chunk(context.size, dim=-1)[context.rank].contiguous(), None, None, None


@dataclass
class ModelConfig:
    width: int = 3072
    heads: int = 48
    feedforward: int = 12288
    temporal_depth: int = 16
    cross_depth: int = 12
    window: int = 256
    patch: int = 8
    basket: int = 64
    tickers: int = 39260
    rematerialize: bool = True
    seed: int = 1337

    def validate(self, tp_size: int) -> None:
        if self.width % self.heads or self.heads % tp_size or self.feedforward % tp_size:
            raise ValueError("Attention and feed-forward dimensions must divide the TP mesh")
        if self.window % self.patch:
            raise ValueError("Patches must exactly partition the input history")


class ParameterFactory:
    def __init__(self, context: ParallelContext, seed: int):
        self.context = context
        self.seed = seed
        self.counter = 0

    def parameter(self, shape, kind=None, scale=0.02, zeros=False):
        values = torch.empty(shape, dtype=torch.float32, device=self.context.device)
        generator = torch.Generator(device=self.context.device)
        shard_seed = self.context.tp_rank if kind is not None else 0
        generator.manual_seed(self.seed + self.counter * 104729 + shard_seed * 1009)
        self.counter += 1
        if zeros:
            values.zero_()
        else:
            values.normal_(0, scale, generator=generator)
        parameter = nn.Parameter(values)
        parameter.tp_kind = kind
        return parameter


class ColumnLinear(nn.Module):
    def __init__(self, inputs, outputs, factory):
        super().__init__()
        self.context = factory.context
        local_outputs = outputs // self.context.tp_size
        self.weight = factory.parameter((local_outputs, inputs), 0, scale=inputs**-0.5)
        self.bias = factory.parameter((local_outputs,), 0, zeros=True)

    def forward(self, values):
        copied = CopyToTensorGroup.apply(values, self.context.tp_group, self.context.tp_size)
        return functional.linear(copied, self.weight, self.bias)


class RowLinear(nn.Module):
    def __init__(self, inputs, outputs, factory):
        super().__init__()
        self.context = factory.context
        self.weight = factory.parameter((outputs, inputs // self.context.tp_size), 1, scale=inputs**-0.5)
        self.bias = factory.parameter((outputs,), None, zeros=True)

    def forward(self, values):
        local = functional.linear(values, self.weight)
        result = ReduceFromTensorGroup.apply(local, self.context.tp_group, self.context.tp_size)
        return result + self.bias.to(result.dtype)


class TensorAttention(nn.Module):
    def __init__(self, config, factory):
        super().__init__()
        self.context = factory.context
        self.local_heads = config.heads // self.context.tp_size
        self.head_width = config.width // config.heads
        local_width = config.width // self.context.tp_size
        self.qkv_weight = factory.parameter((3 * local_width, config.width), "qkv", scale=config.width**-0.5)
        self.qkv_bias = factory.parameter((3 * local_width,), "qkv", zeros=True)
        self.output = RowLinear(config.width, config.width, factory)

    def forward(self, values):
        batch, length, _ = values.shape
        copied = CopyToTensorGroup.apply(values, self.context.tp_group, self.context.tp_size)
        projected = functional.linear(copied, self.qkv_weight, self.qkv_bias)
        query, key, value = projected.reshape(batch, length, 3, self.local_heads, self.head_width).unbind(2)
        attended = functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0,
        )
        return self.output(attended.transpose(1, 2).contiguous().reshape(batch, length, -1))


class EncoderBlock(nn.Module):
    def __init__(self, config, factory):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width, device=factory.context.device)
        self.attention = TensorAttention(config, factory)
        self.norm_feedforward = nn.LayerNorm(config.width, device=factory.context.device)
        self.expand = ColumnLinear(config.width, config.feedforward, factory)
        self.contract = RowLinear(config.feedforward, config.width, factory)

    def forward(self, values):
        values = values + self.attention(self.norm_attention(values))
        return values + self.contract(functional.gelu(self.expand(self.norm_feedforward(values)), approximate="tanh"))


class ForecastModel(nn.Module):
    def __init__(self, config: ModelConfig, context: ParallelContext):
        super().__init__()
        config.validate(context.tp_size)
        self.config, self.context = config, context
        factory = ParameterFactory(context, config.seed)
        local_width = config.width // context.tp_size
        self.patch_projection = ColumnLinear(config.patch, config.width, factory)
        self.ticker_embedding = factory.parameter((config.tickers, local_width), 1)
        self.role_embedding = factory.parameter((2, local_width), 1)
        self.position_embedding = factory.parameter((config.window // config.patch, local_width), 1)
        self.temporal_blocks = nn.ModuleList([EncoderBlock(config, factory) for _ in range(config.temporal_depth)])
        self.temporal_norm = nn.LayerNorm(config.width, device=context.device)
        self.cross_blocks = nn.ModuleList([EncoderBlock(config, factory) for _ in range(config.cross_depth)])
        self.cross_norm = nn.LayerNorm(config.width, device=context.device)
        self.head_norm = nn.LayerNorm(2 * config.width, device=context.device)
        self.head_hidden = ColumnLinear(2 * config.width, config.width, factory)
        self.head_output = RowLinear(config.width, 1, factory)

    def gather(self, values):
        return GatherFromTensorGroup.apply(values, self.context.tp_group, self.context.tp_size, self.context.tp_rank)

    def stack(self, values, blocks):
        for block in blocks:
            if self.config.rematerialize and self.training:
                values = checkpoint(block, values, use_reentrant=False)
            else:
                values = block(values)
        return values

    def forward(self, inputs, ticker_ids, target_position):
        batch, basket, window = inputs.shape
        if window != self.config.window or basket != self.config.basket:
            raise ValueError("Input shape differs from the model contract")
        patches = inputs.reshape(batch, basket, window // self.config.patch, self.config.patch)
        tokens = self.patch_projection(patches)
        roles = torch.zeros_like(ticker_ids)
        roles.scatter_(1, target_position[:, None], 1)
        identity = functional.embedding(ticker_ids, self.ticker_embedding)
        role = functional.embedding(roles, self.role_embedding)
        tokens = tokens + identity[:, :, None, :].to(tokens.dtype) + role[:, :, None, :].to(tokens.dtype)
        tokens = self.gather(tokens + self.position_embedding[None, None, :, :].to(tokens.dtype))
        tokens = tokens.reshape(batch * basket, -1, self.config.width)
        tokens = self.stack(tokens, self.temporal_blocks)
        states = self.temporal_norm(tokens).mean(dim=1).reshape(batch, basket, self.config.width)
        states = self.cross_norm(self.stack(states, self.cross_blocks))
        rows = torch.arange(batch, device=inputs.device)
        target = states[rows, target_position]
        target_id = ticker_ids[rows, target_position]
        target_identity = self.gather(functional.embedding(target_id, self.ticker_embedding)).to(target.dtype)
        combined = self.head_norm(torch.cat((target, target_identity), dim=-1))
        hidden = functional.gelu(self.head_hidden(combined), approximate="tanh")
        return self.head_output(hidden).squeeze(-1).float()

    def parameter_count(self):
        return sum(parameter.numel() * (self.context.tp_size if getattr(parameter, "tp_kind", None) is not None else 1)
                   for parameter in self.parameters())

    def partition_metadata(self):
        return {name: getattr(parameter, "tp_kind", None) for name, parameter in self.named_parameters()}


def merge_shards(shards: list[dict], kinds: dict) -> dict:
    result = {}
    for name, kind in kinds.items():
        pieces = [shard[name] for shard in shards]
        if kind is None:
            result[name] = pieces[0]
        elif kind == "qkv":
            reshaped = [piece.reshape(3, piece.shape[0] // 3, *piece.shape[1:]) for piece in pieces]
            merged = torch.cat(reshaped, dim=1)
            result[name] = merged.reshape(-1, *pieces[0].shape[1:])
        else:
            result[name] = torch.cat(pieces, dim=kind)
    return result


def clip_global_norm(model: ForecastModel, maximum: float) -> float:
    context = model.context
    squares = torch.zeros((), device=context.device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            factor = 1.0 if getattr(parameter, "tp_kind", None) is not None else 1.0 / context.tp_size
            squares += parameter.grad.detach().double().square().sum() * factor
    if context.tp_size > 1:
        dist.all_reduce(squares, group=context.tp_group)
    norm = squares.sqrt()
    coefficient = (maximum / (norm + 1e-6)).clamp(max=1.0).float()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient)
    return float(norm.item())