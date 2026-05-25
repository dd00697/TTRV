import math
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class LowMemoryAdamW(Optimizer):
    """AdamW with chunked parameter updates to avoid full-size temporary tensors."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        chunk_size: int = 16_777_216,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size value: {chunk_size}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, chunk_size=chunk_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._cuda_graph_capture_health_check()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            chunk_size = int(group["chunk_size"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("LowMemoryAdamW does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2_sqrt = math.sqrt(1 - beta2**step)
                step_size = lr / bias_correction1

                param_flat = param.view(-1)
                exp_avg_flat = exp_avg.view(-1)
                exp_avg_sq_flat = exp_avg_sq.view(-1)
                for start in range(0, param_flat.numel(), chunk_size):
                    end = min(start + chunk_size, param_flat.numel())
                    denom = exp_avg_sq_flat[start:end].sqrt()
                    denom.div_(bias_correction2_sqrt).add_(eps)
                    param_flat[start:end].addcdiv_(exp_avg_flat[start:end], denom, value=-step_size)

        return loss
