# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


_GENERATE_INPUT_KEYS = {"input_ids", "attention_mask", "position_ids"}


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _visual_pruning_enabled(module) -> bool:
    enabled = False
    try:
        from src.ttrv_pruning.qwen_hf_pruning import is_visual_pruning_enabled
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        enabled = enabled or is_visual_pruning_enabled(module)
    try:
        from src.ttrv_pruning.mmtok_ttrv import is_mmtok_pruning_enabled
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        enabled = enabled or is_mmtok_pruning_enabled(module)
    try:
        from src.ttrv_pruning.visionzip_ttrv import is_visionzip_pruning_enabled
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        enabled = enabled or is_visionzip_pruning_enabled(module)
    return enabled


def _visual_pruning_context(module, *, stage: str, extra_info=None):
    stack = contextlib.ExitStack()
    try:
        from src.ttrv_pruning.qwen_hf_pruning import visual_pruning_context
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        stack.enter_context(visual_pruning_context(module, stage=stage, extra_info=extra_info))
    try:
        from src.ttrv_pruning.mmtok_ttrv import mmtok_pruning_context
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        stack.enter_context(mmtok_pruning_context(module, stage=stage, extra_info=extra_info))
    try:
        from src.ttrv_pruning.visionzip_ttrv import visionzip_pruning_context
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        stack.enter_context(visionzip_pruning_context(module, stage=stage, extra_info=extra_info))
    return stack


def _clear_mmtok_selection_cache(module, *, bump_scope: bool = True):
    try:
        from src.ttrv_pruning.mmtok_ttrv import clear_mmtok_selection_cache
    except (ImportError, ModuleNotFoundError):
        return
    clear_mmtok_selection_cache(module, bump_scope=bump_scope)


def _rollout_pruning_stage(prompts: DataProto, *, do_sample: bool) -> str:
    stage = prompts.meta_info.get("pruning_stage")
    if stage:
        return str(stage)
    return "rollout" if do_sample else "validation"


def _visual_pruning_last_forward(module):
    try:
        from src.ttrv_pruning.qwen_hf_pruning import last_pruned_forward_info
    except (ImportError, ModuleNotFoundError):
        return None
    return last_pruned_forward_info(module)


def _position_ids_for_model(position_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    if position_ids.dim() == 3 and position_ids.shape[0] == batch_size and position_ids.shape[1] == 3:
        return position_ids.transpose(0, 1)
    return position_ids


def _past_seq_length(past_key_values) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            return int(first_layer[0].shape[-2])
        if torch.is_tensor(first_layer):
            return int(first_layer.shape[-2])
    raise RuntimeError(f"cannot infer past_key_values sequence length from {type(past_key_values)!r}")


def _sample_next_token(logits: torch.Tensor, *, do_sample: bool, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.float()
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    if temperature is None or temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")
    logits = logits / temperature
    if top_k and top_k > 0 and top_k < logits.size(-1):
        kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _pad_generated_response(
    *,
    idx: torch.Tensor,
    response: torch.Tensor,
    position_ids: torch.Tensor,
    target_response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = idx.size(0)
    seq = torch.cat((idx, response), dim=1)
    current_response_length = response.size(1)
    if current_response_length >= target_response_length:
        return response, seq, position_ids

    pad_length = target_response_length - current_response_length
    pad_tokens = torch.full(
        (batch_size, pad_length),
        fill_value=pad_token_id,
        device=response.device,
        dtype=response.dtype,
    )
    response = torch.cat((response, pad_tokens), dim=1)
    seq = torch.cat((idx, response), dim=1)

    delta_position_id = torch.arange(1, pad_length + 1, device=position_ids.device, dtype=position_ids.dtype)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
    pad_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat((position_ids, pad_position_ids), dim=-1)
    return response, seq, position_ids


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto, n: int = 1) -> DataProto:
        if n < 1:
            raise ValueError(f"HFRollout requires n >= 1, got {n}")
        if n != 1:
            # Keep the same ordering contract as DataProto.repeat(interleave=True)
            # in the PPO trainer: prompt0_vote0, prompt0_vote1, prompt1_vote0, ...
            prompts = prompts.repeat(repeat_times=n, interleave=True)
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        pruning_enabled = _visual_pruning_enabled(self.module)
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        pruning_stage = _rollout_pruning_stage(prompts, do_sample=do_sample)
        keep_mmtok_cache_for_log_prob = pruning_stage == "rollout"
        success = False
        if pruning_enabled:
            _clear_mmtok_selection_cache(self.module, bump_scope=True)
        try:
            output = [self._generate_minibatch(p) for p in batch_prompts]
            output = DataProto.concat(output)
            success = True
            return output
        finally:
            if pruning_enabled and not (success and keep_mmtok_cache_for_log_prob):
                _clear_mmtok_selection_cache(self.module, bump_scope=True)

    def _multi_modal_kwargs(self, prompts: DataProto, device) -> dict:
        if "multi_modal_inputs" not in prompts.non_tensor_batch:
            return {}

        multi_modal_inputs = prompts.non_tensor_batch["multi_modal_inputs"]
        if len(multi_modal_inputs) == 0:
            return {}

        rows = list(multi_modal_inputs)
        kwargs = {}
        for key in rows[0].keys():
            if key in _GENERATE_INPUT_KEYS:
                continue

            values = []
            for row in rows:
                if key not in row:
                    raise KeyError(f"Missing multi_modal_inputs[{key!r}] in one rollout row")
                values.append(row[key])

            first = values[0]
            if torch.is_tensor(first):
                kwargs[key] = torch.cat([value.to(device) for value in values], dim=0)
            else:
                kwargs[key] = _to_device(values, device)

        return kwargs

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        if _visual_pruning_enabled(self.module):
            if self.config.get("pruned_use_cache", False):
                return self._generate_minibatch_pruned_cached(prompts)
            return self._generate_minibatch_pruned(prompts)
        if self.config.get("manual_no_cache", False):
            return self._generate_minibatch_manual_no_cache(prompts)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        multi_modal_kwargs = self._multi_modal_kwargs(prompts, idx.device)

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get("temperature", self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generate_kwargs = {
                    "input_ids": idx,
                    "attention_mask": attention_mask,
                    "do_sample": do_sample,
                    "max_new_tokens": response_length,
                    # max_length=max_length,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": pad_token_id,
                    "generation_config": generation_config,
                    # renormalize_logits=True,
                    "output_scores": False,  # this is potentially very large
                    "return_dict_in_generate": True,
                    "use_cache": True,
                    **multi_modal_kwargs,
                }
                output = self.module.generate(**generate_kwargs)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences

        # Standard HF causal generation returns prompt+response token ids.
        # InternVL's remote generate path uses inputs_embeds for image features
        # and returns response-only token ids, so reconstruct the shared VERL
        # prompt+response layout explicitly in that case.
        sequence_length = prompt_length + self.config.response_length
        if seq.shape[1] <= self.config.response_length:
            response = seq
            delta_length = self.config.response_length - response.shape[1]
            if delta_length > 0:
                delta_tokens = torch.full(
                    size=(batch_size, delta_length),
                    fill_value=pad_token_id,
                    device=response.device,
                    dtype=response.dtype,
                )
                response = torch.cat((response, delta_tokens), dim=1)
            prompt = idx
            seq = torch.cat((prompt, response), dim=1)
        else:
            # huggingface generate will stop generating when all the batch reaches [EOS].
            # We have to pad to response_length
            delta_length = sequence_length - seq.shape[1]
            if delta_length > 0:
                delta_tokens = torch.full(
                    size=(batch_size, delta_length),
                    fill_value=pad_token_id,
                    device=seq.device,
                    dtype=seq.dtype,
                )
                seq = torch.cat((seq, delta_tokens), dim=1)
            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

        assert seq.shape[1] == sequence_length

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)

    @torch.no_grad()
    def _generate_minibatch_pruned_cached(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        multi_modal_kwargs = self._multi_modal_kwargs(prompts, idx.device)

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        if batch_size != 1:
            raise RuntimeError("HF cached pruned rollout requires rollout.micro_batch_size=1")
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        pruning_stage = _rollout_pruning_stage(prompts, do_sample=do_sample)

        running_ids = idx
        running_position_ids = position_ids
        responses = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=idx.device)
        extra_info = prompts.non_tensor_batch.get("extra_info") if "extra_info" in prompts.non_tensor_batch else None
        past_key_values = None
        cached_attention_mask = None
        cache_seq_len = None

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for step_idx in range(response_length):
                    if step_idx == 0:
                        model_position_ids = _position_ids_for_model(running_position_ids, batch_size=batch_size)
                        with _visual_pruning_context(self.module, stage=pruning_stage, extra_info=extra_info):
                            output = self.module(
                                input_ids=running_ids,
                                attention_mask=attention_mask,
                                position_ids=model_position_ids,
                                use_cache=True,
                                **multi_modal_kwargs,
                            )
                        past_key_values = output.past_key_values
                        pruned_info = _visual_pruning_last_forward(self.module) or {}
                        cached_attention_mask = pruned_info.get("attention_mask")
                        if cached_attention_mask is None:
                            cache_seq_len = _past_seq_length(past_key_values)
                            cached_attention_mask = torch.ones(
                                (batch_size, cache_seq_len),
                                device=attention_mask.device,
                                dtype=attention_mask.dtype,
                            )
                        else:
                            cached_attention_mask = cached_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                            cache_seq_len = int(cached_attention_mask.shape[-1])
                    else:
                        if past_key_values is None or cached_attention_mask is None or cache_seq_len is None:
                            raise RuntimeError("cached pruned rollout lost prefill cache state")
                        current_token = responses[-1][:, None]
                        current_attention = torch.ones(
                            (batch_size, 1),
                            device=cached_attention_mask.device,
                            dtype=cached_attention_mask.dtype,
                        )
                        cached_attention_mask = torch.cat([cached_attention_mask, current_attention], dim=-1)
                        cache_position = torch.arange(
                            cache_seq_len,
                            cache_seq_len + 1,
                            device=idx.device,
                            dtype=torch.long,
                        )
                        model_position_ids = _position_ids_for_model(running_position_ids[..., -1:], batch_size=batch_size)
                        with _visual_pruning_context(self.module, stage=pruning_stage, extra_info=extra_info):
                            output = self.module(
                                input_ids=current_token,
                                attention_mask=cached_attention_mask,
                                position_ids=model_position_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                cache_position=cache_position,
                            )
                        past_key_values = output.past_key_values
                        cache_seq_len = _past_seq_length(past_key_values)

                    next_token = _sample_next_token(
                        output.logits[:, -1, :],
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    next_token = torch.where(unfinished, next_token, torch.full_like(next_token, pad_token_id))
                    responses.append(next_token)
                    unfinished = unfinished & (next_token != eos_token_id)

                    running_ids = torch.cat([running_ids, next_token[:, None]], dim=-1)
                    next_position_ids = running_position_ids[..., -1:] + 1
                    running_position_ids = torch.cat([running_position_ids, next_position_ids], dim=-1)
                    if not unfinished.any():
                        break

        response = torch.stack(responses, dim=1) if responses else torch.empty((batch_size, 0), dtype=idx.dtype, device=idx.device)
        response, seq, running_position_ids = _pad_generated_response(
            idx=idx,
            response=response,
            position_ids=running_position_ids,
            target_response_length=response_length,
            pad_token_id=pad_token_id,
        )
        prompt = idx

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "position_ids": running_position_ids,
            },
            batch_size=batch_size,
        )

        torch.cuda.empty_cache()
        self.module.train()
        return DataProto(batch=batch)

    @torch.no_grad()
    def _generate_minibatch_manual_no_cache(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        multi_modal_kwargs = self._multi_modal_kwargs(prompts, idx.device)

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        if batch_size != 1:
            raise RuntimeError("HF manual no-cache rollout requires rollout.micro_batch_size=1")
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        pruning_stage = _rollout_pruning_stage(prompts, do_sample=do_sample)

        running_ids = idx
        running_attention_mask = attention_mask
        running_position_ids = position_ids
        responses = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(response_length):
                    model_position_ids = _position_ids_for_model(running_position_ids, batch_size=batch_size)
                    output = self.module(
                        input_ids=running_ids,
                        attention_mask=running_attention_mask,
                        position_ids=model_position_ids,
                        use_cache=False,
                        **multi_modal_kwargs,
                    )
                    next_token = _sample_next_token(
                        output.logits[:, -1, :],
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    next_token = torch.where(unfinished, next_token, torch.full_like(next_token, pad_token_id))
                    responses.append(next_token)
                    unfinished = unfinished & (next_token != eos_token_id)

                    running_ids = torch.cat([running_ids, next_token[:, None]], dim=-1)
                    next_attention = torch.ones_like(next_token, dtype=running_attention_mask.dtype)
                    running_attention_mask = torch.cat([running_attention_mask, next_attention[:, None]], dim=-1)
                    next_position_ids = running_position_ids[..., -1:] + 1
                    running_position_ids = torch.cat([running_position_ids, next_position_ids], dim=-1)
                    if not unfinished.any():
                        break

        response = torch.stack(responses, dim=1) if responses else torch.empty((batch_size, 0), dtype=idx.dtype, device=idx.device)
        response, seq, running_position_ids = _pad_generated_response(
            idx=idx,
            response=response,
            position_ids=running_position_ids,
            target_response_length=response_length,
            pad_token_id=pad_token_id,
        )
        prompt = idx

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "position_ids": running_position_ids,
            },
            batch_size=batch_size,
        )

        torch.cuda.empty_cache()
        self.module.train()
        return DataProto(batch=batch)

    @torch.no_grad()
    def _generate_minibatch_pruned(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        multi_modal_kwargs = self._multi_modal_kwargs(prompts, idx.device)

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        if batch_size != 1:
            raise RuntimeError("HF pruned rollout requires rollout.micro_batch_size=1")
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        pruning_stage = _rollout_pruning_stage(prompts, do_sample=do_sample)

        running_ids = idx
        running_attention_mask = attention_mask
        running_position_ids = position_ids
        responses = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=idx.device)
        extra_info = prompts.non_tensor_batch.get("extra_info") if "extra_info" in prompts.non_tensor_batch else None

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(response_length):
                    model_position_ids = _position_ids_for_model(running_position_ids, batch_size=batch_size)
                    with _visual_pruning_context(self.module, stage=pruning_stage, extra_info=extra_info):
                        output = self.module(
                            input_ids=running_ids,
                            attention_mask=running_attention_mask,
                            position_ids=model_position_ids,
                            use_cache=False,
                            **multi_modal_kwargs,
                        )
                    next_token = _sample_next_token(
                        output.logits[:, -1, :],
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    next_token = torch.where(unfinished, next_token, torch.full_like(next_token, pad_token_id))
                    responses.append(next_token)
                    unfinished = unfinished & (next_token != eos_token_id)

                    running_ids = torch.cat([running_ids, next_token[:, None]], dim=-1)
                    next_attention = torch.where(
                        unfinished,
                        torch.ones_like(next_token, dtype=running_attention_mask.dtype),
                        torch.ones_like(next_token, dtype=running_attention_mask.dtype),
                    )
                    running_attention_mask = torch.cat([running_attention_mask, next_attention[:, None]], dim=-1)
                    next_position_ids = running_position_ids[..., -1:] + 1
                    running_position_ids = torch.cat([running_position_ids, next_position_ids], dim=-1)
                    if not unfinished.any():
                        break

        response = torch.stack(responses, dim=1) if responses else torch.empty((batch_size, 0), dtype=idx.dtype, device=idx.device)
        response, seq, running_position_ids = _pad_generated_response(
            idx=idx,
            response=response,
            position_ids=running_position_ids,
            target_response_length=response_length,
            pad_token_id=pad_token_id,
        )
        prompt = idx

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": final_attention_mask,
                "position_ids": running_position_ids,
            },
            batch_size=batch_size,
        )

        torch.cuda.empty_cache()
        self.module.train()
        return DataProto(batch=batch)
