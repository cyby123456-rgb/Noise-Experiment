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
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.model import sample_quantile_fractions
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic
from verl.utils.risk_functional import compute_rho_from_dist

__all__ = ["DataParallelPPOCritic"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.is_distributional = getattr(self.config, "distributional", False)
        self.is_distributional_v2 = getattr(self.config, "distributional_v2", False)
        self.is_distributional_v3 = getattr(self.config, "distributional_v3", False)
        self.num_quantiles = getattr(self.config, "num_quantiles", 1)
        self.quantile_huber_kappa = getattr(self.config, "quantile_huber_kappa", 1.0)
        self.quantile_mode = getattr(self.config, "quantile_mode", "iqn")
        self.num_value_heads = int(getattr(self.config, "num_value_heads", 1))
        self.use_multi_head_iqn = self.is_distributional and self.quantile_mode not in ["fixed", "c51"] and self.num_value_heads > 1
        if self.is_distributional:
            if self.quantile_mode == "fixed" and not hasattr(self.critic_module, "qr_head"):
                raise AttributeError("Quantile mode fixed requires critic_module.qr_head for QR-DQN.")
            if self.quantile_mode == "c51" and not hasattr(self.critic_module, "c51_head"):
                raise AttributeError("Quantile mode c51 requires critic_module.c51_head for categorical C51.")
            if self.quantile_mode not in ["fixed", "c51"] and not hasattr(self.critic_module, "iqn_head"):
                raise AttributeError("IQN quantile mode requires critic_module.iqn_head for IQN.")
        self.use_action_response_mask = getattr(self.config, "use_action_response_mask", False)
        self.c51_v_min = getattr(self.config, "c51_v_min", -10.0)
        self.c51_v_max = getattr(self.config, "c51_v_max", 10.0)
        self.sr_lambda = getattr(self.config, "sr_lambda", 0.9)
        self.sr_num_samples = getattr(self.config, "sr_num_samples", 8)
        print(
            f"Critic use_remove_padding={self.use_remove_padding}, "
            f"distributional={self.is_distributional}, distributional_v2={self.is_distributional_v2}, "
            f"distributional_v3={self.is_distributional_v3}, num_value_heads={self.num_value_heads}"
        )
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _build_sr_targets_quantiles(self, old_q, rewards, response_mask):
        if not (0.0 <= self.sr_lambda <= 1.0):
            raise ValueError(f"sr_lambda must be in [0, 1], got {self.sr_lambda}")
        bs, t, k_old = old_q.shape
        m = self.sr_num_samples
        old_q_next = torch.zeros_like(old_q)
        old_q_next[:, :-1, :] = old_q[:, 1:, :]
        next_mask = torch.zeros_like(response_mask)
        next_mask[:, :-1] = response_mask[:, 1:]

        z = torch.zeros(bs, m, device=old_q.device, dtype=old_q.dtype)
        targets = torch.zeros(bs, t, m, device=old_q.device, dtype=old_q.dtype)
        for idx in range(t - 1, -1, -1):
            valid = response_mask[:, idx].to(torch.bool)
            if not valid.any():
                continue
            sample_idx = torch.randint(0, k_old, (bs, m), device=old_q.device)
            fresh = old_q_next[:, idx, :].gather(1, sample_idx)
            replace = torch.rand(bs, m, device=old_q.device) < (1.0 - self.sr_lambda)
            z_new = torch.where(replace, fresh, z)
            z_new = rewards[:, idx].unsqueeze(-1) + self.config.gamma * next_mask[:, idx].unsqueeze(-1) * z_new
            z = torch.where(valid.unsqueeze(-1), z_new, z)
            targets[:, idx, :] = torch.where(valid.unsqueeze(-1), z_new, targets[:, idx, :])
        return targets

    def _project_c51_samples(self, z_samples, atoms):
        bs, t, m = z_samples.shape
        n_atoms = atoms.numel()
        v_min = atoms[0]
        v_max = atoms[-1]
        delta_z = atoms[1] - atoms[0]

        values = z_samples.clamp(min=v_min.item(), max=v_max.item())
        b = (values - v_min) / delta_z
        l = b.floor().long()
        u = (l + 1).clamp(min=0, max=n_atoms - 1)
        l = l.clamp(min=0, max=n_atoms - 1)
        offset = (b - l.to(b.dtype)).clamp(min=0.0, max=1.0)

        target_probs = torch.zeros(bs, t, n_atoms, device=z_samples.device, dtype=z_samples.dtype)
        target_probs_flat = target_probs.view(-1, n_atoms)
        l_flat = l.view(-1, m)
        u_flat = u.view(-1, m)
        weight_l = (1.0 - offset).view(-1, m) / m
        weight_u = offset.view(-1, m) / m

        target_probs_flat.scatter_add_(1, l_flat, weight_l.to(target_probs_flat.dtype))
        target_probs_flat.scatter_add_(1, u_flat, weight_u.to(target_probs_flat.dtype))
        return target_probs

    def _build_sr_targets_c51(self, old_logits, rewards, response_mask, atoms):
        if not (0.0 <= self.sr_lambda <= 1.0):
            raise ValueError(f"sr_lambda must be in [0, 1], got {self.sr_lambda}")
        bs, t, n_atoms = old_logits.shape
        m = self.sr_num_samples
        next_logits = torch.zeros_like(old_logits)
        next_logits[:, :-1, :] = old_logits[:, 1:, :]
        next_mask = torch.zeros_like(response_mask)
        next_mask[:, :-1] = response_mask[:, 1:]
        next_logits = next_logits.masked_fill(~next_mask.bool().unsqueeze(-1), 0.0)
        next_probs = torch.softmax(next_logits.float(), dim=-1).detach()

        z = torch.zeros(bs, m, device=old_logits.device, dtype=old_logits.dtype)
        z_samples = torch.zeros(bs, t, m, device=old_logits.device, dtype=old_logits.dtype)
        for idx in range(t - 1, -1, -1):
            valid = response_mask[:, idx].to(torch.bool)
            if not valid.any():
                continue
            has_next = next_mask[:, idx].to(torch.bool)
            atom_idx = torch.multinomial(next_probs[:, idx, :], m, replacement=True)
            fresh = atoms[atom_idx]
            fresh = torch.where(has_next.unsqueeze(-1), fresh, torch.zeros_like(fresh))
            replace = torch.rand(bs, m, device=old_logits.device) < (1.0 - self.sr_lambda)
            z_new = torch.where(replace, fresh, z)
            z_new = rewards[:, idx].unsqueeze(-1) + self.config.gamma * next_mask[:, idx].unsqueeze(-1) * z_new
            z = torch.where(valid.unsqueeze(-1), z_new, z)
            z_samples[:, idx, :] = torch.where(valid.unsqueeze(-1), z_new, z_samples[:, idx, :])

        target_probs = self._project_c51_samples(z_samples, atoms)
        return target_probs

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        values = None
        hidden_states = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=self.is_distributional,
                    return_dict=True,
                )  # prevent model thinks we are generating
                
                logits_rmpad = output.logits
                hidden_rmpad = output.hidden_states[-1] if self.is_distributional else None
                if not self.is_distributional:
                    values_rmpad = logits_rmpad.squeeze(0)  # (total_nnz)
                else:
                    # (1, total_nnz, K) -> (total_nnz, K)
                    hidden_rmpad = hidden_rmpad.squeeze(0)  # (total_nnz, hidden)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    if not self.is_distributional:
                        values_rmpad = gather_outpus_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if self.is_distributional:
                        hidden_rmpad = gather_outpus_and_unpad(hidden_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                #values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                if not self.is_distributional:
                    values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                    values = values[:, -response_length - 1 : -1]
                else:
                    hidden_states = pad_input(hidden_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                    hidden_states = hidden_states[:, -response_length - 1 : -1, :]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=self.is_distributional,
                    return_dict=True,
                )  # prevent model thinks we are generating
                #values = values[:, -response_length - 1 : -1].squeeze(-1)
                
                if not self.is_distributional:
                    values = output.logits
                    values = values[:, -response_length - 1 : -1]
                else:
                    hidden_states = output.hidden_states[-1][:, -response_length - 1 : -1, :]
            if not self.is_distributional:
                if values is None:
                    raise RuntimeError("Non-distributional critic did not produce values.")
                values = values.squeeze(-1)
                return values

            # distributional IQN path
            if hidden_states is None:
                raise RuntimeError("Distributional critic did not produce hidden_states.")
            bsz, seq_len, _ = hidden_states.shape
            if self.quantile_mode == "fixed" and hasattr(self.critic_module, "qr_head"):
                if isinstance(self.critic_module, FSDP):
                    with FSDP.summon_full_params(self.critic_module, recurse=False):
                        quantiles = self.critic_module.qr_head(hidden_states)  # 不需要 flatten
                else:
                    quantiles = self.critic_module.qr_head(hidden_states)
                #quantiles = self.critic_module.qr_head(hidden_states)
                return quantiles, None
            if self.quantile_mode == "c51" and hasattr(self.critic_module, "c51_head"):
                #head = self.critic_module.c51_head.value_head
                #w = head.weight
                #print("weight type:", type(w), "is_meta:", getattr(w, "is_meta", False), "numel:", w.numel(), "shape:", tuple(w.shape))
                #import pdb
                #pdb.set_trace()
                # 假设 hidden_states 的形状是 (26, 1024, 1536)
                batch_size, seq_len, hidden_size = hidden_states.shape
                # 调整形状为 (batch_size * seq_len, hidden_size)
                hidden_states_flat = hidden_states.reshape(-1, hidden_size)

                #c51_logits = self.critic_module.c51_head(hidden_states_flat)
                if isinstance(self.critic_module, FSDP):
                    with FSDP.summon_full_params(self.critic_module, recurse=False):
                        c51_logits = self.critic_module.c51_head(hidden_states)  # 不需要 flatten
                else:
                    c51_logits = self.critic_module.c51_head(hidden_states)
                # 如果需要将 logits 重新调整为 (batch_size, seq_len, num_atoms)
                #num_atoms = c51_logits.size(-1)
                #c51_logits = c51_logits.reshape(batch_size, seq_len, num_atoms)
                return c51_logits, None
            taus = sample_quantile_fractions(
                batch=bsz,
                seq_len=seq_len,
                num_quantiles=self.num_quantiles,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                mode=self.quantile_mode,
            )
            if isinstance(self.critic_module, FSDP):
                with FSDP.summon_full_params(self.critic_module, recurse=False):
                    quantiles = self.critic_module.iqn_head(hidden_states, taus)  # 不需要 flatten
            else:
                quantiles = self.critic_module.iqn_head(hidden_states, taus)
            #quantiles = self.critic_module.iqn_head(hidden_states, taus)
            return quantiles, taus

    def _compute_multi_head_quantile_loss(self, vpreds, target_quantiles, response_mask, taus):
        num_heads = vpreds.size(2)
        loss = vpreds.new_zeros(())
        for head_idx in range(num_heads):
            head_taus = taus[:, :, head_idx, :] if taus is not None and taus.dim() == 4 else taus
            loss = loss + core_algos.compute_IQN_quantile_value_loss(
                vpreds=vpreds[:, :, head_idx, :],
                target_quantiles=target_quantiles[:, :, head_idx, :],
                response_mask=response_mask,
                kappa=self.quantile_huber_kappa,
                taus=head_taus,
            )
        return loss / num_heads

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> DataProto:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=int(max_token_len*0.75))
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        taus_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)

        # Merge micro-batch outputs
        value_quantiles_heads = None
        if self.is_distributional:
            # tolerate any extra items while only keeping quantiles / logits
            quantiles_or_logits = []
            for item in values_lst:
                if isinstance(item, (tuple, list)):
                    # item is (quantiles, taus) for IQN / fixed
                    quantiles_or_logits.append(item[0])
                    if item[1] is not None:
                        taus_lst.append(item[1])
                else:
                    # item is quantiles (fixed grid) or logits (C51)
                    quantiles_or_logits.append(item)
            values = torch.concat(quantiles_or_logits, dim=0)
            if self.use_multi_head_iqn and values.dim() == 4:
                value_quantiles_heads = values
            if len(taus_lst) > 0:
                taus = torch.concat(taus_lst, dim=0)
            else:
                taus = None
        else:
            values = torch.concat(values_lst, dim=0)

        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        response_mask = (
            attention_mask[:, -response_length:]
            if self.use_action_response_mask
            else attention_mask[:, -response_length - 1 : -1]
        )
        #if self.is_distributional and (self.is_distributional_v2 or self.is_distributional_v3):
        #   response_mask = attention_mask[:, -response_length:]
        # Distributional critic: always return risk-neutral expectation E[Z]
        # Non-distributional critic: pass through scalar values.
        if self.is_distributional:
            if self.quantile_mode == "c51":
                # values are logits over atoms
                values = values * response_mask.unsqueeze(-1)
                probs = torch.softmax(values.float(), dim=-1)
                atoms = torch.linspace(
                    self.c51_v_min,
                    self.c51_v_max,
                    values.size(-1),
                    device=values.device,
                    dtype=torch.float32,
                )
                expect = (probs * atoms.view(1, 1, -1)).sum(dim=-1)
                values_mean = expect * response_mask
            else:
                if values.dim() == 4:
                    values = values * response_mask.unsqueeze(-1).unsqueeze(-1)
                    value_head_means = values.mean(dim=-1)
                    value_head_disagreement = value_head_means.var(dim=-1, unbiased=False) * response_mask
                    values = values.mean(dim=2)
                    values_mean = values.mean(dim=-1)
                else:
                    # IQN / fixed quantile: values are quantiles (B, T, K)
                    values = values * response_mask.unsqueeze(-1)
                    values_mean = values.mean(dim=-1)
        else:
            values = values * response_mask
            values_mean = values

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]
            values_mean = values_mean[revert_indices]
            if value_quantiles_heads is not None:
                value_quantiles_heads = value_quantiles_heads[revert_indices]
            if "value_head_means" in locals():
                value_head_means = value_head_means[revert_indices]
            if "value_head_disagreement" in locals():
                value_head_disagreement = value_head_disagreement[revert_indices]
            # keep taus aligned with values when using dynamic bsz
            if self.is_distributional and "taus" in locals() and taus is not None:
                taus = taus[revert_indices]

        # Package outputs for trainer:
        # - values: risk-neutral baseline E[Z]
        # - optional distribution fields for risk computation on actor side
        tensors = {"values": values_mean}
        if self.is_distributional:
            if self.quantile_mode == "c51":
                tensors["value_logits"] = values  # (B, T, K)
                # atoms are deterministic from config; create here for trainer usage
                atoms = torch.linspace(
                    self.c51_v_min,
                    self.c51_v_max,
                    values.size(-1),
                    device=values.device,
                    dtype=torch.float32,
                )
                #tensors["value_atoms"] = atoms
                atoms_batched = atoms.unsqueeze(0).expand(values.size(0), -1)  # (B, K) 匹配 batch 维
                tensors["value_atoms"] = atoms_batched
            else:
                tensors["value_quantiles"] = values  # (B, T, K) masked on response
                if taus is not None:
                    tensors["value_taus"] = taus
                if self.use_multi_head_iqn and value_quantiles_heads is not None and "value_head_means" in locals():
                    tensors["value_quantiles_heads"] = value_quantiles_heads * response_mask.unsqueeze(-1).unsqueeze(-1)
                    tensors["value_head_means"] = value_head_means
                    tensors["value_head_disagreement"] = value_head_disagreement

        return DataProto.from_dict(tensors=tensors)

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        critic_distributional_weight = 1.0
        data_meta_info = getattr(data, "meta_info", None)
        if isinstance(data_meta_info, dict):
            try:
                critic_distributional_weight = float(data_meta_info.get("critic_distributional_weight", 1.0))
            except (TypeError, ValueError):
                critic_distributional_weight = 1.0

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        if self.is_distributional and (self.is_distributional_v2 or self.is_distributional_v3):
            select_keys.append("token_level_rewards")
            if self.quantile_mode == "c51":
                select_keys.append("value_logits")
            else:
                select_keys.append("value_quantiles")
                if self.use_multi_head_iqn:
                    select_keys.append("value_quantiles_heads")
        if self.is_distributional and self.quantile_mode == "iqn":
            # pull optional target quantiles/taus if available
            for key in ["target_quantiles", "target_taus"]:
                if key in data.batch.keys():
                    select_keys.append(key)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=int(max_token_len*0.75))
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    values = data["values"]
                    returns = data["returns"]
                    response_length = responses.size(1)

                    response_mask = (
                        attention_mask[:, -response_length:]
                        if self.use_action_response_mask
                        else attention_mask[:, -response_length - 1 : -1]
                    )
                    #if self.is_distributional and (self.is_distributional_v2 or self.is_distributional_v3):
                    #    response_mask = attention_mask[:, -response_length:]

                    vpreds = self._forward_micro_batch(data)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    #vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    #    vpreds=vpreds,
                    #    values=values,
                    #    returns=returns,
                    #    response_mask=response_mask,
                    #    cliprange_value=self.config.cliprange_value,
                    #)
                    if self.is_distributional:
                        if self.quantile_mode == "c51":
                            vpreds, _ = vpreds  # vpreds is logits, taus is None
                            # Generate atoms for projection
                            n_atoms = vpreds.size(-1)
                            atoms = torch.linspace(
                                self.c51_v_min, self.c51_v_max, n_atoms,
                                device=vpreds.device, dtype=torch.float32
                            )
                            append_to_dict(
                                metrics,{
                                    "critic/atoms_min": atoms.min().item(),
                                    "critic/atoms_max": atoms.max().item(),
                                    "critic/atoms_var": atoms.var(unbiased=False).item(),
                                    "critic/atoms_std": atoms.std(unbiased=False).item(),
                                },
                            )
                            if self.is_distributional_v3:
                                token_level_rewards = data["token_level_rewards"]
                                assert token_level_rewards.shape[:2] == response_mask.shape, (
                                    "token_level_rewards and response_mask must align in (B, T)."
                                )
                                old_logits = data["value_logits"].detach()
                                with torch.no_grad():
                                    target_probs = self._build_sr_targets_c51(
                                        old_logits=old_logits,
                                        rewards=token_level_rewards,
                                        response_mask=response_mask,
                                        atoms=atoms,
                                    )
                                log_probs = torch.log_softmax(vpreds, dim=-1)
                                loss_per_token = -torch.sum(target_probs * log_probs, dim=-1)
                                vf_loss = masked_mean(loss_per_token, response_mask)
                            elif self.is_distributional_v2:
                                token_level_rewards = data["token_level_rewards"]
                                assert token_level_rewards.shape[:2] == response_mask.shape, (
                                    "token_level_rewards and response_mask must align in (B, T)."
                                )
                                old_logits = data["value_logits"].detach()
                                next_logits = torch.zeros_like(old_logits)
                                next_logits[:, :-1, :] = old_logits[:, 1:, :]
                                next_mask = torch.zeros_like(response_mask)
                                next_mask[:, :-1] = response_mask[:, 1:]
                                vf_loss = core_algos.compute_categorical_value_loss_v2(
                                    logits=vpreds,
                                    next_logits=next_logits,
                                    rewards=token_level_rewards,
                                    response_mask=response_mask,
                                    atoms=atoms,
                                    gamma=self.config.gamma,
                                    next_mask=next_mask,
                                )
                            else:
                                vf_loss = core_algos.compute_categorical_value_loss(
                                    logits=vpreds,
                                    returns=returns,
                                    response_mask=response_mask,
                                    atoms=atoms,
                                )
                            # For logging/metrics, compute expected value
                            probs = torch.softmax(vpreds.float(), dim=-1)
                            expect = (probs * atoms.view(1, 1, -1)).sum(dim=-1)
                            scalar_vf_loss, scalar_vf_clipfrac = core_algos.compute_value_loss(
                                vpreds=expect,
                                values=values,
                                returns=returns,
                                response_mask=response_mask,
                                cliprange_value=self.config.cliprange_value,
                            )
                            vf_loss = (
                                critic_distributional_weight * vf_loss
                                + (1.0 - critic_distributional_weight) * scalar_vf_loss
                            )
                            vpred_mean = masked_mean(expect, response_mask).detach().item()
                            #vf_clipfrac = torch.tensor(0.0, device=vpreds.device)
                            vf_clipfrac = scalar_vf_clipfrac
                        else:
                            #iqn or qrdqn
                            vpreds, taus = vpreds
                            if self.use_multi_head_iqn:
                                quantile_mask = response_mask.unsqueeze(-1).unsqueeze(-1).bool()
                            else:
                                quantile_mask = response_mask.unsqueeze(-1).bool()
                            flat_quantiles = torch.masked_select(vpreds.detach(), quantile_mask)
                            if flat_quantiles.numel() > 0:
                                append_to_dict(
                                    metrics,
                                    {
                                        "critic/quantile_min": flat_quantiles.min().item(),
                                        "critic/quantile_max": flat_quantiles.max().item(),
                                        "critic/quantile_var": flat_quantiles.var(unbiased=False).item(),
                                        "critic/quantile_std": flat_quantiles.std(unbiased=False).item(),
                                    },
                                )
                            if self.use_multi_head_iqn:
                                head_means = vpreds.mean(dim=-1)
                                head_disagreement = masked_mean(head_means.var(dim=-1, unbiased=False), response_mask).detach().item()
                                append_to_dict(metrics, {"critic/head_disagreement": head_disagreement})
                            if self.is_distributional_v3:
                                token_level_rewards = data["token_level_rewards"]
                                assert token_level_rewards.shape[:2] == response_mask.shape, (
                                    "token_level_rewards and response_mask must align in (B, T)."
                                )
                                old_q = data["value_quantiles_heads"].detach() if self.use_multi_head_iqn else data["value_quantiles"].detach()
                                with torch.no_grad():
                                    if self.use_multi_head_iqn:
                                        target_q = torch.stack(
                                            [
                                                self._build_sr_targets_quantiles(
                                                    old_q=old_q[:, :, head_idx, :],
                                                    rewards=token_level_rewards,
                                                    response_mask=response_mask,
                                                )
                                                for head_idx in range(old_q.size(2))
                                            ],
                                            dim=2,
                                        )
                                    else:
                                        target_q = self._build_sr_targets_quantiles(
                                            old_q=old_q,
                                            rewards=token_level_rewards,
                                            response_mask=response_mask,
                                        )
                                if taus is None:
                                    k = vpreds.size(-1)
                                    taus = (torch.arange(k, device=vpreds.device, dtype=vpreds.dtype) + 0.5) / k
                                    if self.use_multi_head_iqn:
                                        taus = taus.view(1, 1, 1, k).expand_as(vpreds)
                                    else:
                                        taus = taus.view(1, 1, k).expand_as(vpreds)
                                if self.use_multi_head_iqn:
                                    vf_loss = self._compute_multi_head_quantile_loss(
                                        vpreds=vpreds,
                                        target_quantiles=target_q,
                                        response_mask=response_mask,
                                        taus=taus,
                                    )
                                else:
                                    vf_loss = core_algos.compute_IQN_quantile_value_loss(
                                        vpreds=vpreds,  # (bs, T, K)
                                        target_quantiles=target_q,
                                        response_mask=response_mask,
                                        kappa=self.quantile_huber_kappa,
                                        taus=taus,
                                    )
                            elif self.is_distributional_v2:
                                token_level_rewards = data["token_level_rewards"]
                                assert token_level_rewards.shape[:2] == response_mask.shape, (
                                    "token_level_rewards and response_mask must align in (B, T)."
                                )
                                old_q = data["value_quantiles_heads"].detach() if self.use_multi_head_iqn else data["value_quantiles"].detach()
                                next_q = torch.zeros_like(old_q)
                                next_q[:, :-1, ...] = old_q[:, 1:, ...]
                                next_mask = torch.zeros_like(response_mask)
                                next_mask[:, :-1] = response_mask[:, 1:]
                                with torch.no_grad():
                                    target_q = token_level_rewards.unsqueeze(-1) + (
                                        self.config.gamma * next_mask.unsqueeze(-1) * next_q
                                    )
                                if taus is None:
                                    k = vpreds.size(-1)
                                    taus = (torch.arange(k, device=vpreds.device, dtype=vpreds.dtype) + 0.5) / k
                                    if self.use_multi_head_iqn:
                                        taus = taus.view(1, 1, 1, k).expand_as(vpreds)
                                    else:
                                        taus = taus.view(1, 1, k).expand_as(vpreds)
                                if self.use_multi_head_iqn:
                                    target_q = target_q.unsqueeze(2).expand_as(vpreds) if target_q.dim() == 3 else target_q
                                    vf_loss = self._compute_multi_head_quantile_loss(
                                        vpreds=vpreds,
                                        target_quantiles=target_q,
                                        response_mask=response_mask,
                                        taus=taus,
                                    )
                                else:
                                    vf_loss = core_algos.compute_IQN_quantile_value_loss(
                                        vpreds=vpreds,  # (bs, T, K)
                                        target_quantiles=target_q,
                                        response_mask=response_mask,
                                        kappa=self.quantile_huber_kappa,
                                        taus=taus,
                                    )
                            else:
                                if self.use_multi_head_iqn:
                                    if taus is None:
                                        k = vpreds.size(-1)
                                        taus = (torch.arange(k, device=vpreds.device, dtype=vpreds.dtype) + 0.5) / k
                                        taus = taus.view(1, 1, 1, k).expand_as(vpreds)
                                    target_q = returns.unsqueeze(-1).unsqueeze(-1).expand_as(vpreds)
                                    vf_loss = self._compute_multi_head_quantile_loss(
                                        vpreds=vpreds,
                                        target_quantiles=target_q,
                                        response_mask=response_mask,
                                        taus=taus,
                                    )
                                else:
                                    vf_loss = core_algos.compute_quantile_value_loss(
                                        vpreds=vpreds,  # (bs, T, K)
                                        returns=returns,  # scalar targets
                                        response_mask=response_mask,
                                        num_quantiles=self.num_quantiles,
                                        tau_mode=self.quantile_mode,  # iqn or fixed
                                        kappa=self.quantile_huber_kappa,
                                        taus=taus,
                                    )
                            vf_clipfrac = torch.tensor(0.0, device=vpreds.device)  # not used for dist. loss
                            vpred_mean = masked_mean(vpreds.mean(dim=-1).mean(dim=-1) if self.use_multi_head_iqn else vpreds.mean(dim=-1), response_mask).detach().item()
                            scalar_vpreds = vpreds.mean(dim=-1).mean(dim=-1) if self.use_multi_head_iqn else vpreds.mean(dim=-1)
                            scalar_vf_loss, scalar_vf_clipfrac = core_algos.compute_value_loss(
                                vpreds=scalar_vpreds,
                                values=values,
                                returns=returns,
                                response_mask=response_mask,
                                cliprange_value=self.config.cliprange_value,
                            )
                            vf_loss = (
                                critic_distributional_weight * vf_loss
                                + (1.0 - critic_distributional_weight) * scalar_vf_loss
                            )
                            vf_clipfrac = scalar_vf_clipfrac
                            vpred_mean = masked_mean(scalar_vpreds, response_mask).detach().item()
                    else:
                        vpreds = vpreds.squeeze(-1)  # (bs, T)
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                        )
                        vpred_mean = masked_mean(vpreds, response_mask).detach().item()  # mean value

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        # loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                        micro_bsz = data["attention_mask"].shape[0]
                        loss = vf_loss * (micro_bsz / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        #"critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        "critic/vpred_mean": vpred_mean,
                        "critic/distributional_weight": critic_distributional_weight,
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
