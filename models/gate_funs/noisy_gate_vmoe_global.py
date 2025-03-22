import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from fmoe.gates.base_gate import BaseGate
from collections import Counter
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE

###############################################################################
# Modified NoisyGlobalGate_VMoE that extends the above noisy gate VMoE object
###############################################################################

class NoisyGlobalGate_VMoE(NoisyGate_VMoE):
    """
    A modified noisy gate that extends a given noise_gate_vmoe by:
      - reinitializing the routing parameters with one extra output (the new global expert at index 0)
      - ensuring that during the top-k selection, the global expert is always active and is not considered in the top-k computation.
    """
    def __init__(self, noise_gate_vmoe: NoisyGate_VMoE):
        # Retrieve parameters from the provided noise_gate_vmoe object.
        d_model = noise_gate_vmoe.w_gate.shape[0]
        num_expert = noise_gate_vmoe.num_expert  # local experts count remains the same
        world_size = noise_gate_vmoe.world_size
        top_k = noise_gate_vmoe.top_k
        noise_std = noise_gate_vmoe.noise_std
        no_noise = noise_gate_vmoe.no_noise
        return_decoupled_activation = noise_gate_vmoe.return_decoupled_activation
        regu_experts_fromtask = noise_gate_vmoe.regu_experts_fromtask
        num_experts_pertask = noise_gate_vmoe.num_experts_pertask
        num_tasks = noise_gate_vmoe.num_tasks
        regu_sem = noise_gate_vmoe.regu_sem
        sem_force = getattr(noise_gate_vmoe, "sem_force", False)
        regu_subimage = noise_gate_vmoe.regu_subimage

        # Initialize parent with the same parameters.
        super().__init__(d_model, num_expert, world_size, top_k, noise_std, no_noise,
                         return_decoupled_activation, regu_experts_fromtask,
                         num_experts_pertask, num_tasks, regu_sem, sem_force, regu_subimage)
        # Update tot_expert to account for the extra global expert.
        original_tot_expert = self.tot_expert  # originally computed in BaseGate
        self.tot_expert = original_tot_expert + 1  # add one extra output

        # Reinitialize the gate weight matrix to have an extra column.
        self.w_gate = nn.Parameter(torch.zeros(d_model, self.tot_expert), requires_grad=True)
        self.reset_parameters()  # reinitialize (this applies Kaiming initialization)

        if self.return_decoupled_activation:
            self.w_gate_aux = nn.Parameter(torch.zeros(d_model, self.tot_expert), requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.w_gate_aux, a=math.sqrt(5))

    def forward(self, inp, task_id=None, sem=None):
        """
        Forward pass:
          1. Compute the clean logits using the new routing matrix.
          2. Add noise as in the base implementation.
          3. Apply softmax.
          4. Separate the global expert (index 0) from the rest.
          5. Compute top_k selection over the local experts only.
          6. Always include the global expert in the final output.
          7. Compute auxiliary losses (if in training mode).
        """
        shape_input = list(inp.shape)
        channel = shape_input[-1]
        other_dim = shape_input[:-1]
        inp = inp.reshape(-1, channel)

        # --- Compute clean logits (the same branching as the parent) ---
        if self.regu_experts_fromtask and (task_id is not None):
            # Note: for regu_experts_fromtask, one must ensure that the global expert is handled properly.
            clean_logits = inp @ self.w_gate[:, self.start_experts_id[task_id]:
                                                  self.start_experts_id[task_id] + self.num_experts_pertask]
            raw_noise_stddev = self.noise_std / (self.num_experts_pertask)
        else:
            # device debugging
            clean_logits = inp @ self.w_gate
            raw_noise_stddev = self.noise_std / self.tot_expert

        noise_stddev = raw_noise_stddev * self.training

        if self.regu_sem and (sem is not None):
            batch = sem.shape[0]
            # Exclude the global expert (first column) from semantic regularization.
            prior_selection = clean_logits.reshape(batch, -1, self.num_expert)[:, 1:, :]
            prior_selection = prior_selection.reshape(-1, self.num_expert)
            prior_out = self.head(prior_selection)
            prior_out = prior_out.reshape(batch, sem.shape[2], sem.shape[3], self.num_class)
            prior_out = prior_out.permute(0, 3, 1, 2)
            semregu_loss = self.criterion(prior_out, sem)
            self.semregu_loss = semregu_loss

        if self.regu_subimage and (sem is not None):
            self.regu_subimage_loss = 0
            batch_size = sem.shape[0]
            prior_selection = clean_logits.reshape(batch_size, -1, self.num_expert)[:, 1:, :]
            prior_selection = prior_selection.reshape(batch_size, 30, 40, self.num_expert)
            for k in range(batch_size):
                for i in range(int(30 / self.subimage_tokens)):
                    for j in range(int(40 / self.subimage_tokens)):
                        subimage_selection = prior_selection[k,
                                                             self.subimage_tokens * i:self.subimage_tokens * (i + 1),
                                                             self.subimage_tokens * j:self.subimage_tokens * (j + 1),
                                                             :]
                        subimage_selection = subimage_selection.reshape(-1, self.num_expert)
                        top_subimage_values, top_subimage_index = torch.topk(torch.sum(subimage_selection, dim=0), 2)
                        gt_logit = torch.zeros(self.num_expert, device=clean_logits.device)
                        gt_logit[top_subimage_index[0]] = top_subimage_values[0]
                        gt_logit[top_subimage_index[1]] = top_subimage_values[1]
                        gt_logit = gt_logit.repeat(subimage_selection.shape[0], 1)
                        kl1 = F.kl_div(subimage_selection.softmax(dim=-1).log(),
                                       gt_logit.softmax(dim=-1), reduction='batchmean')
                        self.regu_subimage_loss += kl1
            self.regu_subimage_loss = self.regu_subimage_loss / (batch_size * 30 * 40 /
                                                                 self.subimage_tokens / self.subimage_tokens)

        if self.no_noise:
            noise_stddev *= 0

        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)

        if self.select_idx is not None:
            assert len(self.select_idx) >= self.top_k
            noisy_logits = noisy_logits[:, self.select_idx]

        if self.return_decoupled_activation:
            clean_logits_aux = inp @ self.w_gate_aux
            raw_noise_stddev_aux = self.noise_std / self.tot_expert
            noise_stddev_aux = (torch.randn_like(clean_logits) * raw_noise_stddev_aux) * self.training
            if self.no_noise:
                noise_stddev_aux *= 0
            noisy_logits_aux = clean_logits_aux + (torch.randn_like(clean_logits_aux) * noise_stddev_aux)

        # --- Separate the global expert from the rest ---
        logits = self.softmax(noisy_logits)
        # Global expert is the extra output at index 0.
        global_logits = logits[:, :1]  # always active
        other_logits = logits[:, 1:]   # local experts

        # --- Top-k selection: choose from local experts only ---
        top_local_logits, top_local_indices = other_logits.topk(min(self.top_k, self.tot_expert - 1), dim=1)
        # Adjust indices to account for the offset (since index 0 is global)
        top_local_indices = top_local_indices + 1

        # --- Combine the always-on global expert with the selected local experts ---
        final_indices = torch.cat([torch.zeros((logits.size(0), 1), device=logits.device, dtype=torch.long),
                                   top_local_indices], dim=1)
        final_gates = torch.cat([global_logits, top_local_logits], dim=1)

        # --- (Optional) Loss computation based on local experts only ---
        zeros = torch.zeros_like(other_logits, requires_grad=True)
        local_gates = zeros.scatter(1, top_local_indices - 1, top_local_logits)
        if self.training:
            if self.top_k < (self.tot_expert - 1) and (not self.no_noise) and abs(noise_stddev) > 1e-6:
                load = self._prob_in_top_k(clean_logits[:, 1:], noisy_logits[:, 1:],
                                           noise_stddev,
                                           other_logits.topk(self.top_k + 1, dim=1)[0]).sum(0)
            else:
                load = self._gates_to_load(local_gates)
            importance = local_gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
        else:
            loss = 0

        self.set_loss(loss)
        self.activation = logits.reshape(other_dim + [-1, ]).contiguous()
        if self.return_decoupled_activation:
            self.activation = noisy_logits_aux.reshape(other_dim + [-1, ]).contiguous()
        final_indices = final_indices.reshape(other_dim + [self.top_k + 1]).contiguous()
        final_gates = final_gates.reshape(other_dim + [self.top_k + 1]).contiguous()

        return final_indices, final_gates
