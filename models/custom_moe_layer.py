r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from fmoe.layers import FMoE, _fmoe_general_global_forward
from fmoe.linear import FMoELinear
from functools import partial
import tree
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NaiveGate
from fmoe.linear_proj import FMoELinearProj

from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.gate_funs.noisy_gate_vmoe_global import NoisyGlobalGate_VMoE

from utils.perpca import PerPCA
import wandb

from pdb import set_trace
import numpy as np

from utils.global_components import get_num_global_components

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0, top_k=4):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation
        self.outputs = None
        self.record_output = False
        self.num_experts = num_expert
        self.stage = 0 # set this to 1 once components are calculated
        self.outputs_size_limit = 1000
        self.loss = 0
        self.top_k = top_k

    def reset_loss(self):
        self.loss = 0

    def reset_outputs(self):
        self.outputs = None

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        # make sure everything is on cuda
        if inp.device != 'cuda':
            inp = inp.to('cuda')

        inp_flat = inp.view(-1, inp.size(-1))

        # sanity checks
        assert inp_flat.ndim == 2, "Input must be 2‑D"
        assert fwd_expert_count.ndim == 1, "fwd_expert_count must be 1‑D"
        assert fwd_expert_count.shape[0] == self.num_experts, (
            f"Expected {self.num_experts} experts, got {fwd_expert_count.shape[0]}"
        )
        assert fwd_expert_count.sum().item() == inp_flat.shape[0], (
            f"Sum of counts ({fwd_expert_count.sum().item()}) != rows ({inp_flat.shape[0]})"
        )
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)

        if self.record_output and self.stage == 0:

            # reshaping into top_k
            assert x.shape[0] % self.top_k == 0, (
                f"Expected {self.top_k} experts, got {x.shape[0]} rows"
            )
            n = int(x.shape[0] / self.top_k)
            new_shape = (n, self.top_k) + x.shape[1:]
            x = x.view(new_shape)

            normalised_x = F.normalize(x, p=2, dim=-1)
            sim_matrix = torch.matmul(normalised_x, normalised_x.transpose(1, 2))

            print('sim_matrix', sim_matrix.shape)

            # ignore self-similarity (in diagonal)
            mask = ~torch.eye(self.top_k, dtype=bool, device=x.device).unsqueeze(0)
            sim_sum = sim_matrix[mask].view(n, -1).sum(dim=1)  # sum over off-diagonal
            avg_sim = sim_sum / (self.top_k * (self.top_k - 1))  
            loss += torch.abs(avg_sim.unsqueeze(1))
            
            splits = torch.split(x, fwd_expert_count.tolist(), dim=0)
            min_count = int(fwd_expert_count.min().item())
            out = torch.stack([chunk[:min_count] for chunk in splits], dim=0)
            if self.outputs is None:
                self.outputs = out
            elif self.outputs.shape[1] < self.outputs_size_limit:
                self.outputs = torch.cat((self.outputs, out), dim=1)

        rank = torch.distributed.get_rank()
        if rank == 1:
            wandb.log({'max expert output': x.max(), 'min expert output': x.min()})
        x = x.clamp(min=-0.1, max=0.1)
        return x
    
    def get_components(self, num_components=50):
        r'''
        Assuming the output matrix is non-empty, calculates the global
        and local components for each expert

        num_local is per expert
        '''
        print('calculting components')
        # send output to CPU and concvert to numpy
        # write matrix to a file so it can be reloaded
        get_num_global_components(self.outputs)
        ppca = PerPCA(num_components, num_components)
        if self.outputs is not None:
            return ppca.fit(np.array(self.outputs))
        else:
            raise ValueError('No outputs to calculate components')
        
    def factorise(self):
        '''
        Calculates components of the expert's outputs,
        then creates a new global expert'''
        get_num_global_components(self.outputs.swapaxes(-1, -2))
        # global_comp, local_comp = self.get_components()
        # global_comp = torch.tensor(np.array(global_comp), device='cuda')
        # local_comp = torch.tensor(np.array(local_comp), device='cuda')
        # # creating an array of component matricies
        # components = torch.cat((global_comp.unsqueeze(0), local_comp), dim=0)
        # # make sure components are float
        # components = components.float()
        # self.htoh4 = FMoELinearProj(components, prev_experts=self.htoh4)
        # self.h4toh = FMoELinearProj(components, prev_experts=self.h4toh)
        # self.stage = 1
        # self.num_experts += 1

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_gate=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        gate=NaiveGate,
        world_size=1,
        top_k=2,
        vmoe_noisy_std=1,
        gate_return_decoupled_activation=False,
        gate_task_specific_dim=-1,
        multi_gate=False,
        regu_experts_fromtask = False,
        num_experts_pertask = -1,
        num_tasks = -1,
        regu_sem = False,
        sem_force = False,
        regu_subimage = False,
        expert_prune = False,
        prune_threshold = 0.1,
        diversity_loss_weight = 0.0,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.our_d_gate = d_gate
        self.our_d_model = d_model
        self.factorised = False
        self.num_expert = num_expert
        self.regu_experts_fromtask = regu_experts_fromtask
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        self.regu_subimage = regu_subimage
        self.expert_prune = expert_prune
        self.prune_threshold = prune_threshold
        self.forced_expert = None
        self.diversity_loss_weight = diversity_loss_weight
        self.expert_outputs = None

        if self.sem_force:
            self.force_id=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]
        if self.regu_experts_fromtask:
            self.start_experts_id=[]
            start_id = 0
            for i in range(self.num_tasks):
                start_id = start_id + int(i* (self.num_expert-self.num_experts_pertask)/(self.num_tasks-1))
                self.start_experts_id.append(start_id)
            print('self.start_experts_id',self.start_experts_id)

        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank, top_k=top_k
        )
        self.gate_task_specific_dim = gate_task_specific_dim
        self.multi_gate=multi_gate
        if gate_task_specific_dim<0:
            d_gate = d_model
        else:
            d_gate = d_model+gate_task_specific_dim
        print('multi_gate',self.multi_gate)
        if gate == NoisyGate:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=self.regu_sem,sem_force = self.sem_force)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=self.regu_sem,sem_force = self.sem_force)
        elif gate == NoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation,
                    noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation,
                noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)

        else:
            raise ValueError("No such gating type")
        self.mark_parallel_comm(expert_dp_comm)

    def factorise_block(self):
        """
        Create a new global expert and factorise the current expert into local components
        """
        self.experts.factorise()
        # change the gate to route to the new global expert too
        if self.gate is not None:
            if self.multi_gate:
                self.gate = nn.ModuleList([NoisyGlobalGate_VMoE(g) for g in self.gate])
            else:
                self.gate = NoisyGlobalGate_VMoE(self.gate)
        self.gate.to('cuda')
        self.num_expert += 1
        self.factorised = True

    def dump_output(self):
        '''get each expert to print out the shape of its output matrix'''
        print(f'Experts output shape: {np.array(self.experts.outputs).shape}')


    def forward(self, inp: torch.Tensor, gate_inp=None, task_id = None, task_specific_feature = None, sem=None, record_expert_outputs=False, verbose=False):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        if gate_inp is None:
            gate_inp = inp

        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        gate_channel = gate_inp.shape[-1]
        gate_inp = gate_inp.reshape(-1, gate_channel)
        # print('task_id, task_specific_feature',task_id, task_specific_feature)
        if (task_id is not None) and (task_specific_feature is not None):
            assert self.multi_gate is False
            size = gate_inp.shape[0]
            gate_inp = torch.cat((gate_inp,task_specific_feature.repeat(size,1)),dim=-1)
        output = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, sem=sem, record_outputs=record_expert_outputs, verbose=verbose)
        return output.reshape(original_shape)
    
    def get_output_matrix(self):
        return self.experts.outputs.swapaxes(-1, -2)

    def set_forced_expert(self, expert_idx):
        """Force the layer to use a specific expert"""
        self.forced_expert = expert_idx
        
    def clear_forced_expert(self):
        """Clear the forced expert setting"""
        self.forced_expert = None

    def compute_diversity_loss(self, expert_outputs):
        """
        Compute diversity loss based on cosine similarity between expert outputs.
        
        Args:
            expert_outputs: Tensor of shape (num_experts, batch_size, d_model)
            
        Returns:
            diversity_loss: Scalar tensor measuring similarity between expert outputs
        """
        if expert_outputs is None or self.diversity_loss_weight == 0:
            return 0.0
        
        # Normalize expert outputs
        expert_outputs = F.normalize(expert_outputs, p=2, dim=-1)
        
        # Compute pairwise cosine similarities
        similarities = torch.matmul(expert_outputs, expert_outputs.transpose(1, 2))
        
        # Create mask to exclude self-similarities
        mask = torch.eye(similarities.size(1), device=similarities.device)
        mask = 1 - mask
        
        # Average similarity excluding self-similarity
        diversity_loss = (similarities * mask).sum() / (mask.sum() + 1e-6)
        
        return self.diversity_loss_weight * diversity_loss

    def forward_moe(self, gate_inp, moe_inp, task_id=None, sem=None, record_outputs=False, verbose =False):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
            tree.map_structure(ensure_comm_func, gate_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        if self.forced_expert is not None:
            # Override gate outputs to force specific expert
            batch_size = gate_inp.shape[0]
            gate_top_k_idx = torch.full((batch_size, self.top_k), self.forced_expert, 
                                      device=gate_inp.device)
            gate_score = torch.ones((batch_size, self.top_k), 
                                  device=gate_inp.device) / self.top_k
        else:
            if (task_id is not None) and self.multi_gate:
                # print('in custom moe_layer,task_id',task_id)
                gate_top_k_idx, gate_score = self.gate[task_id](gate_inp)
            else:
                gate_top_k_idx, gate_score = self.gate(gate_inp, task_id=task_id,sem=sem)

        if self.expert_prune:
            gate_score = torch.where(gate_score>self.prune_threshold,gate_score,0.)
            prune_prob = 1-torch.nonzero(gate_score).shape[0]/torch.cumprod(torch.tensor(gate_score.shape),dim=0)[-1]
            print('prune_prob',prune_prob)
        if self.sem_force and (sem is not None):
            batch = sem.shape[0]
            gate_top_k_idx = gate_top_k_idx.reshape(batch,-1,self.top_k)
            sem = sem.reshape(batch,-1)
            for k in range(batch):
                for i in range(sem.shape[-1]):
                    for j in range(len(self.force_id)):
                        if sem[k,i] in self.force_id[j]:
                            gate_top_k_idx[k,i+1,:]=[j*2,j*2+1]
            gate_top_k_idx = gate_top_k_idx.reshape(-1,self.top_k)
            gate_score =  torch.ones((gate_score.shape[0],self.top_k),device=gate_score.device)*0.5


        if self.regu_experts_fromtask and (task_id is not None):
            # print('task_id',self.start_experts_id[task_id],task_id)
            gate_top_k_idx = gate_top_k_idx + self.start_experts_id[task_id]

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # no idea how to actually pass this into the experts
        if record_outputs:
            self.experts.record_output = True

        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size
        )

        if verbose:
            # testing something
            other_gate_top_k_idx = gate_top_k_idx.clone()
            ones = torch.ones_like(gate_top_k_idx)
            other_gate_top_k_idx = other_gate_top_k_idx + ones
            if other_gate_top_k_idx[0][0] == self.num_expert:
                other_gate_top_k_idx = torch.zeros_like(gate_top_k_idx)
            
            other_fwd = _fmoe_general_global_forward(
                moe_inp, other_gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size
            )

            print(f'--- are they the same? (lower level) {torch.allclose(fwd, other_fwd)} {fwd.shape} \n {gate_top_k_idx} \n {other_gate_top_k_idx} \n original: {fwd} \n other: {other_fwd} ---')
        self.experts.record_output = False

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                total = tensor.numel()

                if not self.factorised:
                    group_size = self.top_k * dim
                    batch_positions = total // group_size
                    return tensor.view(batch_positions, self.top_k, dim)
                else:
                    group_size = (1 + self.top_k) * dim
                    batch_positions = total // group_size
                    return tensor.view(batch_positions, self.top_k + 1, dim)


            moe_outp = tree.map_structure(view_func, fwd)

        if self.factorised:
            gate_score = gate_score.view(-1, 1, self.top_k + 1)
        else:
            gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        # Store expert outputs for diversity loss if weight > 0
        if self.diversity_loss_weight > 0:
            # Reshape expert outputs to (num_experts, batch_size, d_model)
            expert_outputs = moe_outp.view(-1, self.num_expert, moe_outp.size(-1))
            expert_outputs = expert_outputs.transpose(0, 1)
            self.expert_outputs = expert_outputs
            
            # Compute diversity loss
            diversity_loss = self.compute_diversity_loss(expert_outputs)
            
            # Store loss for collection by the main training loop
            if not hasattr(self, 'diversity_losses'):
                self.diversity_losses = []
            self.diversity_losses.append(diversity_loss)

        return moe_outp
