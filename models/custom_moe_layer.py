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

from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.gate_funs.noisy_gate_vmoe_global import NoisyGlobalGate_VMoE

from utils.perpca import PerPCA

from pdb import set_trace
import numpy as np

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

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
        self.cosine_loss = 0
        self.cosine_normalise_weight = 0
        self.lambda_max_loss = 0
        self.lambda_max_normalise_weight = 0
        self.frobenius_loss = 0
        self.frobenius_normalise_weight = 0
        self.use_lambda = False
        self.use_cosine = False
        

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
            num_expert, d_model, d_hidden, activation, rank=expert_rank
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

    def reset_cosine_loss(self):
        self.cosine_loss = 0
        self.cosine_normalise_weight = 0


    def reset_lambda_loss(self):
        self.lambda_max_loss = 0
        self.lambda_max_normalise_weight = 0

    def reset_frobenius_loss(self):
        self.frobenius_loss = 0
        self.frobenius_loss_normalise_weight = 0

    def dump_output(self):
        '''get each expert to print out the shape of its output matrix'''
        print(f'Experts output shape: {np.array(self.experts.outputs).shape}')


    def forward(self, inp: torch.Tensor, gate_inp=None, task_id = None, task_specific_feature = None, sem=None, record_expert_outputs=False):
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
        output = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, sem=sem, record_outputs=record_expert_outputs)
        return output.reshape(original_shape)
    
    def get_output_matrix(self):
        return torch.cat(self.experts.outputs, dim=0).T


    def forward_moe(self, gate_inp, moe_inp, task_id=None, sem=None, record_outputs=False):
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


        gate_score = gate_score.view(-1, 1, self.top_k)

        #self.calculate_lambda_max_loss(moe_outp, gate_top_k_idx)
        self.calculate_frobenius_loss(moe_outp, gate_top_k_idx)
        #self.calculate_cosine_loss(moe_outp)

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
        return moe_outp


    def calculate_cosine_loss(self, moe_outp):
        '''
        moe output has shape (batch_positions, top_k, dim)
        '''
        # Normalize the tokens along the feature dimension:
        norm_tokens = F.normalize(moe_outp, p=2, dim=-1)  # shape: (batch_positions, top_k, dim)
        
        # Compute cosine similarity matrix for each sample:
        # This produces a (batch_positions, top_k, top_k) tensor where each [i] contains the pairwise similarities.
        cos_sim_matrix = torch.abs(torch.bmm(norm_tokens, norm_tokens.transpose(1, 2)))
        
        # Create a mask to remove self-similarities (the diagonal elements for each sample)
        top_k = moe_outp.size(1)
        diag_mask = torch.eye(top_k, device=moe_outp.device, dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.expand_as(cos_sim_matrix)
        cos_sim_matrix = cos_sim_matrix.masked_fill(diag_mask, 0)
        
        # Calculate the mean cosine similarity loss per sample.
        # Since each sample has top_k tokens, there are top_k * (top_k - 1) off-diagonals.
        # Sum across the top_k x top_k matrix (which now contains zeros on the diagonal), then average.
        cosine_loss = cos_sim_matrix.sum(dim=(1, 2)) / (top_k * (top_k - 1))
        
        # Finally, take the mean over all batch positions.
        cosine_loss = cosine_loss.mean()
        
        # Record the loss
        self.cosine_loss += cosine_loss
        self.cosine_normalise_weight += 1


    def old_calculate_lambda_max_loss(self, moe_outp, gate_top_k_idx):
        # shapes and dims
        batch_positions = moe_outp.shape[0]
        dim = moe_outp.shape[-1]
        device = moe_outp.device

       
       # adding zero vectors for padding at each forward pass
        expert_out_matrix = torch.zeros(
            batch_positions, self.num_expert, dim, device=device
        )
        rows = torch.arange(batch_positions, device=device).unsqueeze(-1) 
        expert_out_matrix[rows, gate_top_k_idx, :] = moe_outp


        # limiting the number of outputs to a certain size
        clients_tensor = expert_out_matrix[:, :, :]


        # reshaping to(batch_positions, dim, n_experts)
        clients_tensor = clients_tensor.swapaxes(-1, -2)

    

        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor before normalization.")

        clients_tensor = F.normalize(clients_tensor, p=2, dim=-1)

        
        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor after normalization.")

        d = clients_tensor.shape[1]
        avg_proj = torch.zeros(d, d, device=device)


        for i in range(self.num_expert):
            eps = 1e-6
            A = clients_tensor[:, :, i]
            Q, R = torch.linalg.qr(A.T, mode="reduced")
            r_diag = torch.diagonal(R, dim1=-2, dim2=-1)
            k = torch.min((r_diag.abs() > eps).sum())
            Q = Q[:, :k]
            if torch.isnan(Q).any():
                raise ValueError("NaNs detected in Q after SVD‐based basis extraction.")
            projs = Q.matmul(Q.transpose(-2, -1))
            avg_proj += projs

        avg_proj /= self.num_expert

        eigvals = torch.linalg.eigvalsh(avg_proj)
        lambda_max = eigvals[-1]
        
        return lambda_max

    def calculate_lambda_max_loss(self, moe_outp, gate_top_k_idx):       
        # shapes and dims
        batch_size = moe_outp.shape[0]
        dim = moe_outp.shape[-1]
        device = moe_outp.device

       # adding zero vectors for padding at each forward pass
        expert_out_matrix = torch.zeros(
            batch_size, self.num_expert, dim, device=device
        )
        rows = torch.arange(batch_size, device=device).unsqueeze(-1) 
        expert_out_matrix[rows, gate_top_k_idx, :] = moe_outp

        clients_tensor = expert_out_matrix.swapaxes(-1, -2).contiguous()

        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor before normalization.")

        clients_tensor = F.normalize(clients_tensor, p=2, dim=-1)

    
        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor after normalization.")

        A = clients_tensor.permute(2, 1, 0).contiguous()   
        eps = 1e-6

        Q, R = torch.linalg.qr(A, mode="reduced")

            
        r_diag = R.abs().diagonal(dim1=-2, dim2=-1)           # (E, min(d,B))
        k      = (r_diag > eps).sum(dim=1)                    # (E,)
        cols   = torch.arange(Q.size(-1), device=Q.device)    # (d,)
        mask   = cols[None, None, :] < k[:, None, None]       # (E, 1, d)
        Qm     = Q * mask                                     
        projs  = Qm @ Qm.transpose(-2, -1) 
        avg_proj    = projs.mean(dim=0) 

        eigvals = torch.linalg.eigvalsh(avg_proj)
        lambda_max = eigvals[-1]
    
        self.lambda_max_loss += lambda_max
        self.lambda_max_normalise_weight += 1



    def calculate_frobenius_loss(self, moe_outp, gate_top_k_idx):
        # shapes and dims
        batch_size = moe_outp.shape[0]
        dim = moe_outp.shape[-1]
        device = moe_outp.device

       # adding zero vectors for padding at each forward pass
        expert_out_matrix = torch.zeros(
            batch_size, self.num_expert, dim, device=device
        )
        rows = torch.arange(batch_size, device=device).unsqueeze(-1) 
        expert_out_matrix[rows, gate_top_k_idx, :] = moe_outp

        clients_tensor = expert_out_matrix.swapaxes(-1, -2).contiguous()

        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor before normalization.")

        clients_tensor = F.normalize(clients_tensor, p=2, dim=-1)

    
        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor after normalization.")

        A = clients_tensor.permute(2, 1, 0).contiguous()   
        eps = 1e-6

        Q, R = torch.linalg.qr(A, mode="reduced")

            
        r_diag = R.abs().diagonal(dim1=-2, dim2=-1)           # (E, min(d,B))
        k      = (r_diag > eps).sum(dim=1)                    # (E,)
        cols   = torch.arange(Q.size(-1), device=Q.device)    # (d,)
        mask   = cols[None, None, :] < k[:, None, None]       # (E, 1, d)
        Qm     = Q * mask                                     
        projs  = Qm @ Qm.transpose(-2, -1) 

        pairwise_loss = 0.0
        num_pairs = 0
        for i in range(self.num_expert):
            for j in range(i + 1, self.num_expert):
                diff = projs[i] - projs[j]
                pairwise_loss += torch.norm(diff, p='fro')**2
                num_pairs += 1


        pairwise_loss = pairwise_loss / max(1, num_pairs)
        
        self.frobenius_loss += pairwise_loss
        self.frobenius_normalise_weight += 1