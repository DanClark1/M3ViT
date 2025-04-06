#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
import numpy as np
from collections import Counter
import torch.nn.functional as F
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
import pickle
from os.path import join
from utils.moe_utils import collect_noisy_gating_loss,collect_semregu_loss, collect_regu_subimage_loss, collect_diversity_loss
from tqdm import tqdm
import wandb
def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES


    if p['model'] == 'mti_net': # Extra losses at multiple scales
        losses = {}
        for scale in range(4):
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    elif p['model'] == 'pad_net': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'padnet_vit': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'papnet_vit': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    
    elif p['model'] == 'jtrl': # Extra losses because of deepsupervision
        losses = {}
        if p['model_kwargs']['tam']:
            for task in tasks:
                losses['tam_%s' %(task)] = AverageMeter('Loss tam %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}

    if 'model_kwargs' in p:
        if 'tam' in p['model_kwargs']:
            if p['model_kwargs']['tam']:
                for task in tasks:
                    # losses['tam_%s' %(task)] = AverageMeter('Loss tam %s' %(task), ':.4e')
                    if 'tam_level0' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level0']:
                            losses['tam_level0_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(0, task), ':.4e')
                    else:
                        losses['tam_level0_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(0, task), ':.4e')
                    
                    if 'tam_level1' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level1']:
                            losses['tam_level1_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(1, task), ':.4e')
                    else:
                        losses['tam_level1_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(1, task), ':.4e')

                    if 'tam_level2' in p['model_kwargs']:
                        if p['model_kwargs']['tam_level2']:
                            losses['tam_level2_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(2, task), ':.4e')
                    else:
                        losses['tam_level2_%s' %(task)] = AverageMeter('Loss tam level %s %s' %(2, task), ':.4e')

    if p['multi_level']:
        for task in tasks:
            for i in range(1,4):
                losses['level%s_%s'%(i,task)] = AverageMeter('At level %s Loss %s' %(i,task), ':.4e')

    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses

def logits_aug(logits, low=0.01, high=9.99):
    batch_size = logits.size(0)
    temp = torch.autograd.Variable(torch.rand(batch_size, 1) * high + low).cuda()
    logits_temp = logits / temp
    return logits_temp

def adjust_epsilon_greedy(p, epoch):
    return 0.5 * max((1 - epoch/(p['epochs'] - p['left'])), 0)

def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def train_mixture_vanilla(p, train_loader, model,prior_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    prior_model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        
        # input_var = Variable(images)
        prior_out, overhead_flop = prior_model(images)
        if p['anneal']:
            prob = adjust_epsilon_greedy(p, epoch)
            print('epsilon greedy prob: {}'.format(prob))
        # output = model(images)
        if p['anneal']:
            # out, masks, costs, flop_percent = model(
            #     images, F.softmax(logits_aug(prior_out) if p['data_aug']
            #                          else prior_out, dim=-1), overhead_flop, prob)
            output = model(
                images, F.softmax(logits_aug(prior_out) if p['data_aug']
                                     else prior_out, dim=-1), overhead_flop, prob)
        else:
            # out, masks, costs, flop_percent = model(
            #     images, F.softmax(logits_aug(prior_out) if p['data_aug'] else
            #                          prior_out, dim=-1), overhead_flop)
            
            output = model(
                images, F.softmax(logits_aug(prior_out) if p['data_aug'] else
                                     prior_out, dim=-1), overhead_flop)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def train_vanilla_distributed(args, p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    accumulation_steps = args.train_accumulation_steps
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(args.local_rank, non_blocking=True)
        targets = {task: batch[task].cuda(args.local_rank, non_blocking=True) for task in p.ALL_TASKS.NAMES}
        
        if args.one_by_one:
            optimizer.zero_grad()
            id=0
            for single_task in p.TASKS.NAMES:
                if args.task_one_hot:
                    output = model(images,single_task=single_task, task_id = id)
                else:
                    output = model(images,single_task=single_task)
                id=id+1
                loss_dict = criterion(output, targets, single_task)

                for k, v in loss_dict.items():
                    losses[k].update(v.item())
                performance_meter.update({single_task: get_output(output[single_task], single_task)}, 
                                 {single_task: targets[single_task]})
                
                if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                    loss_dict['total'] += collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                # Backward
                loss_dict['total'].backward()
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                    model.allreduce_params()

            optimizer.step()
                
        else:
            # if (args.regu_sem or args.sem_force or args.regu_subimage) and epoch<args.warmup_epochs:
            #     output = model(images,sem=targets['semseg'])
            # else:
            output = model(images, isval=True)
            rank = torch.distributed.get_rank()
            if rank == 1:
                # log the max and min value in the output
                wandb.log({"max output": output['semseg'].max(), "min output": output['semseg'].min()})
                
            # Measure loss and performance
            loss_dict = criterion(output, targets)

            matricies = []
           
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                diversity_loss_coeff = 1 # might be too high tbf
                main_loss = loss_dict['total']
                gating_loss = collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                loss_dict['total'] += gating_loss
                similarity_loss= calculate_moe_cosine_similarity_loss(model).squeeze().cpu().detach()
                lambda_loss = calculate_power_iteration_diversity_loss(model).squeeze().cpu().detach()


                per_token_cosine_loss = 0
                layer_n = 0
                for block in model.module.backbone.blocks:
                    if block.moe:
                        per_token_cosine_loss += block.mlp.experts.loss / block.mlp.experts.loss_normalise_weight
                        block.mlp.experts.reset_loss()
                        layer_n += 1

                
                # loss_dict['total'] += per_token_cosine_loss / layer_n
                # # lambda_loss.register_hook(lambda grad: grad.clamp(-0.5, 0.5))
                # diversity_loss = calculate_moe_diversity_loss(model)

                # loss_dict['total'] += (diversity_loss * diversity_loss_coeff)
                
                # wandb.log({"overall loss": loss_dict['total'].item(), "main loss": main_loss.item(), "diversity loss": diversity_loss.item(), "gating_loss": gating_loss.item()})
                #print(loss_dict['total'])
                #print(calculate_moe_cosine_similarity_loss(model).shape)
                # Force both to be scalars before summing, then restore shape if necessary
                
                loss_total = loss_dict['total'].squeeze() + similarity_loss
                loss_dict['total'] = loss_total.unsqueeze(0)  # If downstream code expects shape [1]
                rank = torch.distributed.get_rank()
                if rank == 1:
                    wandb.log({"per token cosine":(per_token_cosine_loss / layer_n), "overall loss": loss_dict['total'].item(), "main loss": main_loss.item(), "similarity loss": similarity_loss.item(), "gating_loss": gating_loss.item()})

                for block in model.module.backbone.blocks:
                    if block.moe:
                        block.mlp.experts.reset_outputs()
                    
            for k, v in loss_dict.items():
                losses[k].update(v.item())
            performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                    {t: targets[t] for t in p.TASKS.NAMES})
            # Backward

            
            loss = loss_dict['total'] / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # print('backward step')
                optimizer.step()
                optimizer.zero_grad()

                if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                    model.allreduce_params()
            
            
        if i % 25 == 0:
            print('similarity_loss',similarity_loss)
            progress.display(i)
            # for name, param in model.named_parameters():
            #     if 'gamma' in name:
            #         print('gamma',param)
            # if args.regu_sem and epoch<args.warmup_epochs:
            #     print('semregu_loss',semregu_loss)
            # if args.regu_subimage and epoch<args.warmup_epochs:
            #     print('regu_subimage_loss',regu_subimage_loss)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def calculate_moe_cosine_similarity_loss(model, coefficient=0.1):
    '''
    Takes the an image in a batch and computes the diversity loss (assuming model is moe)

    This works by basically forcing a certain expert, and then doing n forward
    passes through the model and recording it. This is done for every expert,
    creating an nxd matrix for each expert.

    We then take these and measure the alignment of their bases
    '''
    backbone = model.module.backbone
    num_experts = 16
    num_layers = 6

    # Assuming that for each block, block.mlp.get_output_matrix() returns a list or tensor for each expert.
    layers = [block.mlp.get_output_matrix() for block in backbone.blocks if block.moe]
    
    total_cosine = 0.0


    for layer_idx in range(num_layers):
        # `clients` is assumed to be a list with length = num_experts,
        # each element with shape (d, b) (d: feature dimension, b: batch size or other dimension)
        clients = layers[layer_idx]
        # Stack expert outputs to create a tensor of shape (num_experts, d, b)
        clients_tensor = torch.stack([clients[e] for e in range(num_experts)], dim=0)
        # normalising for similarity + reshape into (b, d, num_experts)
        clients_tensor = F.normalize(clients_tensor, dim=1).transpose(1, 2)

        # Compute pairwise cosine similarity for each pair of experts
        layer_cosine = 0.0
        pair_count = 0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # F.cosine_similarity returns a 1-element tensor when inputs are 1D
                similarity = F.cosine_similarity(clients_tensor[:, :, i], clients_tensor[:, :, j], dim=1)
                print('similarity', similarity.mean())
                layer_cosine += similarity.mean() / 2
                pair_count += 1
        # Average cosine similarity for the current layer
        total_cosine += layer_cosine / pair_count

    
    # Optionally, log the total similarity for debugging (consider logging less frequently)
    # print(total_similarity, ', end')
    return torch.abs(coefficient * total_cosine)



def power_iteration(A, num_iters=20):
    """
    Compute the dominant eigenvector of a square matrix A using power iteration.
    A: square matrix of shape (d, d)
    Returns a unit vector of shape (d, 1) approximating the top eigenvector.
    """
    d = A.shape[0]
    v = torch.randn(d, 1, device=A.device)
    v = v / v.norm()
    for _ in range(num_iters):
        v = A @ v
        v = v / v.norm()
    return v



def compute_local_basis(Y, num_iters=20):
    """
    Compute the dominant eigenvector (local basis) from data matrix Y.
    Y: data matrix for a client of shape (d, n)
    Returns a unit vector v (d x 1) approximating the top eigenvector of S = Y Y^T / n.
    """
    n = Y.shape[1]
    S = Y @ Y.t() / n
    v = power_iteration(S, num_iters=num_iters)
    return v


def calculate_moe_diversity_loss(model):
    '''
    Takes the an image in a batch and computes the diversity loss (assuming model is moe)

    This works by basically forcing a certain expert, and then doing n forward
    passes through the model and recording it. This is done for every expert,
    creating an nxd matrix for each expert.

    We then take these and measure the alignment of their bases
    '''
    backbone = model.module.backbone
    num_experts = 16
    num_layers = 6

    # Assuming that for each block, block.mlp.get_output_matrix() returns a list or tensor for each expert.
    layers = [block.mlp.get_output_matrix() for block in backbone.blocks if block.moe]
    
    lambda_total = 0.0
    layer_count = 0.0

    for layer_idx in range(num_layers):
        # `clients` is assumed to be a list with length = num_experts,
        # each element shape (d, b)
        clients = layers[layer_idx]

        # Stack expert outputs to create tensor of shape (num_experts, d, b)
        clients_tensor = torch.stack([clients[e] for e in range(num_experts)], dim=0)


        if torch.isnan(clients_tensor).any():
            raise ValueError(f"NaNs detected in clients_tensor after normalization.")
        
        eps = 1e-6
        # Adding eps for numerical stability if needed
        Q, _ = torch.linalg.qr(clients_tensor + eps, mode='reduced')
        # Q now has shape (num_experts, d, r) where r = min(d, b)

        # alternatively, try this but with svd
        U, _, _ = torch.linalg.svd(clients_tensor + eps)
        Q = U
        if torch.isnan(Q).any():
            raise ValueError("NaNs detected in Q matrix after QR decomposition.")


        projs = torch.matmul(Q, Q.transpose(1, 2))


        # Compute the average projection across experts: shape (d, d)
        avg_proj = torch.mean(projs, dim=0)
        
        # Compute the eigenvalues of the averaged projection (avg_proj is symmetric)
        eigvals = torch.linalg.eigvalsh(avg_proj)
        lambda_max = eigvals[-1]
        
        # Calculate theta for this layer
        lambda_total += lambda_max
        layer_count += 1

    # Average theta across layers
    avg_lambda = lambda_total / layer_count if layer_count > 0 else 0.0

    rank = torch.distributed.get_rank()
    if rank == 1:
        wandb.log({"diversity loss": lambda_max.item()})
    target = 1.0 / 16 # Assuming num_experts = 16
    alpha = 10
    loss_below = alpha * torch.square(torch.maximum(target - avg_lambda, torch.tensor(0, device='cuda')))
    loss_above = torch.square(torch.maximum(avg_lambda - target, torch.tensor(0, device='cuda')))
    return loss_below + loss_above




def calculate_power_iteration_diversity_loss(model):
    '''
    Same as diversity loss but trying to estimate without doing any matrix decomposition
    '''


    def batched_power_iteration(cov, num_iters=100):
        """
        Perform batched power iteration on a batch of covariance matrices.
        cov: tensor of shape (N, d, d)
        Returns: tensor of shape (N, d, 1) containing the top eigenvector for each matrix.
        """
        N, d, _ = cov.shape
        # Initialize a random vector for each client (batch element)
        v = torch.randn(N, d, 1, device=cov.device)
        # Normalize along the d-dimension
        v = v / v.norm(dim=1, keepdim=True)
        for _ in range(num_iters):
            v = torch.bmm(cov, v)      # Batched matrix multiplication: (N, d, d) x (N, d, 1) -> (N, d, 1)
            v = v / v.norm(dim=1, keepdim=True)

        print('power iteration: ',v.shape)
        return v
    

    def batched_svd(cov):
        """
        Compute the SVD of a batch of matrices.
        cov: tensor of shape (N, d, d)
        Returns: tensor of shape (N, d, 1) containing the top eigenvector.
        """
        U, _, _ = torch.linalg.svd(cov)
        #return U[:, :, 0].unsqueeze(-1)
        return U

    def compute_avg_projection(Y, num_iters=20):
        """
        Given a batch of data matrices Y of shape (N, d, n),
        compute the average projection matrix from the top eigenvectors of the covariance matrices.
        Returns:
        avg_proj: averaged projection matrix of shape (d, d)
        v: tensor of shape (N, d, 1) with the top eigenvector for each client.
        """
        N, d, n = Y.shape
        # Compute the covariance matrix for each client: S = Y Y^T / n.
        cov = torch.bmm(Y, Y.transpose(1, 2)) / n  # shape (N, d, d)
        # Compute top eigenvectors using batched power iteration.
        v = batched_power_iteration(cov, num_iters=num_iters)  # shape (N, d, 1)
        v_other = batched_svd(cov)  # shape (N, d, 1)
        # Check if the two methods give similar results
        # if torch.allclose(v, v_other, atol=1e-2):
        #     print("Both methods yield similar results.")
        # else:
        #     print("Methods yield different results.")
        v = v_other
        # Build projection matrices for each client: Pi = v v^T.
        proj = torch.bmm(v, v.transpose(1, 2))  # shape (N, d, d)
        # Average the projection matrices over all clients.
        avg_proj = proj.mean(dim=0)  # shape (d, d)
        return avg_proj, v

    def batched_power_iteration_single(A, num_iters=20):
        """
        Run power iteration on a single matrix A (d x d) to compute its top eigenvector.
        Returns the top eigenvalue (scalar) as (v^T A v).
        """
        d = A.shape[0]
        v = torch.randn(d, 1, device=A.device)
        v = v / v.norm()
        for _ in range(num_iters):
            v = A @ v
            v = v / v.norm()
        lambda_max = (v.transpose(0, 1) @ A @ v).squeeze()
        return lambda_max

    def asymmetric_loss(lambda_max, N, alpha=10.0):
        """
        Compute the asymmetric loss on lambda_max.
        Target value is 1/N. Values below 1/N are penalized by a factor of alpha.
        """
        target = 1.0 / N
        loss_below = alpha * torch.square(torch.clamp(target - lambda_max, min=0))
        loss_above = torch.square(torch.clamp(lambda_max - target, min=0))
        return loss_below + loss_above
    

    backbone = model.module.backbone
    num_experts = 16
    num_layers = 6

    # Assuming that for each block, block.mlp.get_output_matrix() returns a list or tensor for each expert.
    layers = [block.mlp.get_output_matrix() for block in backbone.blocks if block.moe]
    
    avg_lambda = 0.0
    layer_count = 0.0

    for layer_idx in range(num_layers):
        # `clients` is assumed to be a list with length = num_experts,
        # each element shape (d, b)
        clients = layers[layer_idx]

        # Stack expert outputs to create tensor of shape (num_experts, d, b)
        clients_tensor = torch.stack([clients[e] for e in range(num_experts)], dim=0)

        avg_proj, v = compute_avg_projection(clients_tensor, num_iters=50)
        # Compute the largest eigenvalue of the averaged projection matrix via power iteration.
        lambda_max = torch.linalg.torch.linalg.svdvals(avg_proj)[0] 
        avg_lambda += lambda_max
        layer_count += 1
    
    avg_lambda /= layer_count

    rank = torch.distributed.get_rank()
    if rank == 1:
        wandb.log({"lambda_max": lambda_max.item()})
    return asymmetric_loss(lambda_max, num_experts, alpha=10.0)
