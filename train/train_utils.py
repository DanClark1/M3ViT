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
            
            
            # Measure loss and performance
            loss_dict = criterion(output, targets)

            matricies = []
           
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                loss_dict['total'] += collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                # loss_dict['total'] += calculate_moe_diversity_loss(model)
                loss_dict['total'] += calculate_moe_cosine_similarity_loss(model)
                    
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
            # don't know what this does or when it should be called for accumulated training so i'm ignoring it
            # if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
            #     model.allreduce_params()
            
            
        if i % 25 == 0:
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

def calculate_moe_cosine_similarity_loss(model, coefficient=1):
    '''
    Computes a diversity loss based on cosine similarity between the outputs of experts.
    
    For each MoE layer in the backbone, this function stacks the expert outputs,
    normalizes them, flattens each expert’s output, and then computes the pairwise
    cosine similarity. High cosine similarity indicates less diversity, so the loss
    encourages the experts to have lower (or more negative) similarity.
    
    Args:
        model: The model containing a MoE backbone.
        coefficient: A scaling coefficient for the loss.
    
    Returns:
        A scalar loss encouraging diversity between expert outputs.
    '''
    backbone = model.module.backbone
    num_experts = 16  # Adjust if needed
    num_layers = 6    # Adjust if needed

    # Get output matrices from each MoE block in the backbone
    layers = [block.mlp.get_output_matrix() for block in backbone.blocks if block.moe]
    
    total_cosine = 0.0

    for layer_idx in range(num_layers):
        # `clients` is assumed to be a list with length = num_experts,
        # each element with shape (d, b) (d: feature dimension, b: batch size or other dimension)
        clients = layers[layer_idx]
        # Stack expert outputs to create a tensor of shape (num_experts, d, b)
        clients_tensor = torch.stack([clients[e] for e in range(num_experts)], dim=0)
        # Normalize along the feature dimension for cosine similarity (here dim=1)
        clients_tensor = F.normalize(clients_tensor, dim=1)
        # Flatten each expert output to a vector of shape (d*b,)
        clients_flat = clients_tensor.view(num_experts, -1)
        
        # Compute pairwise cosine similarity for each pair of experts
        layer_cosine = 0.0
        pair_count = 0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # F.cosine_similarity returns a 1-element tensor when inputs are 1D
                cos_sim = F.cosine_similarity(clients_flat[i].unsqueeze(0), clients_flat[j].unsqueeze(0), dim=1)
                layer_cosine += cos_sim
                pair_count += 1
        # Average cosine similarity for the current layer
        total_cosine += layer_cosine / pair_count

    # Average over all layers
    total_cosine /= num_layers

    # Reset expert outputs for each block
    for block in backbone.blocks:
        if block.moe:
            block.mlp.experts.reset_outputs()
    
    # Return the loss scaled by the coefficient.
    # If the experts are highly similar (cosine close to 1), the loss is high.
    return (coefficient * total_cosine).unsqueeze(0)



def calculate_moe_diversity_loss(model, coefficient=1):
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
    
    total_similarity = 0.0

    for layer_idx in range(num_layers):
        # `clients` is assumed to be a list with length = num_experts,
        # each element shape (d, b)
        clients = layers[layer_idx]

        # Stack expert outputs to create tensor of shape (num_experts, d, b)
        clients_tensor = torch.stack([clients[e] for e in range(num_experts)], dim=0)
        clients_tensor = F.normalize(clients_tensor, dim=1)  
        
        # Batched QR decomposition (in reduced mode), Q: (num_experts, d, r)
        eps = 1e-6
        U, _, _ = torch.linalg.svd(clients_tensor + eps, full_matrices=False)
        Q = U  # Use left-singular vectors as basis
        
        # Compute pairwise similarity between the orthonormal bases
        # Q: (N, d, r) -> transpose Q for inner product: (N, r, d)
        Q_T = Q.transpose(1, 2)
        # Compute inner products for each pair: shape (N, N, r, r)
        inner = torch.matmul(Q_T.unsqueeze(1), Q.unsqueeze(0))
        # Squared Frobenius norm of each inner product matrix: (N, N)
        pairwise_similarity = inner.pow(2).sum(dim=(-1, -2))
        
        # Use only upper triangle (exclude diagonal) and sum
        total_similarity += pairwise_similarity.triu(diagonal=1).sum()

    # Normalize by the number of pairs and layers
    num_pairs = num_experts * (num_experts - 1) / 2 * num_layers
    total_similarity /= num_pairs

    # divide by dimension of the output
    total_similarity /= clients_tensor.size(1)

    # Reset expert outputs for each block
    for block in backbone.blocks:
        if block.moe:
            block.mlp.experts.reset_outputs()
    
    # Optionally, log the total similarity for debugging (consider logging less frequently)
    # print(total_similarity, ', end')

    return coefficient * total_similarity
