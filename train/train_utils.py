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


global_step = 0


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
        step = epoch * len(train_loader) + i  
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
                # log the max and min value in the output
            wandb.log({"max output": output['semseg'].max(), "min output": output['semseg'].min()}, commit=False, step=step)
                
            # Measure loss and performance
            loss_dict = criterion(output, targets)

            matricies = []
           
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                main_loss = loss_dict['total'].clone()
                gating_loss = collect_noisy_gating_loss(model, args.moe_noisy_gate_loss_weight)
                loss_dict['total'] += gating_loss
                # lambda_loss = get_lambda_loss(model, step, detach=True)
                # cosine_loss = get_cosine_loss(model, step)
                # frobenius_loss = get_frobenius_loss(model, step, detach=True)

                # loss_dict['total'] += cosine_loss
                
                rank = torch.distributed.get_rank()
                wandb.log({"overall loss": loss_dict['total'].item(), "main loss": main_loss.item(),"gating_loss": gating_loss.item()}, step=step)

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
            progress.display(i)
            # for name, param in model.named_parameters():
            #     if 'gamma' in name:
            #         print('gamma',param)
            # if args.regu_sem and epoch<args.warmup_epochs:
            #     print('semregu_loss',semregu_loss)
            # if args.regu_subimage and epoch<args.warmup_epochs:
            #     print('regu_subimage_loss',regu_subimage_loss)

    eval_results = performance_meter.get_score(verbose = True)
    wandb.log({'train semseg mean iou': eval_results['semseg']['mIoU']}, commit=False, step=step)
    wandb.log({'train human_parts mean iou': eval_results['human_parts']['mIoU']}, commit=False, step=step)
    wandb.log({'train normals mean error': eval_results['normals']['mean']}, commit=False, step=step)
    wandb.log({'train sal mean iou': eval_results['sal']['mIoU']}, commit=False, step=step)
    wandb.log({'train edge loss': eval_results['edge']['loss']}, step=step)

    return eval_results



def get_lambda_loss(model, step,  coeff=1.0, T=0.85, detach=False):
    '''
    Computes the lambda_max loss for the model using a thresholded squared penalty.
    
    Instead of directly using the layer.loss values, we calculate the excess over T
    and square that. This gives a stronger gradient when the loss is above T, but no
    penalty when the value is below T.
    
    Args:
        model: The model with a backbone that contains MoE blocks.
        coeff (float): Coefficient to scale the loss.
        T (float): The threshold below which no penalty is applied (e.g. 0.85).
        detach (bool): Whether to detach the computed loss.
    
    Returns:
        The aggregated lambda loss multiplied by coeff.
    '''
    backbone = model.module.backbone
    layers = [block.mlp for block in backbone.blocks if block.moe]
    loss = 0.0
    total_lambda_val = 0.0

    for layer in layers:
        # Compute the normalized lambda value for this layer:
        lambda_val = layer.loss / layer.loss_normalise_weight

        total_lambda_val += lambda_val
        # Reset the stored lambda loss for the next forward pass
        layer.reset_lambda_loss()

    total_lambda_val = (total_lambda_val / len(layers)).detach().cpu()
    
    # Log the loss (you could also log the individual lambda value if needed)
    wandb.log({"lambda loss": total_lambda_val.item()}, step=step, commit=False)
    # wandb.log({"thresholded lambda loss": loss.item()})

    return total_lambda_val * coeff



def get_cosine_loss(model, step, coeff=1.0, detach=False):

    backbone = model.module.backbone

    layers = [block.mlp for block in backbone.blocks if block.moe]
    loss = 0.0

    for layer in layers:
        if detach:
            loss += (layer.cosine_loss / layer.cosine_normalise_weight).detach().cpu()
        else:
            loss += layer.cosine_loss / layer.cosine_normalise_weight
        layer.reset_cosine_loss()

    loss = loss / len(layers)
    
    rank = torch.distributed.get_rank()
    wandb.log({"cosine loss": loss.item()}, step=step, commit=False)

    return loss * coeff


def get_frobenius_loss(model, step, coeff=1.0, detach=False):
    """
    Aggregates the Frobenius norm regularisation loss over all MoE layers in the model.
    
    For each MoE layer in model.module.backbone.blocks with MoE enabled,
    it calls the layer's calculate_frobenius_loss() method and then averages the losses.
    
    Args:
        model: The overall model (assumed to have model.module.backbone.blocks).
        coeff (float): Coefficient to scale the aggregated loss.
        detach (bool): If True, the loss is detached from the graph.
        
    Returns:
        A scalar tensor representing the aggregated Frobenius loss multiplied by coeff.
    """
    backbone = model.module.backbone
    # Gather the MLP modules from the blocks that are MoE layers.
    layers = [block.mlp for block in backbone.blocks if block.moe]
    
    total_loss = 0.0
    num_layers = 0
    for layer in layers:
        # Compute the Frobenius loss for this layer.
        loss = layer.frobenius_loss / layer.frobenius_normalise_weight
        layer.reset_frobenius_loss()
        total_loss += loss
        num_layers += 1

    if num_layers > 0:
        avg_loss = total_loss / num_layers
    else:
        # If no MoE layers, return zero.
        avg_loss = torch.tensor(0.0, device=model.module.backbone.blocks[0].mlp.experts.htoh4.weight.device)

    if detach:
        avg_loss = avg_loss.detach().cpu()

    wandb.log({"frobenius loss": avg_loss.item()}, step=step, commit=False)
    return avg_loss * coeff
