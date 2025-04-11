#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import glob
import torch
from torch.nn.parallel import DistributedDataParallel
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.factorise_model import factorise_model
from utils.logger import Logger
from train.train_utils import train_vanilla,train_vanilla_distributed
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results,validate_results_v2
from termcolor import colored
import wandb
import torch.distributed as dist
import subprocess
import random
from utils.custom_collate import collate_mil
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_config import build_train_dataloader,build_val_dataloader
from utils.moe_utils import sync_weights,save_checkpoint
import time
import fmoe
from thop import clever_format
from utils.model_utils import save_model

from thop import profile
from torch.utils.data import Subset
import random
import datetime
def save_checkpoint(model, optimizer, epoch, path):
        # Only save from process with global rank 0
        if dist.get_rank() == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),  # unwrap DDP
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
        dist.barrier()  # sync all ranks


def load_for_training(model, optimizer, path, device):

        # Wrap model first
        dist.barrier()  # wait for rank 0 to write file

        checkpoint = torch.load(path, map_location=f"{device}")
        try:
            model.load_state_dict(checkpoint["model_state"])
        except:
            model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

        return model, optimizer, start_epoch

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
parser.add_argument("--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
parser.add_argument('--moe_data_distributed', action='store_true', help='if employ moe data distributed')
parser.add_argument('--moe_experts', default=16, type=int, help='moe experts number')
parser.add_argument('--moe_mlp_ratio', default=None, type=int, help='moe experts mlp ratio')
parser.add_argument('--moe_top_k', default=None, type=int, help='top k expert number')
parser.add_argument('--trBatch', default=None, type=int, help='train batch size')
parser.add_argument('--valBatch', default=None, type=int, help='validation batch size')
parser.add_argument('--moe_gate_arch', default="", type=str)
parser.add_argument('--moe_gate_type', default="noisy", type=str)
parser.add_argument('--vmoe_noisy_std', default=1, type=float)
parser.add_argument('--backbone_random_init',default=False, type=str2bool, help='whether randomly initialize backbone')
parser.add_argument('--pretrained', default='', type=str, help='path to moe pretrained checkpoint')
parser.add_argument('--moe_noisy_gate_loss_weight', default=0.01, type=float)


parser.add_argument('--pos_emb_from_pretrained', default=False, type=str, help='pos embedding load from pretrain weights')
parser.add_argument('--lr', default=None, type=float)
# parser.add_argument('--weight_decay', default=None, type=float)

parser.add_argument('--one_by_one',default=False, type=str2bool, help='whether train task one after another')
parser.add_argument('--task_one_hot',default=False, type=str2bool, help='whether use Task-conditioned MoE')
parser.add_argument('--multi_gate',default=False, type=str2bool, help='whether use Multi gate MoE')

parser.add_argument('--eval', action='store_true',help='if only do evaluation')
parser.add_argument('--flops', action='store_true',
        help='flops calculation')
parser.add_argument('--ckp',type=str,default=None,help='checkpoint path during evaluation')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--gate_task_specific_dim', default=-1, type=int, help='gate task specific dims')

parser.add_argument('--regu_experts_fromtask',default=False, type=str2bool, help='whether use task id to guide expert selection')
parser.add_argument('--num_experts_pertask',default=-1, type=int)

parser.add_argument('--gate_input_ahead',default=False, type=str2bool, help='whether make gate input different from token')

parser.add_argument('--regu_sem',default=False, type=str2bool, help='whether use segmentation map to guide expert selection')
parser.add_argument('--semregu_loss_weight', default=0.01, type=float)

parser.add_argument('--sem_force',default=False, type=str2bool, help='whether use segmentation map to guide expert selection')
parser.add_argument('--warmup_epochs',default=5, type=int, help='whether need warmup train expert')

parser.add_argument('--epochs',default=None, type=int, help='number of train epochs')

parser.add_argument('--regu_subimage',default=False, type=str2bool, help='whether use subimage regulation for expert selection')
parser.add_argument('--subimageregu_weight', default=0.01, type=float)

parser.add_argument('--multi_level',default=None, type=str2bool, help='whether use multi level loss')

parser.add_argument('--opt', default=None, type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
parser.add_argument('--weight_decay', type=float, default=0.0001,help='weight decay (default: 0.05)')

parser.add_argument('--expert_prune',default=False, type=str2bool, help='whether use expert pruning')

parser.add_argument('--tam_level0',default=None, type=str2bool, help='use tamlevel0 to boost training')
parser.add_argument('--tam_level1',default=None, type=str2bool, help='use tamlevel1 to boost training')
parser.add_argument('--tam_level2',default=None, type=str2bool, help='use tamlevel2 to boost training')

parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--time', action='store_true', help='if wanna get inference time')
parser.add_argument('--visualize_features', action='store_true', 
                   help='whether to visualize intermediate features')
parser.add_argument('--viz_dir', type=str, default='feature_visualizations',
                   help='directory to save feature visualizations')
parser.add_argument('--diversity_loss_weight', default=0.0, type=float,
                   help='Weight for expert diversity loss (0.0 to disable)')
parser.add_argument('--train_accumulation_steps', default=1, type=int, help='Number of steps to accumulate gradients')

args = parser.parse_args()

if args.task_one_hot:
    args.one_by_one = True
# print('os.environ["LOCAL_RANK"]',os.environ["LOCAL_RANK"],args.local_rank)
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    # print(os.environ["LOCAL_RANK"])

def main():

    wandb.init(project='m3vit_diss')
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp, local_rank=args.local_rank, args=args)
    args.num_tasks = len(p.TASKS.NAMES)
    p['multi_gate'] = args.multi_gate
    if args.tam_level0 is not None:
        p['model_kwargs']['tam_level0']=args.tam_level0
    if args.tam_level1 is not None:
        p['model_kwargs']['tam_level1']=args.tam_level1
    if args.tam_level2 is not None:
        p['model_kwargs']['tam_level2']=args.tam_level2
    if args.lr is not None:
        p['optimizer_kwargs']['lr'] = args.lr
    if args.opt is not None:
        p['optimizer'] == args.opt
    if args.weight_decay is not None:
        p['optimizer_kwargs']['weight_decay'] = args.weight_decay
    if args.epochs is not None:
        p['epochs'] = args.epochs
    if args.backbone_random_init is not None:
        p['backbone_kwargs']['random_init']=args.backbone_random_init
    if args.moe_mlp_ratio is not None:
        p['backbone_kwargs']['moe_mlp_ratio']=args.moe_mlp_ratio
    if args.moe_top_k is not None:
        p['backbone_kwargs']['moe_top_k']=args.moe_top_k
    if args.trBatch is not None:
        p['trBatch'] = args.trBatch
    if args.valBatch is not None:
        p['valBatch'] = args.valBatch
    if args.multi_level is not None:
        p['multi_level'] = args.multi_level
    
    args.distributed = False
    if args.local_rank >=0:
        args.distributed = True
        print(os.environ["WORLD_SIZE"])
        print('args.local_rank',args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"])
    if args.local_rank >=0:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'),local_rank=args.local_rank)
    if args.distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            torch.distributed.barrier()
            p['local_rank'] = args.local_rank
        elif args.launcher == "slurm":
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            p['gpus'] = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                # 29500 is torch.distributed default port
                os.environ["MASTER_PORT"] = "29501"
            # use MASTER_ADDR in the environment variable if it already exists
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            p['local_rank'] = int(os.environ["LOCAL_RANK"])

        p['gpus'] = dist.get_world_size()
    else:
        p['local_rank'] = args.local_rank 
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    print(colored(p, 'red'))
    print("Distributed training: {}".format(args.distributed))
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(str(args))
    if args.distributed:
        args.rank = torch.distributed.get_rank()
    
    print(colored('Retrieve model', 'blue'))
    args.moe_use_gate = (args.moe_gate_arch != "")
    model = get_model(p,args)
    
    if not torch.cuda.is_available():
        raise NotImplementedError()
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                print('Use fast moe distributed learning==================>>')
                model = fmoe.DistributedGroupedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True,)
                sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
        else:
            model.cuda()
            if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.local_rank is not None:
        model.cuda()
    else:
        raise NotImplementedError()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda(args.local_rank)
    print(criterion)

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model, args)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 

    train_dataloader = build_train_dataloader(
        train_dataset, p['trBatch'], p['nworkers'], dist=args.distributed, shuffle=True)
    val_dataloader = build_val_dataloader(
        val_dataset, p['valBatch'], p['nworkers'], dist=args.distributed)

    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)

    if args.flops:
        for ii, sample in enumerate(val_dataloader):
            inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
            assert inputs.size(0)==1
            flops, params = profile(model, inputs=(inputs, ),)
            flops, params = clever_format([flops, params], "%.3f")
            print(flops,params)
            exit()

    if args.eval:
        if os.path.isdir(args.ckp):
            print("=> loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(os.path.join(args.ckp, "0.pth".format(torch.distributed.get_rank())),
                                    map_location="cpu")
            len_save = len([f for f in os.listdir(args.ckp) if "pth" in f])
            assert len_save % torch.distributed.get_world_size() == 0
            response_cnt = [i for i in range(
                torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.ckp, "{}.pth".format(cnt_model)),
                                                    map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                    dim=0)
                    else:
                        checkpoint["model_state"].update(checkpoint_specific["stmodel_stateate_dict"])
                moe_dir_read = True
        else:
            print("=> loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp, map_location='cpu')
        # state_dict = checkpoint['model_state']
        # model = cvt_state_dict_(state_dict, model,args, linear_keyword, moe_dir_read)
        # msg = model.load_state_dict(state_dict, strict=False)
        # print('=================model unmatched keys:================',msg)
        # save_model_predictions(p, val_dataloader, model, args)
        # if args.distributed:
        #     torch.distributed.barrier()
        # eval_stats = eval_all_results(p)

        model, optimizer, start_epoch = load_for_training(model, optimizer, args.ckp, 'cuda')


        if args.visualize_features and hasattr(model.module.backbone, 'visualize_features'):
            viz_save_dir = os.path.join(args.viz_dir, 'inference')
            viz_batch = next(iter(val_dataloader))
            viz_inputs = viz_batch['image'].cuda(non_blocking=True)
            
            with torch.no_grad():
                # First do normal forward pass
                _ = model(viz_inputs)
                
                # Then visualize with forced experts
                model.module.backbone.visualize_features(
                    save_dir=viz_save_dir,
                    input_image=viz_inputs,
                    expert_indices=range(16)  # Visualize all 16 experts
                )
                print(f'Saved feature visualizations to {viz_save_dir}')
                model.module.backbone.clear_intermediate_features()
        print('hello')
        exit()

    if args.resume:
        if os.path.isdir(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, "0.pth".format(torch.distributed.get_rank())),
                                    map_location="cpu")
            len_save = len([f for f in os.listdir(args.resume) if "pth" in f])
            assert len_save % torch.distributed.get_world_size() == 0
            response_cnt = [i for i in range(
                torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.resume, "{}.pth".format(cnt_model)),
                                                    map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                    dim=0)
                    else:
                        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                moe_dir_read = True
        else:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            for cnt, cnt_model in enumerate(response_cnt):
                print("=> loading checkpoint optimizer")
                if cnt_model != 0:
                    optimizer.load_state_dict(checkpoint_specific['optimizer'])
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        if args.distributed:
            torch.distributed.barrier()
        best_result = {'multi_task_performance':-200}  
    
    # Main loop
    print(colored('Starting main loop', 'blue'))

    if p['backbone'] == 'VisionTransformer_moe':
        using_vision_transformer = True

    rank = torch.distributed.get_rank()

    test_ckpt_path = os.path.join("/app/saved_stuff", "{}.pth".format(rank))

    device = torch.device(f"cuda:{args.local_rank}")
    local_rank = torch.distributed.get_rank()

    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if rank == 1:
        wandb.watch(model, log='all', log_freq=100)


    # SAVE
    # save_checkpoint(model, optimizer, 0, "/app/checkpoint.pt")

    # # LOAD for training
    # model, optimizer, start_epoch = load_for_training(model, optimizer, "checkpoint.pt", device)

    factorise_model(p, val_dataset, model, n=1, distributed=args.distributed)

    # this takes way too long so limit to 400 samples
    import random
    subset_size = 200
    val_dataset = val_loader.dataset
    random_indices = random.sample(range(len(val_dataset)), subset_size)
    subset = torch.utils.data.Subset(val_dataset, random_indices)
    val_loader = torch.utils.data.DataLoader(subset, batch_size=val_loader.batch_size, shuffle=False)

    save_model_predictions(p, val_dataloader, model, args)
    if args.distributed:
        torch.distributed.barrier()
    curr_result = eval_all_results(p)
    

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        # factorise_model(p, val_dataset, model, n=1, distributed=args.distributed)
        eval_train = train_vanilla_distributed(args, p, train_dataloader, model, criterion, optimizer, epoch)

        save_checkpoint(model, optimizer, 0, f"/app/checkpoint_{epoch}_{date_string}.pt")


        # Evaluate
            # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > p['epochs']-10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True


        eval_bool = True # hardcoding for now

        
        
        if eval_bool:
            print('Evaluate ...')
            
            # If visualization is requested
            if args.visualize_features and hasattr(model.module, 'visualize_features'):
                # Create visualization directory with epoch number
                viz_save_dir = os.path.join(args.viz_dir, f'epoch_{epoch}')
                
                # Get one batch for visualization
                viz_batch = next(iter(val_dataloader))
                viz_inputs = viz_batch['image'].cuda(non_blocking=True)
                
                # Forward pass to get intermediate features
                with torch.no_grad():
                    _ = model(viz_inputs)
                    
                    # Pass the input image to the visualization function
                    model.module.backbone.visualize_features(
                        save_dir=viz_save_dir,
                        input_image=viz_inputs
                    )
                    print(f'Saved feature visualizations to {viz_save_dir}')
                    
                    # Clear features to free memory
                    model.module.clear_intermediate_features()
            
            # Continue with normal evaluation
            save_model_predictions(p, val_dataloader, model, args)
            if args.distributed:
                torch.distributed.barrier()
            curr_result = eval_all_results(p)
            improves, best_result = validate_results_v2(p, curr_result, best_result)

        # factorise_model(p, val_dataset, model, n=1, distributed=args.distributed)
            
        
        # if args.distributed:
        #     torch.distributed.barrier()
    print('done!')
    torch.cuda.empty_cache()   
    # Evaluate best model at the end
    # if p['backbone'] == 'VisionTransformer_moe' and (not args.moe_data_distributed):
    #     # state_dict = read_specific_group_experts(checkpoint['state_dict'], args.local_rank, args.moe_experts)
    #     checkpoint_specific = torch.load(os.path.join(p['best_model'], "{}.pth".format(torch.distributed.get_rank())), map_location="cpu")
    #     checkpoint = torch.load(os.path.join(p['best_model'], "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
    #     checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
    #     state_dict = checkpoint["state_dict"]
    # else:
    #     # if args.local_rank==0:
    #     print(colored('Evaluating best model at the end', 'blue'))
    #     state_dict = torch.load(p['best_model'])['state_dict']
    # if args.distributed:
    #     torch.distributed.barrier()
    # model.load_state_dict(state_dict)
    # save_model_predictions(p, val_dataloader, model, args)
    # if args.distributed:
    #     torch.distributed.barrier()
    # eval_stats = eval_all_results(p)


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

if __name__ == "__main__":
    rank = args.local_rank
    # for writing the errors to a file
    try:
        main()
    except Exception as e:
        with open(f'error_log_{rank}.txt', 'a') as f:
            f.write(str(e))
        raise e
