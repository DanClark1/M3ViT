#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from audioop import mul
import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output, mkdir_if_missing
import numpy as np
from collections import Counter
import wandb

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p):
        self.database = p['train_db_name']
        self.tasks = p.TASKS.NAMES
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        # print('in performancemeter',pred.keys(),self.tasks)
        if len(pred.keys())<len(self.tasks):
            for t in pred.keys():
                self.meters[t].update(pred[t], gt[t])
        else:
            for t in self.tasks:
                self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert(set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)    
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]
        
        if task == 'depth': # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']: # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']

        elif task == 'normals': # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']

        elif task == 'edge': # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks



def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter()

    elif task == 'edge': # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'])

    else:
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = p.TASKS.NAMES
    
    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False
        
        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False
        
        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True
            
            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False


    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference

def validate_results_v2(p, current, reference):
    tasks = p.TASKS.NAMES

    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False
        
        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False
        
        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True
            
            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False


    else: # Multi-task performance
        multi_task_performance = calculate_multi_task_performance(current, reference)
        if multi_task_performance>0:
            print('New best multi-task model from previous model %.2f' %(100*multi_task_performance))
            improvement = True
        else:
            print('Current model fail to surpass last stage model')
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference




@torch.no_grad()
def eval_model(p, val_loader, model, step):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
        output = model(images)

        # Measure performance
        performance_meter.update({t: get_output(output[t], t) for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose = True)
    if torch.distrbuted.get_rank() == 0:
        wandb.log({'val semseg mean iou': eval_results['semseg']['mIoU']}, step=step, commit=False)
        wandb.log({'val human_parts mean iou': eval_results['human_parts']['mIoU']} , step=step, commit=False)
        wandb.log({'val normals mean error': eval_results['normals']['mean']}, step=step, commit=False)
        wandb.log({'val sal mean iou': eval_results['sal']['mIoU']}, step=step, commit=False)
        wandb.log({'val edge loss': eval_results['edge']['loss']}, step=step)
    return eval_results


@torch.no_grad()
def save_model_predictions(p, val_loader, model, args=None):
    """ Save model predictions for all tasks """

    print('Save model predictions to {}'.format(p['save_dir']))
    model.eval()
    print(p['backbone'])
    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    for ii, sample in enumerate(val_loader):      
        inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
        img_size = (inputs.size(2), inputs.size(3))

        if args is not None:
            output={}
            if args.one_by_one:
                id=0
                for single_task in p.TASKS.NAMES:
                    if args.task_one_hot:
                        output.update(model(inputs,single_task=single_task, task_id = id))
                        id=id+1
                    else:
                        output.update(model(inputs,single_task=single_task))
            else:
                if p['backbone']=='VisionTransformer_moe':
                    output = model(inputs, isval=True)
                else:
                    output = model(inputs)
        else:
            if p['backbone']=='VisionTransformer_moe':
                    output = model(inputs, isval=True)
            else:
                output = model(inputs)
        if ii%50==0:
            print('has saved samples',ii,len(val_loader))
        for task in p.TASKS.NAMES:
            output_task = get_output(output[task], task).cpu().data.numpy()
            for jj in range(int(inputs.size()[0])):
                if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
                    continue
                fname = meta['image'][jj]               
                result = cv2.resize(output_task[jj], dsize=(int(meta['im_size'][1][jj]),int(meta['im_size'][0][jj])), interpolation=p.TASKS.INFER_FLAGVALS[task])
                if task == 'depth':
                    sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
                else:
                    imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))


def eval_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir """
    save_dir = p['save_dir'] 
    results = {}
    print('p.TASKS.NAMES',p.TASKS.NAMES)
    # if 'edge' in p.TASKS.NAMES: 
    #     from evaluation.eval_edge import eval_edge_predictions
    #     results['edge'] = eval_edge_predictions(p, database=p['val_db_name'],
    #                          save_dir=save_dir)
    
    if 'semseg' in p.TASKS.NAMES:
        from evaluation.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['val_db_name'],
                              save_dir=save_dir, overfit=p.overfit)
    
    if 'human_parts' in p.TASKS.NAMES: 
        from evaluation.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['val_db_name'],
                                   save_dir=save_dir, overfit=p.overfit)

    if 'normals' in p.TASKS.NAMES:
        from evaluation.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['val_db_name'],
                               save_dir=save_dir, overfit=p.overfit)

    if 'sal' in p.TASKS.NAMES:
        from evaluation.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['val_db_name'],
                           save_dir=save_dir, overfit=p.overfit)

    if 'depth' in p.TASKS.NAMES:
        from evaluation.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['val_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        print('data set processed is ',p['train_db_name'])
        if p['train_db_name']=='NYUD':
            single_task_test_dict = {'depth':{'rmse':0.585},'semseg':{'mIoU':0.439}, 'normals':{'mean':19.763}}
        elif p['train_db_name']=='PASCALContext':
            single_task_test_dict = {'human_parts':{'mIoU':0.599},'semseg':{'mIoU':0.662}, 'normals':{'mean':13.9},'sal':{'mIoU':0.663}} #'edge':{'odsF':0.688},
        elif p['train_db_name']=='CityScapes':
            single_task_test_dict = {'depth':{'rmse':0.585},'semseg':{'mIoU':0.727}}
        # single_task_test_dict = {}
        # for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
        #     with open(test_dict, 'r') as f_:
        #          single_task_test_dict[task] = json.load(f_)
        for key in list(single_task_test_dict.keys()):
            if key not in p.TASKS.NAMES:
                single_task_test_dict.pop(key)
                
        if 'depth' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['depth']['rmse'])
        if 'semseg' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['semseg']['mIoU'])
        if 'normals' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['normals']['mean'])
        
        if 'human_parts' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['human_parts']['mIoU'])
        if 'edge' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['edge']['odsF'])
        if 'sal' in single_task_test_dict: # rmse lower is better
            print('single_task_test_dict: ',single_task_test_dict['sal']['mIoU'])

        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)            
        print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results
