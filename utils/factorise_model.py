import torch
import random
from torch.utils.data import Subset
from utils.common_config import build_val_dataloader

@torch.no_grad()
def factorise_model(p, val_dataset, model, n=50, distributed=False):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    # tasks = p.TASKS.NAMES
    # subset_ratio = n / len(val_dataset)
    # val_indices = random.sample(range(len(val_dataset)), int(len(val_dataset) * subset_ratio))
    # val_dataset = Subset(val_dataset, val_indices)

    # val_dataloader = build_val_dataloader(
    #     val_dataset, p['valBatch'], p['nworkers'], dist=distributed)

    # model.eval()

    # print('recording expert outputs...')

    # for i, batch in enumerate(val_dataloader):
    #     # Forward pass
    #     images = batch['image'].cuda(non_blocking=True)
    #     print('images shape:', images.shape)
    #     targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
    #     output = model(images, isval=True)

    # model.module.dump_output_matricies()
    print('factorising model...')
    model.module.factorise_model()