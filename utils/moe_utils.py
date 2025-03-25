from collections import OrderedDict
from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.custom_moe_layer import FMoETransformerMLP
import torch.distributed
import os
from pdb import set_trace

import torch.nn.functional as F
import shutil


from fmoe.utils import scatter_model, gather_model

def save_full_checkpoint(model, path):
    # model is DistributedGroupedDataParallel
    full_state = gather_model(model)                # merges all expert shards
    if torch.distributed.get_rank() == 0:
        torch.save(full_state, path)
    torch.distributed.barrier()


def load_full_checkpoint(model, path, device):
    # Only rank0 loads full weights
    full_state = torch.load(path, map_location="cpu") if torch.distributed.get_rank()==0 else None

    # Broadcast full dict to every rank
    obj = [full_state]
    torch.distributed.broadcast_object_list(obj, src=0)
    full_state = obj[0]

    # model is DistributedGroupedDataParallel
    scatter_model(model, full_state)                # splits weights back onto each GPU
    torch.distributed.barrier()



def gather_features(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def save_consolidated_checkpoint(state, dirname):
    os.makedirs(dirname, exist_ok=True)
    rank = torch.distributed.get_rank()

    local_ckpt = {k: v.detach().cpu() for k, v in state["state_dict"].items()}
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_ckpt)

    if rank == 0:
        merged = {}
        for shard in gathered:
            for k, v in shard.items():
                if k not in merged:
                    merged[k] = v
                elif v.ndim >= 1 and merged[k].ndim >= 1:
                    merged[k] = torch.cat([merged[k], v], dim=0)
                # else: skip scalar stats
        torch.save(
            {"state_dict": merged, **{k: v for k, v in state.items() if k != "state_dict"}},
            os.path.join(dirname, "model.pth")
        )
        print(f"✅ Saved consolidated checkpoint to {dirname}/model.pth")

    torch.distributed.barrier()

def load_consolidated_checkpoint(model, path, device):
    rank = torch.distributed.get_rank()
    # Rank0 loads from disk, others start with None
    ckpt = torch.load(path, map_location="cpu")["state_dict"] if rank == 0 else None

    obj = [ckpt]
    torch.distributed.broadcast_object_list(obj, src=0)
    ckpt = obj[0]  # now every rank has the full dict

    aligned = align_state_dict_keys(model, ckpt)
    msg = model.load_state_dict(aligned, strict=False)
    if rank == 0:
        print("Loaded checkpoint:", len(msg.missing_keys), "missing,", len(msg.unexpected_keys), "unexpected")
    torch.distributed.barrier()
    return model



def align_state_dict_keys(model, ckpt_state):
        model_keys = next(iter(model.state_dict().keys()))
        ckpt_keys  = next(iter(ckpt_state.keys()))
        
        # If model expects “module.” but ckpt keys do NOT start with it → add it
        if model_keys.startswith("module.") and not ckpt_keys.startswith("module."):
            return {f"module.{k}": v for k, v in ckpt_state.items()}
        
        # If ckpt has “module.” but model does NOT → strip it
        if ckpt_keys.startswith("module.") and not model_keys.startswith("module."):
            return {k.replace("module.", "", 1): v for k, v in ckpt_state.items()}
        
        return ckpt_state


def collect_moe_model_state_dict(moe_state_dict):
    collect_moe_state_dict = OrderedDict()

    for key, item in moe_state_dict.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            collect_moe_state_dict[key] = gather_features(item, torch.distributed.get_rank(), torch.distributed.get_world_size())
        else:
            collect_moe_state_dict[key] = item

    return collect_moe_state_dict

def filter_state(state):
    from collections import OrderedDict
    new_state = OrderedDict()
    for key, item in state.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            new_state[key] = item
    return new_state

# def save_moe_model_to_dir(state, filename, save_dir):
#     rank = torch.distributed.get_rank()
#     dirname = os.path.join(save_dir, filename)
#     if rank == 0:
#         if os.path.isfile(dirname):
#             os.system("rm {}".format(dirname))
#         os.system("mkdir -p {}".format(dirname))
#     torch.distributed.barrier()

#     save_name = os.path.join(dirname, "{}.pth".format(rank))
#     if rank != 0:
#         state["state_dict"] = filter_state(state["state_dict"])
#     torch.save(state, save_name)

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_dir="checkpoints", moe_save=False, only_best=False):
#     if moe_save:
#         if "optimizer" in state:
#             del state["optimizer"]
#         if not only_best:
#             save_moe_model_to_dir(state, filename, save_dir)
#         if is_best:
#             save_moe_model_to_dir(state, "model_best.pth.tar", save_dir)
#     else:
#         if not only_best:
#             torch.save(state, os.path.join(save_dir, filename))
#         if is_best:
#             shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))

def save_moe_model_to_dir(state, dirname):
    dirname = "/app/saved_stuff"
    rank = torch.distributed.get_rank()
    if rank == 0:
        if os.path.isfile(dirname):
            os.system("rm {}".format(dirname))
        os.system("mkdir -p {}".format(dirname))
    torch.distributed.barrier()

    save_name = os.path.join(dirname, "{}.pth".format(rank))
    if rank != 0:
        state["state_dict"] = filter_state(state["state_dict"])
    torch.save(state, save_name) 
    print(f"Model saved to {dirname}")

def save_checkpoint(state, is_best, p, moe_save=False, only_best=False):
    if moe_save:
        # if "optimizer" in state:
        #     del state["optimizer"]
        if not only_best:
            save_moe_model_to_dir(state, p['checkpoint'])
        if is_best:
            save_moe_model_to_dir(state, p['best_model'])
    else:
        if not only_best:
            torch.save(state, p['checkpoint'])
        if is_best:
            shutil.copyfile(p['checkpoint'], p['best_model'])

def read_specific_group_experts(moe_state_dict, rank, num_experts):
    for key, item in moe_state_dict.items():
        if "mlp.experts.htoh4" in key or "mlp.experts.h4toh" in key:
            moe_state_dict[key] = item[rank * num_experts: (rank + 1) * num_experts]
        else:
            moe_state_dict[key] = item

    return moe_state_dict


def collect_noisy_gating_loss(model, weight):
    loss = 0
    for module in model.modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_loss:
            # print(module)
            loss += module.get_loss()
    return loss * weight


def collect_semregu_loss(model, weight):
    loss = 0
    for module in model.modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)):
            # print('during collection semregu loss',module.get_semregu_loss())
            loss += module.get_semregu_loss()
    return loss * weight

def collect_regu_subimage_loss(model, weight):
    loss = 0
    for module in model.modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)):
            # print('during collection semregu loss',module.get_semregu_loss())
            loss += module.get_regu_subimage_loss()
    return loss * weight
    
def collect_moe_activation(model, batch_size, activation_suppress="pool", return_name=False):
    gate_activations = []
    names = []
    for name, module in model.named_modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)) and module.has_activation:
            activation = module.get_activation()
            _, c = activation.shape
            activation = activation.reshape(batch_size, -1, c)
            if activation_suppress == "pool":
                activation = activation.mean(dim=1)
            elif activation_suppress == "concat":
                activation = torch.reshape(activation.shape[0], -1)
            elif activation_suppress == "origin":
                pass
            else:
                raise ValueError("No activation_suppress of {}".format(activation_suppress))
            gate_activations.append(activation)
            names.append(name)

    if not return_name:
        return gate_activations
    else:
        return gate_activations, names


def set_moe_mask(model, select_idx_dict):
    for name, module in model.named_modules():
        if (isinstance(module, NoisyGate) or isinstance(module, NoisyGate_VMoE)):
            module.select_idx = select_idx_dict[name]

class feature_avger(object):
    def __init__(self):
        self.avg = None
        self.cnt = 0

    def update(self, features):
        if self.avg is None:
            self.avg = features.mean(0)
        else:
            self.avg = self.avg         * (self.cnt          / (self.cnt + features.shape[0])) + \
                       features.mean(0) * (features.shape[0] / (self.cnt + features.shape[0]))

        self.cnt += features.shape[0]

def prune_moe_experts(model, train_loader, log, moe_experts_prune_num):
    model.train()

    for cnt, (image, label) in enumerate(train_loader):
        image = image.cuda(non_blocking=True)
        pred = model(image)
        gate_activations, gate_names = collect_moe_activation(model, pred.shape[0], return_name=True)

        if cnt == 0:
            gate_activations_avger_dict = {name: feature_avger() for name in gate_names}

        gate_activations = [F.softmax(g, dim=1) for g in gate_activations]
        for gate_name, gate_activation in zip(gate_names, gate_activations):
            gate_activations_avger_dict[gate_name].update(gate_activation.detach())

        if cnt % 100 == 0:
            log.info("prune gate stat cal: [{}/{}]".format(cnt, len(train_loader)))


    save_gate_activations_avger_dict = {k: item.avg for k, item in gate_activations_avger_dict.items()}
    torch.save(save_gate_activations_avger_dict, os.path.join(log.path, "save_gate_activations_avger_dict.pth"))

    # prune
    select_idx_dict = {}
    for n, item in save_gate_activations_avger_dict.items():
        assert moe_experts_prune_num <= len(item)
        top_logits, top_indices = item.topk(moe_experts_prune_num)
        select_idx_dict[n] = top_indices
    set_moe_mask(model, select_idx_dict)

    return gate_activations_avger_dict


def set_moe_layer_train_mode(model):
    for module in model.modules():
        if isinstance(module, FMoETransformerMLP):
            module.train()



def sync_weights(model, except_key_words):
    state_dict = model.state_dict()
    for key, item  in state_dict.items():
        flag_sync = True
        for key_word in except_key_words:
            if key_word in key:
                # print('key',key)
                flag_sync = False
                break

        if flag_sync:
            torch.distributed.broadcast(item, 0)

    model.load_state_dict(state_dict)
    return

