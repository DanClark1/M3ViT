import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
# from torch._six import container_abcs
import collections.abc
import warnings
from collections import OrderedDict
from utils.helpers import load_pretrained,load_pretrained_pos_emb
from models.custom_moe_layer import FMoETransformerMLP
# from .layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers  import lecun_normal_
# from ..builder import BACKBONES
import numpy as np
from collections import Counter
from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from utils.perpca import PerPCA
from tqdm import tqdm

from sklearn.decomposition import PCA


import os
import matplotlib.pyplot as plt
import torch.nn.functional as F



a=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


def perpca_reconstruction_error(clients, U, V_list):
        """
        Compute the overall reconstruction error for PerPCA.
        
        Args:
            clients (list of torch.Tensor): Each element is a client's data matrix of shape (d, n_i).
            U (torch.Tensor): Global PC matrix of shape (d, r1).
            V_list (list of torch.Tensor): List of local PC matrices (one per client) of shape (d, r2_i).
        
        Returns:
            total_error (float): Average reconstruction error over clients.
        """
        total_error = 0.0
        clients = torch.stack(clients)
        clients = clients.cpu()
        U = U.cpu()
        V_list = [torch.tensor(V, device='cpu') for V in V_list]
        for client_data, V in zip(clients, V_list):
            # Reconstruction for this client
            reconstruction = U @ (U.T @ client_data) + V @ (V.T @ client_data)
            error = torch.norm(client_data - reconstruction, p='fro') ** 2
            total_error += error.item()
        return total_error / len(clients)

def reconstruction_error(X, U):
        """
        Compute the reconstruction error for standard PCA.
        
        Args:
            X (torch.Tensor): Data matrix of shape (d, n) where each column is a data sample.
            U (torch.Tensor): Principal component matrix of shape (d, k) (assumed to be orthonormal).
        
        Returns:
            error (float): The reconstruction error as the Frobenius norm squared.
        """
        # Reconstruct the data from the top k components
        X_hat = torch.tensor(U @ (U.T @ X), device='cpu')
        X = torch.tensor(X, device='cpu')
        # Compute the Frobenius norm squared of the difference
        error = torch.norm(X - X_hat, p='fro') ** 2
        return error.item()


default_cfgs = {
    # patch models
    'vit_tiny_patch16_224': _cfg(
        url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    ),
    'vit_small_patch16_224': _cfg(
        url = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        # url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth',
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
    'deit_base_distilled_path16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, checkpoint=True,
    ),
}


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class new_Mlp(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        if qk_scale is None:
            self.scale = head_dim ** -0.5
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print('for attention',x.shape)
        # print(self.scale)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moe=False, moe_mlp_ratio=-1, moe_experts=64,
                 moe_top_k=2, moe_gate_dim=-1, world_size=1, gate_return_decoupled_activation=False,
                 moe_gate_type="noisy", vmoe_noisy_std=1, gate_task_specific_dim=-1, multi_gate=False, 
                 regu_experts_fromtask = False, num_experts_pertask = -1, num_tasks = -1,
                 gate_input_ahead = False,regu_sem=False,sem_force=False,regu_subimage=False,expert_prune=False):
        super().__init__()
        self.moe = moe
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if 
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gate_input_ahead = gate_input_ahead
        self.expert_prune = expert_prune
        if moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(drop)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            if moe_gate_type == "noisy":
                moe_gate_fun = NoisyGate
            elif moe_gate_type == "noisy_vmoe":
                moe_gate_fun = NoisyGate_VMoE
            else:
                raise ValueError("unknow gate type of {}".format(moe_gate_type))

            print('mlp')
            self.mlp = FMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std, 
                                          gate_task_specific_dim=gate_task_specific_dim,multi_gate=multi_gate,
                                          regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                                          regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune)
            self.mlp_drop = nn.Dropout(drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, sem=None, record_expert_outputs=False, verbose=False):
        if self.gate_input_ahead:
            gate_inp = x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if not self.moe:
            return x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            moe_out = self.mlp(self.norm2(x), gate_inp, task_id, task_specific_feature, sem, record_expert_outputs=record_expert_outputs, verbose=verbose)
            return x + moe_out, moe_out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformerMoE(nn.Module):
    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                    num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None,  representation_size=None, distilled=False, 
                    drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                    pos_embed_interp=False, random_init=False, align_corners=False,
                    act_layer=None, weight_init='', moe_mlp_ratio=-1, moe_experts=64, moe_top_k=2, world_size=1, gate_dim=-1,
                    gate_return_decoupled_activation=False, moe_gate_type="noisy", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=False,
                    regu_experts_fromtask = False, num_experts_pertask = -1, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                    expert_prune=False, **kwargs):
        super(VisionTransformerMoE, self).__init__(**kwargs)
        # print(hybrid_backbone is None)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.h = int(self.img_size[0]/self.patch_size)
        self.w = int(self.img_size[1]/self.patch_size)

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        self.num_token = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.moe_experts = moe_experts
        self.moe_top_k = moe_top_k
        self.gate_return_decoupled_activation = gate_return_decoupled_activation
        self.multi_gate = multi_gate
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        # print(self.hybrid_backbone is None)
        self.expert_prune = expert_prune
        self.moe_block_index = {} # dicitonary mapping the nth moe layer to the actual layer index
        print('set expert prune as ',self.expert_prune)
        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        blocks = []
        self.num_tasks = gate_dim-embed_dim
        self.gate_task_specific_dim = gate_task_specific_dim
        self.gate_input_ahead = gate_input_ahead
        if self.gate_task_specific_dim<0 or self.multi_gate:
            self.gate_task_represent = None
        else:
            self.gate_task_represent = new_Mlp(in_features=self.num_tasks, hidden_features=int(self.gate_task_specific_dim), out_features=self.gate_task_specific_dim,)
            # self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        for i in range(self.depth):
            if i % 2 == 0:
                blocks.append(Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer))
            else:
                blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                              moe=True, moe_mlp_ratio=moe_mlp_ratio, moe_experts=moe_experts, moe_top_k=moe_top_k, moe_gate_dim=gate_dim, world_size=world_size,
                              gate_return_decoupled_activation=self.gate_return_decoupled_activation,
                              moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std, 
                              gate_task_specific_dim=self.gate_task_specific_dim,multi_gate=self.multi_gate,
                              regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                              gate_input_ahead = self.gate_input_ahead,regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune))
        self.blocks = nn.Sequential(*blocks)
        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)

        self.init_weights()
        self.idx = 0

    def factorise_model(self):
        for block in self.blocks:
            if block.moe:
                block.mlp.factorise_block()

    def dump_output(self):
        for block in self.blocks:
            if block.moe:
                block.mlp.dump_output()
 
    def init_weights(self, pretrained=None):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.default_cfg = default_cfgs[self.model_name]
        load_pretrained_pos_emb(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
        num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, img_h=self.h, img_w=self.w)
        if not self.random_init:
            self.default_cfg = default_cfgs[self.model_name]
            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, filter_fn=self._conv_filter, img_h=self.h, img_w=self.w)
            else:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, img_h=self.h, img_w=self.w)
        else:
            print('Initialize weight randomly')


    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def to_1D(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1).transpose(1, 2)
        return x

    def get_groundtruth_sem(self, sem):
        batch = sem.shape[0]
        hint = np.ones((batch,1,int(sem.shape[2]/self.patch_size),int(sem.shape[3]/self.patch_size)))*255
        idx = 0
        for k in range(batch):
            for i in range(int(sem.shape[2]/self.patch_size)):
                for j in range(int(sem.shape[3]/self.patch_size)):
                    patch = sem[k][:,self.patch_size*i:self.patch_size*(i+1),self.patch_size*j:self.patch_size*(j+1)].cpu().numpy().flatten()
                    index , num=Counter(patch).most_common(1)[0]
                    if num>0.4*(self.patch_size*self.patch_size):
                        hint[k,:,i,j]=index
                        if index != 255:
                            idx = idx+1
        filename = 'gt_patch_{}.npy'.format(self.idx)
        self.idx=self.idx+1
        np.save(filename, hint)
        return torch.tensor(hint, device=sem.device) 

    def forward_features(self, x, gate_inp, task_id, sem, isval=False, verbose=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        task_specific_feature = None
        if (task_id is not None) and (self.gate_task_represent is not None):
            task_specific = torch.zeros(self.num_tasks,device=x.device)
            task_specific[task_id]=1.0
            task_specific_feature = self.gate_task_represent(task_specific)
        
        # Store intermediate features
        intermediate_features = []
        # intermediate_features.append(x)  # Store input features
        
        outs = []
        printed = False
        moe_index = 0
        for i, blk in enumerate(self.blocks):
            if blk.moe:
                self.moe_block_index[moe_index] = i
                moe_index += 1
                x, intermediate_x = blk(x, gate_inp, task_id, task_specific_feature, sem=sem, record_expert_outputs=isval, verbose=((not printed) and verbose))
                if not printed:
                    printed = True
                intermediate_features.append(intermediate_x)
            else:
                x = blk(x)
              # Store features after each block
            if i in self.out_indices:
                outs.append(x)
            
        # Store the intermediate features as a class attribute
        self.intermediate_features = intermediate_features
        
        return tuple(outs)

    def get_intermediate_features(self):
        """Returns the stored intermediate features if they exist"""
        if hasattr(self, 'intermediate_features'):
            return self.intermediate_features
        else:
            raise AttributeError("No intermediate features found. Run a forward pass first.")

    def clear_intermediate_features(self):
        """Clears stored intermediate features to free memory"""
        if hasattr(self, 'intermediate_features'):
            del self.intermediate_features

    def forward(self, x, gate_inp=None, task_id=None, sem=None, isval=False, verbose=False):
        if sem is not None and (self.regu_sem or self.sem_force):
            sem = self.get_groundtruth_sem(sem)
        out = self.forward_features(x, gate_inp, task_id=task_id, sem=sem, isval=isval, verbose=verbose)
        return out
    
    def compute_misalignment(self, V_list):
        """
        Computes the misalignment parameter (θ) from a list of local PC matrices.
        
        Args:
            V_list (list of torch.Tensor): List of local PC matrices, each of shape (d, r),
                where each matrix is orthonormal (i.e. V.T @ V = I).
                
        Returns:
            theta (float): The misalignment parameter defined as θ = 1 - lambda_max,
                where lambda_max is the largest eigenvalue of the average projection matrix.
            lambda_max (float): The largest eigenvalue of the average projection matrix.
        """
        # Assume at least one local PC matrix exists.
        d = V_list[0].shape[0]
        device = V_list[0].device
        N = len(V_list)
        
        # Compute the average projection matrix: P_avg = (1/N) * sum(V @ V.T)
        P_avg = torch.zeros(d, d, device=device)
        for V in V_list:
            P = V @ V.T
            P_avg += P
            print(f'')
        P_avg /= N

        # Compute eigenvalues of the average projection matrix.
        # torch.linalg.eigvalsh returns sorted eigenvalues in ascending order.
        eigvals = torch.linalg.eigvalsh(P_avg)
        lambda_max = eigvals[-1].item()
        
        # Misalignment parameter: theta = 1 - lambda_max
        theta = 1 - lambda_max
        
        return theta, lambda_max

    def visualize_features(self, save_dir='feature_viz', layer_indices=None, input_image=None, expert_indices=None):
        """
        Visualizes intermediate features as heatmaps and saves them to disk.
        Also analyzes expert specialization using PerPCA.
        """

        if not hasattr(self, 'intermediate_features'):
            raise AttributeError("No intermediate features found. Run a forward pass first.")
                
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save input image if provided
        if input_image is not None:
            for b in range(input_image.shape[0]):
                img = input_image[b].cpu().detach()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                img = img * std + mean
                img = img.clamp(0, 1)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(img.permute(1, 2, 0))
                plt.title(f'Input Image - Sample {b}')
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f'input_image_sample_{b}.png'))
                plt.close()
        
        if layer_indices is None:
            layer_indices = range(len(self.intermediate_features))
        
        if expert_indices is not None:
            # Create results dictionary to store component analysis
            results = {
                'layer_results': {},
                'expert_results': {}
            }

            prev_features = None
            features_list = []
            
            # Dictionary to store features for each expert
            expert_features = {}
            expert_datasets = {}  # Store 100xd matrices for each expert

            for expert_idx in expert_indices:
                # Set forced expert for all MoE layers
                for block in self.blocks:
                    if hasattr(block, 'moe') and block.moe:
                        block.mlp.set_forced_expert(expert_idx)
                
                # Run forward pass with forced expert multiple times to build dataset
                expert_data = {idx: [] for idx in layer_indices}
                for _ in range(40):  # Run 10 times to get 100 samples (assuming batch_size=10)
                    with torch.no_grad():
                        _ = self.forward(input_image)
                        for idx in layer_indices:
                            features = self.intermediate_features[idx].cpu().detach()
                            features_list.append(features)
                            # Flatten features for each sample
                            flat_features = features.reshape(-1, features.shape[-1])
                            expert_data[idx].append(flat_features)
                
                # Store features and datasets
                expert_features[expert_idx] = {
                    idx: self.intermediate_features[idx].cpu().detach()
                    for idx in layer_indices
                }
                expert_datasets[expert_idx] = {
                    idx: torch.cat(expert_data[idx], dim=0)[:1000].T  #  transpose to dxn
                    for idx in layer_indices
                }
                
                # Visualize features for this expert
                for idx in layer_indices:
                    features = self.intermediate_features[idx]
                    B, N, D = features.shape
                    
                    patch_features = features[:, 1:, :]
                    h = w = int(math.sqrt(N - 1))
                    
                    feature_magnitudes = torch.norm(patch_features, dim=-1)
                    feature_magnitudes = feature_magnitudes.reshape(B, h, w)
                    
                    for b in range(B):
                        plt.figure(figsize=(10, 10))
                        plt.imshow(feature_magnitudes[b].cpu().detach(), cmap='viridis')
                        plt.colorbar()
                        plt.title(f'Layer {idx} - Sample {b} - Expert {expert_idx}')
                        plt.savefig(os.path.join(save_dir, f'layer_{idx}_sample_{b}_expert_{expert_idx}.png'))
                        plt.close()

            # make sure the expert outputs aren't the same
            print(torch.allclose(expert_datasets[0][1], expert_datasets[1][1]))

            
            # Analyze expert specialization using PerPCA
            print("\nAnalyzing expert specialization using PerPCA:")
            for layer_idx in layer_indices:
                if self.blocks[self.moe_block_index[layer_idx]].moe:
                    print(f"\nMoE Layer {layer_idx}:")
                    layer_results = {}
                    
                    # Analyze global components first
                    clients = [expert_datasets[exp_idx][layer_idx] for exp_idx in expert_indices]


                    print('clients shape:',[client.shape for client in clients])

                    print(f'sample cosine similarity between clients: {F.cosine_similarity(clients[0], clients[1], dim=1).mean().item()}')    

                    # Stack expert outputs to create tensor of shape (num_experts, d, b)
                    clients_tensor = torch.stack([clients[e] for e in range(16)], dim=0)


                    if torch.isnan(clients_tensor).any():
                        raise ValueError(f"NaNs detected in clients_tensor after normalization.")
                
                    eps = 1e-6
                    # Adding eps for numerical stability if needed
                    Q, _ = torch.linalg.qr(clients_tensor + eps, mode='reduced')
                    # Q now has shape (num_experts, d, r) where r = min(d, b)

                    if torch.isnan(Q).any():
                        raise ValueError("NaNs detected in Q matrix after QR decomposition.")


                    projs = torch.matmul(Q, Q.transpose(1, 2))


                    # Compute the average projection across experts: shape (d, d)
                    avg_proj = torch.mean(projs, dim=0)
                    
                    # Compute the eigenvalues of the averaged projection (avg_proj is symmetric)
                    eigvals = torch.linalg.eigvalsh(avg_proj)
                    lambda_max = eigvals[-1]
                    
                    # Calculate theta for this layer
                    theta_layer = 1 - lambda_max

                    print(f'theta approximation for layer {layer_idx}: {theta_layer} (lambda_max: {lambda_max})')

                    import itertools
                    r_values = [0,1,2,3,4,5]
                    r_value_combinations = list(itertools.combinations(r_values, 2))
                    for combination in r_value_combinations:
                        pca_model = PerPCA(r1=combination[0], r2=combination[1])
                        U, V_list = pca_model.fit(clients)
                        # Compute misalignment
                        theta, lambda_max = self.compute_misalignment(V_list)
                        print(f'theta: {theta}, lambda_max: {lambda_max}, r1 = {combination[0]}, r2 = {combination[1]}')
                    exit()

                    # generate all subsets of clients
                    # import itertools
                    # clients_combinations = list(itertools.combinations(clients, 2))
                    # clients_combinations = [torch.cat(pair, dim=0) for pair in clients_combinations]


                    # for combination in tqdm(clients_combinations):
                    #     U, V_list = pca_model.fit(combination)
                    #     # Compute misalignment
                    #     theta, lambda_max = self.compute_misalignment(V_list)
                    #     print('Combination of clients:', combination)
                    #     print(f"Misalignment (theta) for layer {layer_idx}: {theta:.4f} (lambda_max: {lambda_max:.4f})")
                    


                    print('Computing reconstruction errors for global components...')
                    
                    U, V_list = pca_model.fit(clients)

                    # measure cosine similarity between each client
                    client_cosine_similarities = []
                    for i in range(len(clients)):
                        for j in range(i + 1, len(clients)):
                            cos_sim = F.cosine_similarity(clients[i], clients[j], dim=1)
                            client_cosine_similarities.append(cos_sim.mean().item())
                    avg_cosine_similarity = np.mean(client_cosine_similarities)
                    print(f"Average cosine similarity between clients: {avg_cosine_similarity}")

                    
                    # Now compute cumulative reconstruction error and variance using best components
                    reconstruction_errors = []
                    explained_vars = []
                    for n_components in tqdm(component_nums):
                        # Use the n best components
                        U_subset = U[:, :n_components]
                        # Compute reconstruction error
                        recon_error = perpca_reconstruction_error(clients, U_subset, 
                                                               [torch.zeros_like(U_subset) for _ in clients])
                        reconstruction_errors.append(recon_error)
                        # Compute explained variance
                        var = pca_model.compute_explained_variance(clients, U_subset).sum().item()
                        explained_vars.append(var)
                    
                    # Plot reconstruction error and variance curves
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                    
                    # Reconstruction error
                    ax1.plot(component_nums, reconstruction_errors, 'bo-')
                    ax1.set_xlabel('Number of Best Components')
                    ax1.set_ylabel('Reconstruction Error')
                    ax1.set_yscale('log')
                    ax1.set_title('Reconstruction Error vs Best Components')
                    ax1.grid(True)
                    
                    # Explained variance
                    ax2.plot(component_nums, explained_vars, 'ro-')
                    ax2.set_xlabel('Number of Best Components')
                    ax2.set_ylabel('Cumulative Explained Variance')
                    ax2.set_title('Explained Variance vs Best Components')
                    ax2.grid(True)
                    
                    plt.suptitle(f'Layer {layer_idx} - Global Component Analysis')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'layer_{layer_idx}_global_analysis.png'))
                    plt.close()
                    
                    # Find optimal number of components using reconstruction error
                    optimal_global = component_nums[np.argmin(explained_vars)]
                    min_recon_error = min(explained_vars)
                    
                    # Store results
                    layer_results['global_components'] = optimal_global
                    layer_results['global_recon_error'] = min_recon_error
                    print(f"Optimal global components: {optimal_global} (recon error: {min_recon_error:.3f})")
                    
                    # Do the same for local components
                    expert_results = {}
                    for exp_idx in expert_indices:
                        expert_data = expert_datasets[exp_idx][layer_idx]
                        V = V_list[exp_idx]
                        
                        # Compute contribution of each local component
                        local_component_errors = []
                        local_component_vars = []
                        for j in range(V.shape[1]):
                            V_j = V[:, j:j+1]
                            recon_error = perpca_reconstruction_error([expert_data], 
                                                                   torch.zeros_like(V_j), 
                                                                   [V_j])
                            local_component_errors.append((j, recon_error))
                            var = pca_model.compute_explained_variance([expert_data], V_j).sum().item()
                            local_component_vars.append((j, var))
                        
                        # Sort local components
                        sorted_local = sorted(local_component_errors, key=lambda x: x[1])
                        best_local_indices = [idx for idx, _ in sorted_local]
                        
                        # Compute cumulative reconstruction error and variance
                        local_recon_errors = []
                        local_explained_vars = []
                        for n_components in component_nums:
                            best_indices = best_local_indices[:n_components]
                            V_best = V[:, best_indices]
                            recon_error = perpca_reconstruction_error([expert_data], 
                                                                   torch.zeros_like(V_best), 
                                                                   [V_best])
                            local_recon_errors.append(recon_error)
                            var = pca_model.compute_explained_variance([expert_data], V_best).sum().item()
                            local_explained_vars.append(var)
                        
                        # Plot local reconstruction error and variance curves
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                        
                        # Reconstruction error
                        ax1.plot(component_nums, local_recon_errors, 'bo-')
                        ax1.set_xlabel('Number of Best Components')
                        ax1.set_ylabel('Reconstruction Error')
                        ax1.set_yscale('log')
                        ax1.set_title('Reconstruction Error vs Best Components')
                        ax1.grid(True)
                        
                        # Explained variance
                        ax2.plot(component_nums, local_explained_vars, 'ro-')
                        ax2.set_xlabel('Number of Best Components')
                        ax2.set_ylabel('Cumulative Explained Variance')
                        ax2.set_title('Explained Variance vs Best Components')
                        ax2.grid(True)
                        
                        plt.suptitle(f'Layer {layer_idx} - Expert {exp_idx} Local Component Analysis')
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, 
                                  f'layer_{layer_idx}_expert_{exp_idx}_local_analysis.png'))
                        plt.close()
                        
                        # Find optimal local components
                        optimal_local = component_nums[np.argmin(local_recon_errors)]
                        min_local_recon = min(local_recon_errors)
                        optimal_local_indices = best_local_indices[:optimal_local]
                        
                        expert_results[exp_idx] = {
                            'local_components': optimal_local,
                            'local_recon_error': min_local_recon,
                            'local_component_indices': optimal_local_indices
                        }
                        print(f"Expert {exp_idx} optimal local components: {optimal_local}")
                        print(f"Reconstruction error: {min_local_recon:.3f}")
                        print(f"Best local indices: {optimal_local_indices[:10]}...")  # Show first 10
                    
                    layer_results['expert_results'] = expert_results
                    results['layer_results'][layer_idx] = layer_results

            # Save results to file
            with open(os.path.join(save_dir, 'component_analysis.txt'), 'w') as f:
                f.write("Component Analysis Results\n")
                f.write("=========================\n\n")
                
                for layer_idx in layer_indices:
                    layer_results = results['layer_results'][layer_idx]
                    f.write(f"Layer {layer_idx}:\n")
                    f.write(f"  Global components: {layer_results['global_components']}\n")
                    f.write(f"  Global reconstruction error: {layer_results['global_recon_error']:.3f}\n")
                    f.write("\n  Expert-specific results:\n")
                    
                    for exp_idx, exp_results in layer_results['expert_results'].items():
                        f.write(f"    Expert {exp_idx}:\n")
                        f.write(f"      Local components: {exp_results['local_components']}\n")
                        f.write(f"      Local reconstruction error: {exp_results['local_recon_error']:.3f}\n")
                    f.write("\n")

            # After finding optimal components, visualize projections
            print("\nVisualizing feature projections...")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get optimal components and ensure they're on the right device
            U_opt = U[:, layer_results['global_component_indices']].to(device)
            V_opt_list = [v[:, exp_results['local_component_indices']].to(device) 
                         for i, v in enumerate(V_list)]
            
            # For each expert
            for exp_idx in expert_indices:
                expert_data = expert_datasets[exp_idx][layer_idx].to(device)
                
                # Project onto global components
                global_proj = (U_opt @ (U_opt.T @ expert_data)).T  # [n, d]
                
                # Project onto local components
                V_opt = V_opt_list[exp_idx]
                local_proj = (V_opt @ (V_opt.T @ expert_data)).T  # [n, d]
                
                # Original features
                orig_features = expert_data.T  # [n, d]
                
                # Move all tensors to CPU for visualization
                orig_features = orig_features.cpu()
                global_proj = global_proj.cpu()
                local_proj = local_proj.cpu()
                
                # Reshape all to image format for visualization
                B = orig_features.shape[0] // (h * w)  # Batch size
                orig_spatial = orig_features.reshape(B, h, w, -1)
                global_spatial = global_proj.reshape(B, h, w, -1)
                local_spatial = local_proj.reshape(B, h, w, -1)
                
                # Compute feature magnitudes
                orig_mag = torch.norm(orig_spatial, dim=-1)
                global_mag = torch.norm(global_spatial, dim=-1)
                local_mag = torch.norm(local_spatial, dim=-1)
                
                # Plot side by side for each sample in batch
                for b in range(min(B, 5)):  # Show first 5 samples
                    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
                    
                    # Original features
                    im0 = axes[0].imshow(orig_mag[b], cmap='viridis')
                    axes[0].set_title(f'Original Features\nExpert {exp_idx}')
                    plt.colorbar(im0, ax=axes[0])
                    
                    # Global component projection
                    im1 = axes[1].imshow(global_mag[b], cmap='viridis')
                    axes[1].set_title(f'Global Component Projection\n({layer_results["global_components"]} components)')
                    plt.colorbar(im1, ax=axes[1])
                    
                    # Local component projection
                    im2 = axes[2].imshow(local_mag[b], cmap='viridis')
                    axes[2].set_title(f'Local Component Projection\n({exp_results["local_components"]} components)')
                    plt.colorbar(im2, ax=axes[2])
                    
                    plt.suptitle(f'Layer {layer_idx} - Sample {b} - Expert {exp_idx} Projections')
                    plt.savefig(os.path.join(save_dir, 
                              f'layer_{layer_idx}_expert_{exp_idx}_sample_{b}_projections.png'))
                    plt.close()
                    
                # Also visualize individual component contributions
                num_components_to_show = min(16, max(layer_results['global_components'], exp_results['local_components']))
                
                # Global components
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))
                axes = axes.flatten()
                
                expert_data = expert_data.to(device)  # Ensure data is on correct device
                for j in range(min(16, layer_results['global_components'])):
                    # Project onto single component
                    single_proj = (U_opt[:, j:j+1] @ (U_opt[:, j:j+1].T @ expert_data)).T
                    single_spatial = single_proj.cpu().reshape(B, h, w, -1)
                    single_mag = torch.norm(single_spatial, dim=-1)
                    
                    im = axes[j].imshow(single_mag[0], cmap='viridis')
                    axes[j].set_title(f'Global Component {j}')
                    plt.colorbar(im, ax=axes[j])
                
                # Remove empty subplots
                for j in range(min(16, layer_results['global_components']), 16):
                    axes[j].remove()
                
                plt.suptitle(f'Layer {layer_idx} - Expert {exp_idx} - Global Component Contributions')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 
                              f'layer_{layer_idx}_expert_{exp_idx}_global_components.png'))
                plt.close()
                
                # Local components
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))
                axes = axes.flatten()
                
                for j in range(min(16, exp_results['local_components'])):
                    # Project onto single component
                    single_proj = (V_opt[:, j:j+1] @ (V_opt[:, j:j+1].T @ expert_data)).T
                    single_spatial = single_proj.cpu().reshape(B, h, w, -1)
                    single_mag = torch.norm(single_spatial, dim=-1)
                    
                    im = axes[j].imshow(single_mag[0], cmap='viridis')
                    axes[j].set_title(f'Local Component {j}')
                    plt.colorbar(im, ax=axes[j])
                
                # Remove empty subplots
                for j in range(min(16, exp_results['local_components']), 16):
                    axes[j].remove()
                
                plt.suptitle(f'Layer {layer_idx} - Expert {exp_idx} - Local Component Contributions')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 
                              f'layer_{layer_idx}_expert_{exp_idx}_local_components.png'))
                plt.close()

        else:
            # Original visualization code for normal routing
            for idx in layer_indices:
                features = self.intermediate_features[idx]
                B, N, D = features.shape
                
                patch_features = features[:, 1:, :]
                h = w = int(math.sqrt(N - 1))
                
                feature_magnitudes = torch.norm(patch_features, dim=-1)
                feature_magnitudes = feature_magnitudes.reshape(B, h, w)
                
                for b in range(B):
                    plt.figure(figsize=(10, 10))
                    plt.imshow(feature_magnitudes[b].cpu().detach(), cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Layer {idx} - Sample {b}')
                    plt.savefig(os.path.join(save_dir, f'layer_{idx}_sample_{b}.png'))
                    plt.close()

        print(f"Visualizations saved to {save_dir}")

def compare_perpca_vs_pca(clients, r1, r2):
    """
    Compare PerPCA global component explained variance with regular PCA on pooled data.
    
    Args:
        clients (list of numpy.ndarray): Each element is a client data matrix of shape (d, n_client),
            where d is the feature dimension.
        r1 (int): Number of global components for PerPCA.
        r2 (int): Number of local components for PerPCA.
    
    This function uses the PerPCA implementation (assumed to have methods fit and compute_explained_variance)
    and the sklearn PCA on the pooled data to generate a scree plot for both approaches.
    """
    

    

    # move clients to numpy
    # Instantiate PerPCA with the specified number of global (r1) and local (r2) components.
    pca_model = PerPCA(r1=r1, r2=r2)
    U, V_list = pca_model.fit(clients)
    
    # Determine maximum number of components from the PerPCA output.
    max_components = U.shape[1]
    component_nums = list(range(1, max_components + 1))
    explained_vars_perpca = []
    explained_vars_perpca_local = []
    
    # Compute the explained variance for the global components from PerPCA.
    for n_components in component_nums:
        U_subset = U[:, :n_components]
        explained_var = pca_model.compute_explained_variance(clients, U_subset)
        explained_vars_perpca.append(explained_var.sum().item())

    for n_components in component_nums:
        V_list_subset = [V[:, :n_components] for V in V_list]
        explained_var_per_client = []
        for i in range(len(clients)):
            explained_var_per_client.append(pca_model.compute_explained_variance(clients, V_list_subset[i]))
        explained_var = np.sum(np.array(explained_var_per_client)) / len(explained_var_per_client)
        explained_vars_perpca_local.append(explained_var.sum().item())
    

    cpu_clients = [client.cpu().numpy() for client in clients]

    # Pool the client data into one matrix.
    pooled_data = np.hstack(cpu_clients)  # Shape: (d, total_samples)
    pooled_data = pooled_data.T         # Reshape to (total_samples, d) for sklearn PCA
    
    # Run regular PCA on the pooled data.
    pca = PCA(n_components=max_components)
    pca.fit(pooled_data)
    explained_vars_pca = pca.explained_variance_ratio_

    # print reconstruction errors
    print('Reconstruction error for standard PCA:', reconstruction_error(pooled_data.T, pca.components_.T))
    print('Reconstruction error for PerPCA:', perpca_reconstruction_error(clients, U, V_list))
    
    # Plot the comparison between PerPCA global components and regular PCA.
    plt.figure(figsize=(10, 6))
    plt.plot(component_nums, explained_vars_perpca, 'bo-', label='PerPCA Global Components')
    plt.plot(component_nums, explained_vars_pca, 'ro-', label='Regular PCA')
    plt.plot(component_nums, explained_vars_perpca_local, 'go-', label='PerPCA Local Components')
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Comparison of PerPCA vs Regular PCA")
    plt.legend()
    plt.grid(True)
    plt.savefig('app/perpca_vs_pca_comparison.png')


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def find_elbow_point(component_nums, explained_vars):
    """
    Find the elbow point in the explained variance curve.
    
    Args:
        component_nums: List of number of components tried
        explained_vars: List of corresponding explained variance values
    
    Returns:
        optimal_components: Number of components at elbow point
        explained_var: Explained variance at elbow point
    """
    diffs = np.abs(np.diff(explained_vars, 2))
    elbow_idx = np.argmin(diffs) + 1
    return component_nums[elbow_idx], explained_vars[elbow_idx]


def find_threshold_point(component_nums, explained_vars, threshold=0.9):
    """
    Finds the minimum number of components that achieve a specified threshold of explained variance.
    
    Args:
        component_nums: List of number of components tried
        explained_vars: List of corresponding explained variance values
        threshold: Desired threshold for explained variance (default is 0.9)
        
    Returns:
        optimal_components: Number of components that achieve the threshold
        explained_var: Explained variance at that point
    """
    cumulative_vars = np.cumsum(explained_vars)
    optimal_idx = np.where(cumulative_vars >= threshold)[0][0]
    return component_nums[optimal_idx], cumulative_vars[optimal_idx]