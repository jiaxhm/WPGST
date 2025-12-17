# coding:utf-8
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
# from timm.models.layers.helpers import to_2tuple
from functools import partial
# from models.local_scan import *
from models.RandomConvs import RandomConv
import seaborn as sns
from models.SWA import *

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    Stage1: down + LayerNormGeneral
    stage2-5: LayerNormGeneral + down
    _R232_loss_random_conv_wadown_224_0423best
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x

# DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
#                                          kernel_size=7, stride=4, padding=2,
#                                          post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
#                                          )] + \
#                                 [partial(Downsampling,
#                                          kernel_size=3, stride=2, padding=1,
#                                          pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
#                                          )] * 3

class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=36, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        N = H * W
        x_local_scan = local_scan_bchw(x, 4, H=H, W=W, flip=flip, column_first=column_first)  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class Attention31(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        N = H * W
        x_local_scan = x.flatten(2)  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention32(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        N = H * W
        x_local_scan = x.flatten(2).flip([-1])  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention33(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        N = H * W
        x_local_scan = rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention34(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        N = H * W
        x_local_scan = rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention1(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=36, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        N = H * W
        x_local_scan = local_scan_bchw(x, 4, H=H, W=W, flip=flip, column_first=column_first)  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #--------------------------------
        # attn = attn.mean(dim=1)
        # attn = attn[0,0].detach().cpu().numpy()
        #
        # plt.figure(figsize=(8, 8))
        # sns.heatmap(attn, cmap='viridis',square=True, cbar=True)
        # plt.show()
#-----------------------------------
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention2(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.

    input； B, H, W, C
    output； B, H, W, C
    """

    def __init__(self, dim, head_dim=36, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, flip, column_first):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        N = H * W
        x_local_scan = local_scan_bchw(x, 2, H=H, W=W, flip=flip, column_first=column_first)  # B, C, L
        x_local_scan = x_local_scan.transpose(1, 2)  # B, L, C
        qkv = self.qkv(x_local_scan).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerNormGeneral(nn.Module):

    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=False, eps=1e-6):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.SiLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        # drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        # self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        # self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.SiLU(inplace=True),
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class LocalFormerBlock11(nn.Module):
    """
    Implementation of one MetaFormer block.
    self.res_scale1 and self.res_scale2 is used in MetaFormer blocks.
    self.layer_scale1 and self.layer_scale2 is not used in MetaFormer blocks.
    self.drop_path1 and self.drop_path2 is not used in MetaFormer blocks.
    """

    def __init__(self, dim, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., res_scale_init_value=True
                 ):
        super().__init__()

        num_channels_reduced = dim // 16
        self.fc1 = nn.Linear(dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, dim, bias=True)
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        # self.norm1 = norm_layer(dim)
        # self.attention1 = Attention1(dim=dim//4, drop=drop)
        # self.attention2 = Attention1(dim=dim//4, drop=drop)
        # self.attention3 = Attention1(dim=dim//4, drop=drop)
        # self.attention4 = Attention1(dim=dim//4, drop=drop)
        self.attention1 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2, shift_size=1))
        self.attention2 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4, shift_size=2))
        self.attention3 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6, shift_size=3))
        self.attention4 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8, shift_size=4))
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.randomconv = RandomConv(dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        xx = x.flatten(2).permute(0, 2, 1)
        xx = self.norm3(xx)
        z = xx.clone().permute(0, 2, 1).mean(dim=2)
        # z_max = xx.permute(0, 2, 1).max(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # x = rearrange(xx, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)
        #
        # flip1 = random.choice([True, False])
        # flip2 = random.choice([True, False])
        # flip3 = random.choice([True, False])
        # flip4 = random.choice([True, False])
        # column1 = random.choice([True, False])
        # column2 = random.choice([True, False])
        # column3 = random.choice([True, False])
        # column4 = random.choice([True, False])
        #
        # x_scan1 = self.attention1(x1, flip1, column1)
        # x_scan2 = self.attention2(x2, flip2, column2)
        # x_scan3 = self.attention3(x3, flip3, column3)
        # x_scan4 = self.attention4(x4, flip4, column4)
        # x_scan1_reverse = local_reverse(x_scan1.transpose(1, 2), 4, H, W, flip=flip1, column_first=column1)
        # x_scan2_reverse = local_reverse(x_scan2.transpose(1, 2), 4, H, W, flip=flip2, column_first=column2)
        # x_scan3_reverse = local_reverse(x_scan3.transpose(1, 2), 4, H, W, flip=flip3, column_first=column3)
        # x_scan4_reverse = local_reverse(x_scan4.transpose(1, 2), 4, H, W, flip=flip4, column_first=column4)

        # x = rearrange(xx, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x_scan1 = self.attention1(x1)
        x_scan2 = self.attention2(x2)
        x_scan3 = self.attention3(x3)
        x_scan4 = self.attention4(x4)
        x_scan_reverse = torch.cat([x_scan1, x_scan2, x_scan3, x_scan4], dim=1)
        # x_scan_reverse = torch.cat([x_scan1_reverse, x_scan2_reverse, x_scan3_reverse, x_scan4_reverse], dim=-1)
        # x_scan_reverse = x_scan_reverse.reshape(B, C, H, W)
        x_scan_reverse = self.randomconv(x_scan_reverse)
        x_scan_out = self.res_scale1(x.permute(0, 2, 3, 1)) + x_scan_reverse
        x_scan_out = rearrange(x_scan_out, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)
        x_scan_out_att = x_scan_out * fc_out_2.unsqueeze(1)
        x_scan_out_att = rearrange(x_scan_out_att, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x = self.res_scale2(x_scan_out_att) + self.mlp(self.norm2(x_scan_out_att))
        x = x.permute(0, 3, 1, 2)
        return x

class LocalFormerBlock22(nn.Module):
    """
    Implementation of one MetaFormer block.
    self.res_scale1 and self.res_scale2 is used in MetaFormer blocks.
    self.layer_scale1 and self.layer_scale2 is not used in MetaFormer blocks.
    self.drop_path1 and self.drop_path2 is not used in MetaFormer blocks.
    """

    def __init__(self, dim, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., res_scale_init_value=True
                 ):
        super().__init__()

        num_channels_reduced = dim // 16
        self.fc1 = nn.Linear(dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, dim, bias=True)
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        # self.norm1 = norm_layer(dim)
        # self.attention1 = Attention2(dim=dim//4, drop=drop)
        # self.attention2 = Attention2(dim=dim//4, drop=drop)
        # self.attention3 = Attention2(dim=dim//4, drop=drop)
        # self.attention4 = Attention2(dim=dim//4, drop=drop)
        self.attention1 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2, shift_size=1))
        self.attention2 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4, shift_size=2))
        self.attention3 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6, shift_size=3))
        self.attention4 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8, shift_size=4))

        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.randomconv = RandomConv(288)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        xx = x.flatten(2).permute(0, 2, 1)
        xx = self.norm3(xx)
        z = xx.clone().permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # x = rearrange(xx, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        # flip1 = random.choice([True, False])
        # flip2 = random.choice([True, False])
        # flip3 = random.choice([True, False])
        # flip4 = random.choice([True, False])
        # column1 = random.choice([True, False])
        # column2 = random.choice([True, False])
        # column3 = random.choice([True, False])
        # column4 = random.choice([True, False])
        #
        # x_scan1 = self.attention1(x1, flip1, column1)
        # x_scan2 = self.attention2(x2, flip2, column2)
        # x_scan3 = self.attention3(x3, flip3, column3)
        # x_scan4 = self.attention4(x4, flip4, column4)
        # x_scan1_reverse = local_reverse(x_scan1.transpose(1, 2), 2, H, W, flip=flip1, column_first=column1)
        # x_scan2_reverse = local_reverse(x_scan2.transpose(1, 2), 2, H, W, flip=flip2, column_first=column2)
        # x_scan3_reverse = local_reverse(x_scan3.transpose(1, 2), 2, H, W, flip=flip3, column_first=column3)
        # x_scan4_reverse = local_reverse(x_scan4.transpose(1, 2), 2, H, W, flip=flip4, column_first=column4)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x_scan1 = self.attention1(x1)
        x_scan2 = self.attention2(x2)
        x_scan3 = self.attention3(x3)
        x_scan4 = self.attention4(x4)
        x_scan_reverse = torch.cat([x_scan1, x_scan2, x_scan3, x_scan4], dim=1)
        # x_scan_reverse = torch.cat([x_scan1_reverse, x_scan2_reverse, x_scan3_reverse, x_scan4_reverse], dim=1)
        # x_scan_reverse = x_scan_reverse.reshape(B, C, H, W)
        x_scan_reverse = self.randomconv(x_scan_reverse)
        x_scan_out = self.res_scale1(x.permute(0, 2, 3, 1)) + x_scan_reverse
        x_scan_out = rearrange(x_scan_out, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)
        x_scan_out_att = x_scan_out * fc_out_2.unsqueeze(1)
        x_scan_out_att = rearrange(x_scan_out_att, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x = self.res_scale2(x_scan_out_att) + self.mlp(self.norm2(x_scan_out_att))
        x = x.permute(0, 3, 1, 2)
        return x

class LocalFormerBlock33(nn.Module):
    """
    Implementation of one MetaFormer block.
    self.res_scale1 and self.res_scale2 is used in MetaFormer blocks.
    self.layer_scale1 and self.layer_scale2 is not used in MetaFormer blocks.
    self.drop_path1 and self.drop_path2 is not used in MetaFormer blocks.
    """

    def __init__(self, dim, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., res_scale_init_value=True
                 ):
        super().__init__()

        num_channels_reduced = dim // 16
        self.fc1 = nn.Linear(dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, dim, bias=True)
        self.relu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        # self.norm1 = norm_layer(dim)
        # self.attention1 = Attention2(dim=dim//4, drop=drop)
        # self.attention2 = Attention2(dim=dim//4, drop=drop)
        # self.attention3 = Attention2(dim=dim//4, drop=drop)
        # self.attention4 = Attention2(dim=dim//4, drop=drop)
        self.attention1 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=2, shift_size=1))
        self.attention2 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=4, shift_size=2))
        self.attention3 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=6, shift_size=3))
        self.attention4 = nn.Sequential(
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8),
                        SwinTransformerBlock(dim=dim // 4, num_heads=1, window_size=8, shift_size=4))

        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.randomconv = RandomConv(576)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        xx = x.flatten(2).permute(0, 2, 1)
        xx = self.norm3(xx)
        z = xx.clone().permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # x = rearrange(xx, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        # flip1 = random.choice([True, False])
        # flip2 = random.choice([True, False])
        # flip3 = random.choice([True, False])
        # flip4 = random.choice([True, False])
        # column1 = random.choice([True, False])
        # column2 = random.choice([True, False])
        # column3 = random.choice([True, False])
        # column4 = random.choice([True, False])
        #
        # x_scan1 = self.attention1(x1, flip1, column1)
        # x_scan2 = self.attention2(x2, flip2, column2)
        # x_scan3 = self.attention3(x3, flip3, column3)
        # x_scan4 = self.attention4(x4, flip4, column4)
        # x_scan1_reverse = local_reverse(x_scan1.transpose(1, 2), 2, H, W, flip=flip1, column_first=column1)
        # x_scan2_reverse = local_reverse(x_scan2.transpose(1, 2), 2, H, W, flip=flip2, column_first=column2)
        # x_scan3_reverse = local_reverse(x_scan3.transpose(1, 2), 2, H, W, flip=flip3, column_first=column3)
        # x_scan4_reverse = local_reverse(x_scan4.transpose(1, 2), 2, H, W, flip=flip4, column_first=column4)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x_scan1 = self.attention1(x1)
        x_scan2 = self.attention2(x2)
        x_scan3 = self.attention3(x3)
        x_scan4 = self.attention4(x4)
        x_scan_reverse = torch.cat([x_scan1, x_scan2, x_scan3, x_scan4], dim=1)
        # x_scan_reverse = torch.cat([x_scan1_reverse, x_scan2_reverse, x_scan3_reverse, x_scan4_reverse], dim=1)
        # x_scan_reverse = x_scan_reverse.reshape(B, C, H, W)
        x_scan_reverse = self.randomconv(x_scan_reverse)
        x_scan_out = self.res_scale1(x.permute(0, 2, 3, 1)) + x_scan_reverse
        x_scan_out = rearrange(x_scan_out, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)
        x_scan_out_att = x_scan_out * fc_out_2.unsqueeze(1)
        x_scan_out_att = rearrange(x_scan_out_att, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x = self.res_scale2(x_scan_out_att) + self.mlp(self.norm2(x_scan_out_att))
        x = x.permute(0, 3, 1, 2)
        return x

class LocalFormerBlock3(nn.Module):
    """
    Implementation of one MetaFormer block.
    self.res_scale1 and self.res_scale2 is used in MetaFormer blocks.
    self.layer_scale1 and self.layer_scale2 is not used in MetaFormer blocks.
    self.drop_path1 and self.drop_path2 is not used in MetaFormer blocks.
    """

    def __init__(self, dim, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., res_scale_init_value=True
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim//9)
        self.attention1 = Attention31(dim=dim//9, drop=drop)
        # self.attention2 = Attention32(dim=dim//9, drop=drop)
        # self.attention3 = Attention33(dim=dim//9, drop=drop)
        # self.attention4 = Attention34(dim=dim//9, drop=drop)
        self.res_scale1 = Scale(dim=dim//9, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim//9)
        self.mlp = mlp(dim=dim//9, drop=drop)
        self.res_scale2 = Scale(dim=dim//9, init_value=res_scale_init_value)if res_scale_init_value else nn.Identity()

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        B, C, H, W = x.shape
        x_scan1 = self.attention1(self.norm1(x.permute(0, 2, 3, 1)), False, False)
        # x_scan2 = self.attention2(self.norm1(x.permute(0, 2, 3, 1)), True, False)
        # x_scan3 = self.attention3(self.norm1(x.permute(0, 2, 3, 1)), False, True)
        # x_scan4 = self.attention4(self.norm1(x.permute(0, 2, 3, 1)), True, True)
        x_scan1_reverse = x_scan1.transpose(1, 2)
        # x_scan2_reverse = x_scan2.transpose(1, 2).flip([-1])
        # x_scan3_reverse = rearrange(x_scan3.transpose(1, 2), 'b d (h w) -> b d (w h)', h=H, w=W)
        # x_scan4_reverse = rearrange(x_scan4.transpose(1, 2).flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        x_scan_reverse = x_scan1_reverse #+ x_scan2_reverse + x_scan3_reverse + x_scan4_reverse
        x_scan_out = self.res_scale1(x.permute(0, 2, 3, 1)) + x_scan_reverse.transpose(1, 2).reshape(B, H, W, C)
        x = self.res_scale2(x.permute(0, 2, 3, 1)) + self.mlp(self.norm2(x_scan_out))
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, C, H*W).transpose(-1, -2)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 52, 52, 96).to(device)
    model = LocalFormerBlock1(96)
    print(model)
    model = model.to(device)
    output = model(x)
    print("X:", x.shape)
    print("OUT", output.shape)