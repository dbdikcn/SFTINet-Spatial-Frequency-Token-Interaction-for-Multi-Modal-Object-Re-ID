import torch.nn as nn
from ..backbones.vit_pytorch import Attention
from ..backbones.vit_pytorch import DropPath
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_
import torch.nn.functional as F
import torch
class ReUnit(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ReBlock(nn.Module):

    def __init__(self, dim, num_heads,depth=1, mode=0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.mode = mode
        for i in range(self.depth):
            self.blocks.append(
                ReUnit(dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                       attn_drop=0.,
                       drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Reconstruct(nn.Module):

    def __init__(self, dim, num_heads,depth=1):
        super().__init__()
        self.re1 = ReBlock(dim, num_heads, depth=depth)
        self.re2 = ReBlock(dim, num_heads, depth=depth)

    def forward(self, x):
        re1 = self.re1(x)
        re2 = self.re2(x)
        return re1, re2

class CRM(nn.Module):
    def __init__(self, dim, num_heads, depth=1, miss='nothing', use_proj_head=False):
        super().__init__()
        self.RGBRE = Reconstruct(dim, num_heads, depth=depth)
        self.NIRE = Reconstruct(dim, num_heads, depth=depth)
        self.TIRE = Reconstruct(dim, num_heads, depth=depth)
        self.miss = miss
        self.use_proj_head = use_proj_head

        # 三种重建损失
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.cos = nn.CosineEmbeddingLoss()

        # 可选投影头用于输出embedding
        if use_proj_head:
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),  # 输出 ReID embedding
                nn.ReLU(inplace=True)
            )

    def reconstruction_loss(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_l1 = self.l1(pred, target)
        cos_target = torch.ones(pred.size(0), device=pred.device)
        loss_cos = self.cos(pred.view(pred.size(0), -1), target.view(target.size(0), -1), cos_target)
        return loss_mse + loss_l1 + 0.5 * loss_cos

    def forward(self, ma, mb, mc):
        if self.training:
            # 三模态重建路径
            RGB_NI, RGB_TI = self.RGBRE(ma)
            NI_RGB, NI_TI = self.NIRE(mb)
            TI_RGB, TI_NI = self.TIRE(mc)

            # 多模态重建损失
            loss_rgb = self.reconstruction_loss(RGB_NI, mb) + self.reconstruction_loss(RGB_TI, mc)
            loss_ni = self.reconstruction_loss(NI_RGB, ma) + self.reconstruction_loss(NI_TI, mc)
            loss_ti = self.reconstruction_loss(TI_RGB, ma) + self.reconstruction_loss(TI_NI, mb)

            total_loss = loss_rgb + loss_ni + loss_ti

            # 可选：输出 embedding，用于 TripletLoss 等外部处理
            if self.use_proj_head:
                feat = self.head((ma + mb + mc) / 3)  # 可替换为 concat 或 attention 融合
                return total_loss, feat  # 返回 loss 和嵌入
            else:
                return total_loss

        else:
            # 推理阶段按缺失模态恢复并输出融合特征
            if self.miss is None:
                return ma
            elif self.miss == 'r':
                NI_RGB, _ = self.NIRE(mb)
                TI_RGB, _ = self.TIRE(mc)
                return (NI_RGB + TI_RGB) / 2
            elif self.miss == 'n':
                RGB_NI, _ = self.RGBRE(ma)
                TI_NI, _ = self.TIRE(mc)
                return (RGB_NI + TI_NI) / 2
            elif self.miss == 't':
                RGB_TI, _ = self.RGBRE(ma)
                NI_TI, _ = self.NIRE(mb)
                return (RGB_TI + NI_TI) / 2
            elif self.miss == 'rn':
                TI_RGB, TI_NI = self.TIRE(mc)
                return TI_RGB, TI_NI
            elif self.miss == 'rt':
                NI_RGB, NI_TI = self.NIRE(mb)
                return NI_RGB, NI_TI
            elif self.miss == 'nt':
                RGB_NI, RGB_TI = self.RGBRE(ma)
                return RGB_NI, RGB_TI

