import torch
import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
from ..backbones.vit_pytorch import DropPath
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math
class DWTForward(nn.Module):#haar小波变换

    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):

        yh = []#torch.Size([16, 3, 256, 128])
        ll = x#torch.Size([64, 3, 256, 128])
        mode = lowlevel.mode_to_int(self.mode)
        for j in range(self.J):
            ll, lh, hl, hh = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)

        return ll, lh, hl, hh



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.3, proj_drop=0.3):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#torch.Size([64, 4, 1, 32])
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#torch.Size([64, 4, 2, 32])
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#torch.Size([64, 4, 2, 32])
        attn = (q @ k.transpose(-2, -1)) * self.scale#torch.Size([64, 4, 1, 2])
        attn = attn.softmax(dim=-1)#torch.Size([64, 4, 1, 2])
        attn = self.attn_drop(attn)#torch.Size([64, 4, 1, 2])
        x = (attn @ v).transpose(1, 2)#torch.Size([64, 1, 4, 32])
        x = x.reshape(B, C)#torch.Size([64, 128])
        x = self.proj(x)#torch.Size([64, 128])
        x = self.proj_drop(x)#torch.Size([64, 128])
        return x


class RotationAttention(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    
class BlockRotation(nn.Module):

    def __init__(self, dim, num_heads, mode=0):
        super().__init__()
        self.Rotation = RotationAttention(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.4,
                                          attn_drop=0.4,
                                          drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.mode = mode
        # self.weighting = ModalityWeighting()
    def forward(self, x, y, z):
        if self.mode == 0:
            x_cls = self.Rotation(x[:, 0, :], y[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], z[:, 1:, :])
            z_cls = self.Rotation(z[:, 0, :], x[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=-2)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=-2)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=-2)
            return x, y, z
        elif self.mode == 1:
            x_cls = self.Rotation(x[:, 0, :], x[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], y[:, 1:, :])
            z_cls = self.Rotation(z[:, 0, :], z[:, 1:, :])
            cls = torch.cat([x_cls, y_cls, z_cls], dim=-1)
            return cls
        else :
            x_cls = self.Rotation(x[:, 0, :], x[:, 1:, :])
            y_cls = self.Rotation(y[:, 0, :], y[:, 1:, :])
            z_cls = self.Rotation(z[:, 0, :], z[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=1)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=1)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=1)
            cls = torch.cat([x, y, z], dim=-1)
            # cls = self.weighting(x_cls, y_cls, z_cls)
            return cls  

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=16, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn( w, h//2 + 1, 3, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=(8,16)):
        # 输入 x: (B, C, N)
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, C)

        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C).to(torch.float32)  # (B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (B, H, W_rfft, C)#torch.Size([64,8, 9, 3])
#w_rfft = w//2+1
        weight = torch.view_as_complex(self.complex_weight)  # (H, W_rfft, C) torch.Size([16, 8, 128])
        x = x * weight  # 广播乘法

        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')  # (B, H, W, C)
        x = x.view(B, N, C)  # (B, N, C)
        x = x.permute(0, 2, 1)  # 恢复为 (B, C, N)

        return x

class Frequency(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.Ro_start = BlockRotation(dim, num_heads)
        self.Ro_middle = BlockRotation(dim, num_heads)
        self.Ro_end = BlockRotation(dim, num_heads, mode=1)
        self.Ro_end1 = BlockRotation(dim, num_heads, mode=2)
        self.DWT = DWTForward(J=4, wave='haar', mode='zero').cuda()
        
        self.filter = GlobalFilter(dim)
        # self.freq_pos_enc = nn.ParameterDict({
        #     'low': nn.Parameter(torch.randn(1, 1, dim))
        # })

        self.gate_low = nn.Sequential(
            nn.Linear(dim* 3, dim* 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(dim* 3, dim* 3)
        )
        self.gate_high = nn.Sequential(
            nn.Linear(dim * 3, dim* 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(dim* 3, dim* 3)
        )
        # 多尺度低频引导高频
        # self.low_guided_high = CrossAttention(dim, num_heads)
        self.cross_high = CrossAttention(dim = 384, num_heads = 4)
        # 增强分类层
        self.proj_low = nn.Sequential(
            nn.Linear(dim* 3, dim * 3),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(dim * 3, dim * 3)
        )
        self.dropout_low= nn.Dropout(0.7)
        self.norm_cls = nn.BatchNorm1d(dim * 3)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(dim * 3, dim * 18)#3,6,18
        self.drop_path = DropPath(0.4) #64.9 0.4
        self.apply(self._init_weights)
        self.alpha = nn.Parameter(torch.tensor(0.5))#0.7
        self.beta = nn.Parameter(torch.tensor(0.2))#0.5
        self.hf_fusion = nn.Sequential(
                        nn.Conv1d(in_channels=3*3, out_channels=3, kernel_size=1),
                        nn.BatchNorm1d(3),
                        nn.ReLU(),
                        nn.Dropout(0.5)
                                    )
        self.stride = 16
        self.keep = 10
        print("FTM HERE!!!")
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def frequency_loss(self, cls, low, high):
        low_proj = self.proj_low[0](low)
        mse_loss = F.mse_loss(cls, low_proj)
        cosine_loss = 1 - F.cosine_similarity(cls, low_proj, dim=-1).mean()
        l1_loss = high.abs().mean()
        total_loss = mse_loss + 0.5 * cosine_loss + 0.01 * l1_loss#0.05
        return total_loss


    def mask(self, Inverse,window_size=16):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        Inverse = torch.mean(Inverse, dim=1)
        # create a tensor to store the count of non-zero elements
        count_tensor = torch.zeros((batch_size, height //self.stride, width // self.stride),
                                   dtype=torch.int).cuda()
        # For each image in the batch
        for batch_idx in range(batch_size):
            image = Inverse[batch_idx]  # 获取当前图像
            # With a sliding window, unfold the image into a tensor
            unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), window_size, stride=self.stride)
            # Turns elements greater than 0 into binary, then sums to count the number of elements greater than 0
            count = unfolded.gt(0).sum(1)
            count = count.view(height // self.stride, width // self.stride)
            count_tensor[batch_idx] = count
            # Get the index of the maximum value of each image
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // self.stride) * (width // self.stride)),
                                           dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
    
        return selected_tokens_mask

    def forward(self, x, y, z):
        B, C, H, W = x.shape
        mask_token = self.mask(x)
        lx, lhx, hlx, hhx = [self.dropout(self.filter(t.reshape(B, C, -1))) for t in self.DWT(x)]#torch.Size([64, 3, 128])
        ly, lhy, hly, hhy = [self.dropout(self.filter(t.reshape(B, C, -1))) for t in self.DWT(y)]
        lz, lhz, hlz, hhz = [self.dropout(self.filter(t.reshape(B, C, -1))) for t in self.DWT(z)]

        hf_x = torch.cat([lhx, hlx, hhx], dim=1)
        hf_y = torch.cat([lhy, hly, hhy], dim=1)#torch.Size([64, 9, 128])
        hf_z = torch.cat([lhz, hlz, hhz], dim=1)
        
        hf_x = self.hf_fusion(hf_x)
        hf_y = self.hf_fusion(hf_y)#torch.Size([64, 3, 128])
        hf_z = self.hf_fusion(hf_z)

        lx, ly, lz = self.Ro_start(x=lx, y=ly, z=lz)
        lx, lz, ly = self.Ro_middle(x=lx, y=lz, z=ly)
        low = self.Ro_end(x=lx, y=ly, z=lz)#torch.Size([64, 384])
        low = self.dropout_low(low)

        def process_high(a, b, c, cross):
            a, b, c = self.Ro_start(a, b, c)
            a, c, b = self.Ro_middle(a, c, b)
            feat = self.Ro_end1(a, b, c)
            feats = self.Ro_end(a,b,c)
            feat = self.dropout(feat)
            feats = self.dropout(feats)
            return cross(low, feat) + feats  

        high = process_high(hf_x, hf_y, hf_z, self.cross_high)
        alpha = torch.sigmoid(self.gate_low(low))
        beta = torch.sigmoid(self.gate_high(high))
        fused = self.alpha * alpha * low + self.beta * beta * high

        cls = self.proj_low(fused)
        cls = self.norm_cls(cls)
        cls = self.dropout(cls)

        if self.training:
            loss = self.frequency_loss(cls, low, high)
            cls = self.drop_path(self.linear(cls))
            return cls, mask_token, loss
        else:
            cls = self.drop_path(self.linear(cls))
            return cls, mask_token


