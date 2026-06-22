import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., local_feature=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.local_feature = local_feature

        if local_feature:
            # 引入局部特征的线性映射（等效于轻量卷积）
            self.local_projection = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

    def forward(self, x, get_attn=False):

        B, N, C = x.shape#torch.Size([64, 129, 384])

        # 计算 q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#torch.Size([3, 64, 12, 129, 32])
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别提取 q, k, v,torch.Size([64, 12, 129, 32])

        # 自注意力机制
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力得分 torch.Size([64, 12, 129, 129])
        attn = attn.softmax(dim=-1)  # 归一化注意力得分torch.Size([64, 12, 129, 129])
        attn = self.attn_drop(attn)  # 应用 dropout  torch.Size([64, 12, 129, 129])

        # 将注意力加权的值应用到 v 上
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#torch.Size([64, 129, 384])

        if self.local_feature:
            # 引入局部特征增强
            local_features = self.local_projection(x)  # 线性映射作为局部增强模块,torch.Size([64, 129, 384])
            x = x + local_features  # 残差连接局部增强特征

        # 最后的线性投影和 dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        if get_attn:
            return x, attn  # 可选返回注意力矩阵
        return x#torch.Size([64, 129, 384])

class DynamicConv1d(nn.Module):  # 修改为适用 (B, N, C) 的动态卷积
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=4, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.dim = dim

        # 动态卷积权重
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size), requires_grad=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.GELU(),
            nn.Linear(dim // reduction_ratio, dim * num_groups * kernel_size),
        )

        # 动态卷积偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, N, C = x.shape  # torch.Size([64, 129, 384])

        # 计算动态权重 scale
        scale = self.proj(x.mean(dim=1))  # (B, C -> B, num_groups * C * K)#torch.Size([64, 768]),torch.Size([64, 2304])
        scale = scale.view(B, self.num_groups, C, self.K)  # (B, num_groups, C, K) torch.Size([64, 2, 384, 3])
        scale = torch.softmax(scale, dim=1)  # 对分组权重进行 softmax  torch.Size([64, 2, 384, 3])
        weight = (scale * self.weight.unsqueeze(0)).sum(dim=1)  # 聚合分组权重 (B, C, K)torch.Size([64, 384, 3])

        # 计算动态偏置 bias
        if self.bias is not None:
            scale_bias = self.proj(x.mean(dim=1))  # (B, num_groups * C) orch.Size([64, 2304])
            scale_bias = torch.softmax(scale_bias.view(B, self.num_groups, C, self.K), dim=3)#torch.Size([64, 2, 384, 3])
            bias = (scale_bias * self.bias.unsqueeze(0).unsqueeze(3)).sum(dim=1)  # (B, C, 1)
            bias = bias.squeeze(2) 
            # bias = (scale_bias * self.bias.unsqueeze(0)).sum(dim=1)  # (B, C)
        else:
            bias = None

        # 应用动态卷积

        x = x.transpose(1, 2)  #x = torch.Size([64, 384, 129]),weight = torch.Size([64, 384, 3])
        # weight = weight.view(384, 1, self.K)  # (B * C, 1, K)#torch.Size([24576, 1, 3])
        # x = x.reshape(1, B * C, N)  # (1, B * C, N) 为了支持分组卷积#torch.Size([1, 24576, 129])
        weight= torch.randn(384,384,3).cuda()
        x = F.conv1d(x, weight=weight, bias=None, padding=1, groups = 1)  # 卷积操作,torch.Size([64, 384, 129])
        x = x.view(B, C, N).transpose(1, 2)  # 转回 (B, N, C)

        return x


class HybridTokenMixer(nn.Module):  # 修改后的混合 Token 模块 dim =768 num_heads= 12
    def __init__(self, dim, num_heads, kernel_size=3, num_groups=2, sr_ratio=1, reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv1d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim // 2, num_heads=num_heads)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
    def mixer(self, x):
        output_list= []
        if self.training:
            for x in x:
                B, N, C = x.shape#torch.Size([64, 129, 768])

                # 划分通道
                x1, x2 = torch.chunk(x, 2, dim=2)  # (B, N, C // 2)#torch.Size([64, 129, 384])

                # 局部和全局处理
                x1 = self.local_unit(x1)  # 局部动态卷积
                x2 = self.global_unit(x2)  # 全局注意力

                # 拼接后进行线性投影
                x = torch.cat([x1, x2], dim=2)  # (B, N, C)
                x = self.proj(x) + x  # 残差连接
                
                output_list.append(x)
        return output_list 
       
    def forward(self, RGB_cash, NI_cash, TI_cash, relative_pos_enc=None):
        RGB_cash = []
        NI_cash = []
        TI_cash = []
        if self.training:
            RGB_cash = self.mixer(RGB_cash)
            NI_cash = self.mixer(NI_cash)
            TI_cash = self.mixer(TI_cash)
            loss = nn.MSELoss()(RGB_cash[0], NI_cash)[0] + nn.MSELoss()(RGB_cash[0], TI_cash[0]) + nn.MSELoss()(NI_cash[0], TI_cash[0])
            # for x in x:
            #     B, N, C = x.shape#torch.Size([64, 129, 768])

            #     # 划分通道
            #     x1, x2 = torch.chunk(x, 2, dim=2)  # (B, N, C // 2)#torch.Size([64, 129, 384])

            #     # 局部和全局处理
            #     x1 = self.local_unit(x1)  # 局部动态卷积
            #     x2 = self.global_unit(x2)  # 全局注意力

            #     # 拼接后进行线性投影
            #     x = torch.cat([x1, x2], dim=2)  # (B, N, C)
            #     x = self.proj(x) + x  # 残差连接
            #     output_list.append(x)
            return RGB_cash, NI_cash, TI_cash, loss
        else:
            RGB_cash = self.mixer(RGB_cash)
            NI_cash = self.mixer(NI_cash)
            TI_cash = self.mixer(TI_cash)
            return RGB_cash, NI_cash, TI_cash