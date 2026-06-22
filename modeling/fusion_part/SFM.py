import torch
import torch.nn as nn
import torch.nn.functional as F

class SFM(nn.Module):
    def __init__(self, in_channel):
        super(SFM, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(in_channel)  # 替换为 BatchNorm1d
        
        self.stage1 = nn.Sequential(
            nn.Linear(in_channel, in_channel, bias=False),  # 替换为 Linear
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Linear(in_channel, in_channel, bias=False),  # 替换为 Linear
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )

    def forward(self, fa, fb):
        """
        fa, fb: torch.Size([64, 2304])
        """
        # 计算通道维度上的余弦相似度
        cos_sim = F.cosine_similarity(fa, fb, dim=1, eps=1e-6)  # 计算相似度
        cos_sim = cos_sim.unsqueeze(1)  # 变成 (B, 1) 以进行广播
        
        # 结合余弦相似度进行特征融合
        fa = fa + fb * cos_sim
        fb = fb + fa * cos_sim
        
        # 通过 ReLU 进行非线性变换
        fa = self.relu(fa)
        fb = self.relu(fb)

        # 通过 BatchNorm 进行归一化
        fa = self.bn(fa)
        fb = self.bn(fb)

        # 进一步通过两层线性变换进行特征融合
        fa = self.stage1(fa)
        fb = self.stage2(fb)

        # 输出最终融合特征
        output = fa + fb
        return output