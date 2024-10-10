"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        # 这部分代码是使用slowfast作为backbone的情况
        # x: (B, T, C, H, W)
        feature_map = self.backbone(x)

        # # 这部分代码是使用presnet作为backbone的情况
        # # x: (B, T, C, H, W)
        # feature_map = []
        # for i in range(x.size(1)):
        #     feature_map.append(self.backbone(x[:, i, :, :, :]))
        
        # # 按特征图尺度分为三部分,feature map 3, 4, 5
        # x_3 = [lst[0] for lst in feature_map] # T×(B, C, H, W) 
        # x_4 = [lst[1] for lst in feature_map]
        # x_5 = [lst[2] for lst in feature_map]

        # x_3 = torch.stack(x_3, dim=2) # B, C, T, H, W
        # x_4 = torch.stack(x_4, dim=2)
        # x_5 = torch.stack(x_5, dim=2)
        
        # # encoder
        # x = self.encoder([x_3, x_4, x_5]) # encoder input: N × (B, C, T, H, W) -> N为特征图尺度数, T为时间步数
        
        # encoder
        x = self.encoder(feature_map)

        # decoder
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
    
@register()
class RTDETR_backbone(nn.Module):
    __inject__ = ['backbone']
    def __init__(self, \
            backbone: nn.Module, 
        ):
            super().__init__()
            self.backbone = backbone
            
    def forward(self, x):
        # # 这部分代码是使用slowfast作为backbone的情况
        # # x: (B, T, C, H, W)
        # feature_map = self.backbone(x)

        # 这部分代码是使用presnet作为backbone的情况
        # x: (B, T, C, H, W)
        feature_map = []
        for i in range(x.size(1)):
            feature_map.append(self.backbone(x[:, i, :, :, :]))
            
        # 按特征图尺度分为三部分,feature map 3, 4, 5
        x_3 = [lst[0] for lst in feature_map] # T×(B, C, H, W) 
        x_4 = [lst[1] for lst in feature_map]
        x_5 = [lst[2] for lst in feature_map]

        x_3 = torch.stack(x_3, dim=2) # B, C, T, H, W
        x_4 = torch.stack(x_4, dim=2)
        x_5 = torch.stack(x_5, dim=2)

        feature_map = [x_3, x_4, x_5]
            
        return feature_map
    
@register()
class RTDETR_en_decoder(nn.Module):
    __inject__ = ['encoder', 'decoder' ]

    def __init__(self, \
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, feature_map, targets=None):
        # # encoder
        # feature_map : [x_3, x_4, x_5]
        x = self.encoder(feature_map) # encoder input: N × (B, C, T, H, W) -> N为特征图尺度数, T为时间步数

        # decoder
        x = self.decoder(x, targets)

        return x