##
##
##

from __future__ import annotations

import torch
import torch.nn as nn

from .tools import init_layers
from .embedding import Embeddings
from .msstl import MultiScaleSpatioTemporalLayer
from .mstcl import MultiScaleTemporalConvolutionLayer
from ..dataset.skeleton import SkeletonGraph

class MultiScaleSpatioTemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 cross_view: bool, 
                 stride: int) -> None:
        super().__init__()
        
        self.spatio = MultiScaleSpatioTemporalLayer(
            in_channels=in_channels,
            out_channels=out_channels, 
            windows_size=[3, 3, 5], 
            windows_dilation=[1, 2, 3], 
            num_heads=8, 
            dropout=0.1, 
            cross_view=cross_view)
        
        self.temporal = MultiScaleTemporalConvolutionLayer(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernels_size=[3, 5, 7], 
            dilations=[1, 2, 3],
            stride=stride, 
            residual=True
        )
        
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else: 
            self.residual = nn.Conv2d(in_channels, 
                                      out_channels, 
                                      kernel_size=1,
                                      stride=(stride, 1))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        output = self.spatio(input)
        output = self.temporal(output)
        output += self.residual(input)
        
        return output
    
class Classifier(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, M, C, _, _ = input.shape 
        input = input.view(N, M, C, -1)
        # (N, C)
        avg = input.mean(-1).mean(1)
        avg = self.dropout(avg)
        
        return self.fc(avg)
        

class Model(nn.Module):
    
    def __init__(self, 
                 config: ModelConfig, 
                 skeleton: SkeletonGraph, 
                 num_classes: int, 
                 num_frames: int) -> None:
        super().__init__()
        
        self.joints_norm = nn.BatchNorm1d(9 * 2 * 25)
        self.bones_norm = nn.BatchNorm1d(9 * 2 * 25)
        self.embeddings = Embeddings(9, 64, num_frames, skeleton)
        
        self.norm = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p = 0.1)
        
        self.blocks = nn.ModuleList()
        in_channels = 64
        out_channels = 64
        for idx in range(9):
            cross_view = (idx % 3) == 1
            stride = 2 if idx == 3 or idx == 6 else 1
            if stride > 1:
                out_channels *= 2
            
            self.blocks.append(MultiScaleSpatioTemporalBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
                cross_view=cross_view, 
                stride=stride
            ))
            
            in_channels = out_channels
        
        self.classifier = Classifier(256, num_classes, dropout=0.1)
        
        self.apply(init_layers)
        
    def forward(self, 
                joints: torch.Tensor, 
                bones: torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = joints.shape
        j = joints.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        b = bones.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        '''
        # Normalize input
        j = joints.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        j: torch.Tensor = self.joints_norm(j)
        j = j.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        b = bones.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        b: torch.Tensor = self.bones_norm(b)
        b = b.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        '''
        
        # Apply embeddings
        input: torch.Tensor = self.embeddings(j, b)
        input = self.norm(input)
        input = self.dropout(input)
        
        # Apply blocks 
        for block in self.blocks:
            input = block(input)
        
        # Classify
        _, C, T, V = input.shape
        # (N, M, C_out, T, V)
        output = input.view(N, -1, C, T, V)
        return self.classifier(output)

from .config import ModelConfig