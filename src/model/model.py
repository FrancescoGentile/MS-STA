##
##
##

from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import Embeddings
from .msstl import MultiScaleSpatioTemporalLayer
from .mstcl import MultiScaleTemporalConvolutionLayer
from ..dataset.skeleton import SkeletonGraph

class MultiScaleSpatioTemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_frames: int, 
                 num_nodes: int,
                 cross_view: bool, 
                 stride: int) -> None:
        super().__init__()
        
        self.spatio = MultiScaleSpatioTemporalLayer(
            in_channels=in_channels,
            out_channels=out_channels, 
            num_frames=num_frames, 
            num_nodes=num_nodes,
            windows_size=[5, 5, 5], 
            windows_dilation=[1, 2, 3], 
            num_heads=8, 
            dropout=0.1, 
            cross_view=cross_view)
        
        self.temporal = MultiScaleTemporalConvolutionLayer(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernels_size=[3, 3], 
            dilations=[1, 2],
            stride=stride, 
            residual_kernel_size=3
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
        output= self.temporal(output)
        output += self.residual(input)
        
        return output
    
class Classifier(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        
        self._dropout = nn.Dropout(dropout)
        self._fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, M, C, _, _ = input.shape 
        input = input.view(N, M, C, -1)
        # (N, C)
        avg = input.mean(-1).mean(1)
        avg = self._dropout(avg)
        
        return self._fc(avg)
        

class Model(nn.Module):
    
    def __init__(self, 
                 config: ModelConfig, 
                 skeleton: SkeletonGraph, 
                 num_classes: int, 
                 num_frames: int) -> None:
        super().__init__()
        
        self.embeddings = Embeddings(9, 64, num_frames, skeleton)
        
        self.blocks = nn.ModuleList()
        in_channels = 64
        out_channels = 64
        in_frames = num_frames
        num_nodes = 50
        for idx in range(9):
            cross_view = (idx % 3) == 1
            stride = 2 if idx == 3 or idx == 6 else 1
            if stride > 1:
                out_channels *= 2
            
            self.blocks.append(MultiScaleSpatioTemporalBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                num_frames=in_frames, 
                num_nodes=num_nodes, 
                cross_view=cross_view, 
                stride=stride
            ))
            
            in_channels = out_channels
            if stride > 1:
                in_frames = in_frames // 2
        
        self.classifier = Classifier(256, num_classes, dropout=0.1)
        
    def forward(self, 
                joints: torch.Tensor, 
                bones: torch.Tensor) -> torch.Tensor:
        
        N, C, T, V, M = joints.shape
        joints = joints.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        bones = bones.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # apply embeddings
        input: torch.Tensor = self.embeddings(joints, bones)
        
        # apply blocks 
        for block in self.blocks:
            input = block(input)
        
        # classify
        _, C, T, V = input.shape
        # (N, M, C_out, T, V)
        output = input.view(N, -1, C, T, V)
        return self.classifier(output)

from .config import ModelConfig