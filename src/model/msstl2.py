##
##
##

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
        
class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_heads: int, 
                 cross_view: bool) -> None:
        super().__init__()
        assert out_channels % num_heads == 0,\
            'Number of output channels must be a multiple of the number of heads.'
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        
        self.qkv_linear = nn.Linear(in_channels, 3 * out_channels, bias=False)
        self.linear = nn.Linear(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor, cross_x: torch.Tensor) -> torch.Tensor:
        
        #input, cross_input = x
        input = x
        N, L, C = input.shape
        
        qkv: torch.Tensor = self.qkv_linear(input)
        qkv = qkv.view(N, L, self.num_heads, 3 * self.head_channels)
        # (N, num_heads, L, head_channels)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        
        output = []
        for idx in range(0, L, 25):
            single_q = q[:, :, idx:idx+25]
            
            attn = torch.matmul(single_q, k.transpose(-2, -1))
            attn = attn / math.sqrt(self.out_channels)
            attn = F.softmax(attn, dim=-1)
            
            out = torch.matmul(attn, v)
            output.append(out)
        
        # (N, num_head, L, head_channels)
        output: torch.Tensor = torch.cat(output, dim=2)
        # (N, L, num_heads, head_channels)
        output = output.permute(0, 2, 1, 3).contiguous() 
        output = output.view(N, L, self.out_channels)
        output = self.linear(output)
        
        return output

class SpatioTemporalLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int,
                 num_heads: int,
                 window_size: int, 
                 window_dilation: int, 
                 dropout: float, 
                 cross_view: bool) -> None:
        super().__init__()
        
        self.window_size = window_size
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(window_size, 1),
            dilation=(window_dilation, 1),
            padding=(self.padding, 0))
        
        self.attn = SpatioTemporalAttention(in_channels, out_channels, num_heads, cross_view)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else: 
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
         
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            ##nn.Dropout(dropout)
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        
    def group_window(self, x: torch.Tensor) -> torch.Tensor:
        """
        Groups vectors belonging to the same window
        
        Args:
            x (torch.Tensor): tensor with shape (N, C, T, V)

        Returns:
            torch.Tensor: tensor with shape (N, C, T, V_w)
            where V_w = V * window_size
        """
        N, C, T, V = x.shape

        x = self.unfold(x)
        x = x.view(N, C, self.window_size, T, V) # (N, C, window_size, T, V)
        x = x.permute(0, 1, 3, 2, 4).contiguous() # (N, C, T, window_size, V)
        x = x.view(N, C, T, V * self.window_size) # (N, C, T, V * window_size)

        return x
    
    def forward(self, 
               x: torch.Tensor, 
               cross_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        N, C, T, V = x.shape
        
        # (N, C, T, V * window_size)
        xg = self.group_window(x)
        # (N, T, V * window_size, C)
        xg = xg.permute(0, 2, 3, 1).contiguous()
        # (N * T, V * window_size, C)
        xg = xg.view(N * T, V * self.window_size, C)
        
        # attention sublayer
        intermediate: torch.Tensor = self.attn(xg, cross_x)
        # (N, T, V * window_size, C_out)
        intermediate = intermediate.view(N, T, self.window_size, V, -1)
        # (N, C_out, T, window_size, V)
        intermediate = intermediate.permute(0, 4, 1, 2, 3).contiguous()
        # (N, C_out, T, V)
        intermediate = intermediate.mean(3)
        
        res = self.residual(x)
        intermediate = self.norm1(intermediate + res)
        
        # position-wise feed forward sublayer
        output = self.ffn(intermediate)
        output = self.norm2(output + intermediate)

        return output
        

class MultiScaleSpatioTemporalLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int,
                 windows_size: List[int],
                 windows_dilation: List[int],
                 num_heads: int, 
                 dropout: float,
                 cross_view: bool) -> None:
        super().__init__()
        
        assert len(windows_size) == len(windows_dilation), 'Number of branches not equal.'
        
        self.cross_view = cross_view
        
        #num_branches = len(windows_size)
        self.branches = nn.ModuleList()
        for size, dilation in zip(windows_size, windows_dilation):
            branch = SpatioTemporalLayer(
                in_channels, out_channels,
                num_heads, size, dilation, 
                dropout, cross_view)
            
            self.branches.append(branch)
            
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        cross_input = None
        output = 0
        
        for layer in self.branches:
            tmp = layer(input, cross_input)
            output += tmp
            
            if self.cross_view:
                cross_input = tmp
        
        output = self.norm(output)
        
        return output