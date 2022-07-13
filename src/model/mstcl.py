##
##
##

from typing import List, Optional, Union
import torch 
from torch import nn 

class TemporalConvolution(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            stride=(stride, 1), 
            bias=bias
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)
        

class MultiScaleTemporalConvolutionLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernels_size: Union[List[int], int],
                 dilations: Union[List[int], int],
                 stride: int = 1,
                 activation: nn.Module = nn.Mish(inplace=True),
                 ratio: int = 8,
                 residual: bool = True,
                 ) -> None:
        super().__init__()
        
        if type(kernels_size) == list or type(dilations) == list:
            if type(kernels_size) == list and type(dilations) == list:
                if len(kernels_size) != len(dilations):
                    raise ValueError('The number of kernels does not match the number of dilations.')
                else:
                    num_branches = len(kernels_size)
            elif type(kernels_size) == list:
                num_branches = len(kernels_size)
                dilations = [dilations] * num_branches
            elif type(dilations) == list:
                num_branches = len(dilations)
                kernels_size = [kernels_size] * num_branches
        else: 
            num_branches = 1
            kernels_size = [kernels_size]
            dilations = [dilations]
        
        num_branches += 1 # MaxPool 
        assert out_channels % num_branches == 0, 'Number of branches in TCN not valid.'
        branch_channels = out_channels // num_branches
        
        #####
        
        self.branches = nn.ModuleList()
        for kernel, dilation in zip(kernels_size, dilations):
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels, 
                    kernel_size=1,
                    bias=False,
                ), 
                activation,
                nn.BatchNorm2d(branch_channels), 
                TemporalConvolution(
                    branch_channels, 
                    branch_channels, 
                    kernel_size=kernel,
                    dilation=dilation, 
                    stride=stride,
                    bias=False
                ),
                activation, 
                nn.BatchNorm2d(branch_channels)
            )
            self.branches.append(branch)
        
        # Append MaxPool
        self.branches.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_channels),
            activation,
        ))
        
        # Squeeze and Excitation network
        self.squeeze = nn.Sequential(
            nn.Linear(out_channels, out_channels // ratio), 
            activation,
            nn.Linear(out_channels // ratio, out_channels),
            nn.Sigmoid()
        )
        
        # Residual connections
        
        if not residual: 
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else: 
            self.residual = TemporalConvolution(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride)
          
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        out_branches = []
        for branch in self.branches: 
            out_branches.append(branch(input))
        
        # (N, C_out, T, V)
        intermediate = torch.cat(out_branches, dim=1)
        
        # SENet
        N, C, T, V = intermediate.shape
        s = intermediate.view(N, C, -1)
        # (N, C_out)
        s = s.mean(-1)
        # (N, C_out)
        s: torch.Tensor = self.squeeze(s)
        # (N, C_out, 1, 1)
        s = s.unsqueeze(-1).unsqueeze(-1)
        
        # (N, C_out, T, V)
        output = intermediate * s
        output += self.residual(input)
        
        return output