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
                 stride: int = 1) -> None:
        super().__init__()
        
        padding = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2        
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                stride=(stride, 1)
            ),
            nn.InstanceNorm2d(out_channels)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layer(input)
        

class MultiScaleTemporalConvolutionLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernels_size: Union[List[int], int],
                 dilations: Union[List[int], int],
                 stride: int = 1,
                 activation: nn.Module = nn.SiLU(),
                 residual_kernel_size: Optional[int] = None
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
        
        num_branches += 2 # MaxPool and 1x1 conv
        assert out_channels % num_branches == 0, 'Number of branches in TCN not valid.'
        branch_channels = out_channels // num_branches
        
        self.branches = nn.ModuleList()
        for kernel, dilation in zip(kernels_size, dilations):
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels, 
                    kernel_size=1,
                ), 
                activation,
                nn.InstanceNorm2d(branch_channels), 
                TemporalConvolution(
                    branch_channels, 
                    branch_channels, 
                    kernel_size=kernel,
                    dilation=dilation, 
                    stride=stride 
                )
            )
            self.branches.append(branch)
        
        # Append MaxPool
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            activation,
            nn.InstanceNorm2d(branch_channels),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.InstanceNorm2d(branch_channels)
        ))
        
        # Append 1x1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1,
                      padding=0, stride=(stride, 1)),
            nn.InstanceNorm2d(branch_channels)
        ))
        
        if residual_kernel_size is None: 
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else: 
            self.residual = TemporalConvolution(in_channels, 
                                           out_channels, 
                                           kernel_size=residual_kernel_size, 
                                           stride=stride)

        self.activation = activation
          
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_branches = []
        for branch in self.branches: 
            out_branches.append(branch(input))
        
        output = torch.cat(out_branches, dim=1)
        
        output += self.residual(input)
        output = self.activation(output)
        
        return output