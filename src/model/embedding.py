##
##
##

from typing import Tuple
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
from scipy.linalg import eig
import math

from ..dataset.skeleton import SkeletonGraph

class Embeddings(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_frames: int,
                 skeleton: SkeletonGraph) -> None:
        super().__init__()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_frames = num_frames
        self._skeleton = skeleton
        
        id_embed, id_embed_channels = self._get_id_embeddings()
        self.id_embeddings = Parameter(id_embed)
        self._id_embed_channels = id_embed_channels
    
        joint_embed, bone_embed, type_embed_channels = self._get_type_embeddings()
        self.joint_embeddings = Parameter(joint_embed)
        self.bone_embeddings = Parameter(bone_embed)
        self._type_embed_channels = type_embed_channels
        
        self.register_buffer('temporal_enc', self._get_temporal_encoding(), persistent=False)
        
        self.embed_proj = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
    
    def _get_id_embeddings(self) -> Tuple[torch.Tensor, int]:
        laplacian = self._skeleton.laplacian_matrix
        _, vectors = eig(laplacian, left=False, right=True)
        id_embeddings = torch.from_numpy(vectors).float()
        id_embeddings = id_embeddings.unsqueeze(0).unsqueeze(2) # (1, C, 1, V)
        
        return id_embeddings, id_embeddings.size(1)
    
    def _get_type_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        type_len = self._out_channels - self._in_channels - self._id_embed_channels * 2
        if type_len < 1:
            raise ValueError('Type embeddings length is less than 1.')
        
        bound = 1 / math.sqrt(type_len)
        
        joint_embed = torch.empty(type_len)
        nn.init.uniform_(joint_embed, -bound, +bound)
        joint_embed = joint_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) #(1, C, 1, 1)
        
        bone_embed = torch.empty(type_len)
        nn.init.uniform_(joint_embed, -bound, +bound)
        bone_embed = bone_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1, C, 1, 1)
        
        return joint_embed, bone_embed, type_len

    def _get_temporal_encoding(self) -> torch.Tensor:
        
        te = torch.zeros(self._num_frames, self._out_channels)
        position = torch.arange(0, self._num_frames, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self._out_channels, 2).float() * (-math.log(10000.0) / self._out_channels))
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        
        te = te.transpose(0, 1) # (C, T)
        te = te.unsqueeze(0).unsqueeze(-1) # (1, C, T, 1)
        
        return te

    def forward(self, 
                joints: torch.Tensor,
                bones: torch.Tensor) -> torch.Tensor:
        
        N, C, T, V = joints.shape
        
        ide = self.id_embeddings.expand(N, self._id_embed_channels, T, V)
        te = self.temporal_enc[:, :, :T, :]
        te = te.expand(N, self._out_channels, T, V)
        
        # joints
        je = self.joint_embeddings.expand(N, self._type_embed_channels, T, V)
        j = torch.cat([joints, ide, ide, je, te], dim=1)
        
        # bones
        conn = self._skeleton.joints_connections
        first = ide[:, :, :, conn[:, 0]]
        second = ide[:, :, :, conn[:, 1]]
        be = self.bone_embeddings.expand(N, self._type_embed_channels, T, V)
        b = torch.cat([bones, first, second, be, te], dim=1)
        
        concat = torch.cat([j, b], dim=-1)
        output = self.embed_proj(concat)
        
        return output
        