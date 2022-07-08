##
##
##

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class SpatioTemporalAttention(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 embed_channels: int,
                 num_heads: int, 
                 window_size: int, 
                 window_dilation: int, 
                 cross_view: bool) -> None:
        super().__init__()
        assert embed_channels % num_heads == 0,\
            'Number of embedding channels must be a multiple of the number of heads.'
            
        self.num_heads = num_heads
        self.embed_channels = embed_channels
        self.head_channels = embed_channels // num_heads
        
        self.window_size = window_size
        self.window_dilation = window_dilation
        
        # Layers
        
        self.query = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.query_att = nn.Conv2d(embed_channels, num_heads, kernel_size=1)
        
        self.key = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.key_att = nn.Conv2d(embed_channels, num_heads, kernel_size=1)
        
        if cross_view:
            self.value = nn.Identity()
        else: 
            self.value = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        
        self.key_value = nn.Conv2d(embed_channels, embed_channels, kernel_size=1)
        
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(window_size, 1),
                                dilation=(window_dilation, 1),
                                padding=(self.padding, 0))
        
        self.softmax = nn.Softmax(-1)
    
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
    
    def pool(self, x: torch.Tensor, att: nn.Module) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): tensor with shape (N, C_embed, T, V)
            att (nn.Module): _description_

        Returns:
            torch.Tensor: tensor with shape (N, C_embed, T, 1)
        """
        
        N, C, T, V = x.shape
        
        # (N, num_heads, T, V)
        x_scores: torch.Tensor = self.query_att(x) / (self.head_channels ** 0.5)
        # (N, num_head, T, V * window_size)
        x_scores = self.group_window(x_scores)
        
        # (N, num_head, T, V * window_size)
        x_weights: torch.Tensor = self.softmax(x_scores)
        # (N, T, num_head, V * window_size)
        x_weights = x_weights.transpose(1, 2)
        # (N, T, num_head, 1, V * window_size)
        x_weights = x_weights.unsqueeze(-2)
        
        # (N, num_heads, C_head, T, V)
        x = x.view(N, self.num_heads, self.head_channels, T, V)
        # (N, T, num_heads, V, C_head)
        x = x.permute(0, 3, 1, 4, 2).contiguous()
        
        pooled = 0
        for idx in range(self.window_size):
            widx = idx - (self.window_size // 2)
            # (N, T, num_heads, V, C_head)
            tmp = torch.empty(x.shape, device=x.device)
            
            if widx < 0:
                offset = (-widx) * self.window_dilation
                tmp[:, :offset, :, :, :] = x[:, 0, :, :, :].unsqueeze(1)
                tmp[:, offset:, :, :, :] = x[:, :-offset, :, :, :] 
            elif widx == 0:
                tmp = x
            else:
                offset = widx * self.window_dilation
                tmp[:, -offset:, :, :, :] = x[:, -1, :, :, :].unsqueeze(1)
                tmp[:, :-offset, :, :, :] = x[:, :-offset, :, :, :]
            
            
            # (N, T, num_head, 1, V)
            weights = x_weights[:, :, :, :, (idx * V):((idx + 1) * V)]
            # (N, T, num_head, 1, C_head)
            ptmp = torch.matmul(weights, tmp)
            
            pooled += ptmp
            
            del tmp
            del weights
            del ptmp
        
        # (N, num_heads, C_head, T, 1)
        pooled = pooled.permute(0, 2, 4, 1, 3).contiguous()
        # (N, C_embed, T, 1)
        pooled = pooled.view(N, self.embed_channels, T, 1)
        
        return pooled
    
    def transform_values(self, 
                         values: torch.Tensor, 
                         pooled_key: torch.Tensor, 
                         queries: torch.Tensor) -> torch.Tensor:
        """

        Args:
            values (torch.Tensor): tensor with shape (N, C_embed, T, V)
            pooled_key (torch.Tensor): tensor with shape (N, C_embed, T, 1)
            queries (torch.Tensor): tensor with shape (N, C_embed, T, V)

        Returns:
            torch.Tensor: tensor with shape (N, C_embed, T, V)
        """
        
        res = None
        
        for idx in range(self.window_size):
            widx = idx - (self.window_size // 2)
            # (N, C_embed, T, V)
            vtmp = torch.empty(values.shape, device=values.device)
            # (N, C_embed, T, V)
            qtmp = torch.empty(queries.shape, device=queries.device)
            
            if widx < 0:
                offset = (-widx) * self.window_dilation
                vtmp[:, :, :offset, :] = values[:, :, 0, :].unsqueeze(2)
                vtmp[:, :, offset:, :] = values[:, :,  :-offset, :] 
                
                qtmp[:, :, :offset, :] = queries[:, :, 0, :].unsqueeze(2)
                qtmp[:, :, offset:, :] = queries[:, :,  :-offset, :] 
            elif widx == 0:
                vtmp = values
                qtmp = queries
            else:
                offset = widx * self.window_dilation
                vtmp[:, :, -offset:, :] = values[:, :, -1, :].unsqueeze(2)
                vtmp[:, :, :-offset, :] = values[:, :, :-offset, :]
                
                qtmp[:, :, -offset:, :] = queries[:, :, -1, :].unsqueeze(2)
                qtmp[:, :, :-offset, :] = queries[:, :, :-offset, :]
            
            
            # (N, C_embed, T, V)
            keys_values = pooled_key * vtmp
            # (N, C_embed, T, V)
            keys_values = self.key_value(keys_values)
            # (N, C_embed, T, V)
            out = keys_values + qtmp
            
            if res is None:
                res = out
            else:
                res += out
            
            del vtmp
            del qtmp
            del keys_values
            del out
        
        res /= self.window_size
        
        return res
    
    def forward(self, x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Computes forward step.
        
        Args:
            joints (torch.Tensor): tensor (N, C, T, V) representing the joints
            bones (torch.Tensor): tensor (N, C, T, V) representing the bones
            c_joints (Optional[torch.Tensor], optional): 
            tensor (N, C', T, V) representing the updated joints from the larger view. 
            Defaults to None.
            c_bones (Optional[torch.Tensor], optional): 
            tensor (N, C', T, V) representing the updated bones from the larger view. 
            Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
            updated tensor (N, C', T, V) for the joints, 
            update tensor (N, C', T, V) for the bones
        """
        
        input, cross_input = x
        
        N, C, T, V = input.shape
        
        # (N, C_embed, T, V)
        queries: torch.Tensor = self.query(input)
        # (N, C_embed, T, 1)
        pooled_query = self.pool(queries, self.query_att)
        
        # (N, C_embed, T, V)
        keys = self.key(input)
        # (N, C_embed, T, V)
        query_keys = keys * pooled_query
        # (N, C_embed, T, 1)
        pooled_key = self.pool(query_keys, self.query_att)
        
        # (N, C_embed, T, V)
        values = self.value(cross_input if cross_input is not None else input)
        # (N, C_embed, T, V)
        values = self.transform_values(values, pooled_key, queries)
        
        return values

class SpatioTemporalLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int,
                 num_frames: int, 
                 num_nodes: int, 
                 num_heads: int,
                 window_size: int, 
                 window_dilation: int, 
                 dropout: float, 
                 cross_view: bool) -> None:
        super().__init__()
        
        self.attention = nn.Sequential(
            SpatioTemporalAttention(
                in_channels, out_channels, num_heads, 
                window_size, window_dilation, cross_view),
            nn.Dropout(dropout)
        )
        
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else: 
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.norm1 = nn.LayerNorm([in_channels, num_frames, num_nodes])
        
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm([out_channels, num_frames, num_nodes])   
    
    def forward(self, 
               input: torch.Tensor, 
               cross_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # attention sublayer
        intermediate = self.norm1(input) # We use PreLN
        intermediate = self.attention((input, cross_input))
        intermediate += self.residual(input)
        
        # position-wise feed forward sublayer
        output = self.norm2(intermediate)
        output = self.ffn(intermediate)
        output += intermediate
        
        return output
        

class MultiScaleSpatioTemporalLayer(nn.Module):
    
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 num_frames: int, 
                 num_nodes: int,
                 windows_size: List[int],
                 windows_dilation: List[int],
                 num_heads: int, 
                 dropout: float,
                 cross_view: bool) -> None:
        super().__init__()
        
        assert len(windows_size) == len(windows_dilation), 'Number of branches not equal.'
        
        self.cross_view = cross_view
        
        self.layers = nn.ModuleList()
        for size, dilation in zip(windows_size, windows_dilation):
            layer = SpatioTemporalLayer(
                in_channels, out_channels, num_frames, num_nodes, 
                num_heads, size, dilation, dropout, cross_view)
            
            self.layers.append(layer)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        cross_input = None
        output = 0
        
        for layer in self.layers:
            tmp = layer(input, cross_input)
            output += tmp
            
            if self.cross_view:
                cross_input = tmp
        
        return output