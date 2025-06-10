from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights
class MLPDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3,
                 num_layers: int = 3,
                 dropout: float = 0.1) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale
        
        self.pos_embed = nn.Parameter(torch.Tensor(num_modes, 1, local_channels))
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        # out = global_embed
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale
            return torch.cat((loc, scale), dim=-1), pi
        else:
            return loc, pi