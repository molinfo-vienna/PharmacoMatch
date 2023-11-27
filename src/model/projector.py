import torch
from torch.nn import Linear
import torch.nn.functional as F

from dataset import *
from utils import *


class Projection(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projection head MLP
        self.projection_head = torch.nn.Sequential(
            Linear(self.input_dim, self.hidden_dim, bias=True),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.projection_head(x)
        return F.normalize(x, p=2, dim=1)
