import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F


class Projection(torch.nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection_head(x)
        return F.normalize(x, p=2, dim=1)


class PositiveLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.log_weight.exp())


class ProjectionPhectorMatch(torch.nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Projection head MLP
        self.projection_head = torch.nn.Sequential(
            Linear(self.input_dim, self.hidden_dim, bias=True),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            # Linear(self.hidden_dim, self.hidden_dim, bias=True),
            # torch.nn.BatchNorm1d(self.hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.3),
            # Linear(self.hidden_dim, self.hidden_dim, bias=True),
            # torch.nn.BatchNorm1d(self.hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.3),
            PositiveLinear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection_head(x)
        # x = torch.abs(x)
        normalization = torch.norm(x, p=2, dim=1)
        normalization = 1 / torch.max(torch.ones_like(normalization), normalization)
        normalization = normalization.view(-1, 1).expand(-1, x.shape[1])
        return x * normalization
        # return F.normalize(x, p=2, dim=1)
