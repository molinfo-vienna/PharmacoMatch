import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F


class PositiveLinear(torch.nn.Module):
    """Dense layer with non-negative weights.

    A dense layer that works with the absolute values of the learned weights for the
    affine transformation of the input features. No bias is used.

    Args:
        in_features (int): Input embedding dimension.
        out_features (int): Output embedding dimension.
    """

    def __init__(self, in_features: int, out_features: int):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pos_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.pos_weight)

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, self.pos_weight.abs())


class ProjectionPhectorMatch(torch.nn.Module):
    """MLP to project the input features to the positive real number space.

    The order embedding space is constructed with positive real number. This property
    of the order embedding space is enforced by the use of the PositiveLinear layer
    after ReLU activation.

    Args:
        input_dim (int, optional): Input embedding dimension. Defaults to 2048.
        hidden_dim (int, optional): Hidden embedding dimension. Defaults to 2048.
        output_dim (int, optional): Output embedding dimpension. Defaults to 128.
        num_layers (int, optional): Number of hidden layers. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.3.
        norm (int, optional): Normalization of the output embedding length w.r.t. the
            number of interaction points of the input pharmacophore. If None,
            normalization is not applied. Otherwise, norm correspond to the p-norm for
            normalization. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        norm: int = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm = norm

        # Projection head MLP
        self.projection_head = torch.nn.Sequential()
        for _ in range(num_layers):
            self.projection_head.append(
                Linear(self.input_dim, self.hidden_dim, bias=True)
            )
            self.projection_head.append(torch.nn.BatchNorm1d(self.hidden_dim)),
            self.projection_head.append(torch.nn.ReLU()),
            self.projection_head.append(torch.nn.Dropout(p=self.dropout)),
            input_dim = hidden_dim
        self.projection_head.append(PositiveLinear(self.hidden_dim, self.output_dim))

    def forward(self, x: Tensor, num_ph4_features: Tensor) -> Tensor:
        x = self.projection_head(x)
        if self.norm:
            normalization = torch.norm(x, p=self.norm, dim=1)
            normalization = 1 / torch.max(
                torch.ones_like(normalization), normalization / num_ph4_features
            )
            normalization = normalization.view(-1, 1).expand(-1, x.shape[1])
            return x * normalization
        else:
            return x


class Projection(torch.nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ) -> None:
        """A simple MLP that projects the output embeddings onto a unit sphere.

        Args:
            input_dim (int, optional): Input embedding dimension. Defaults to 2048.
            hidden_dim (int, optional): Hidden embedding dimension. Defaults to 2048.
            output_dim (int, optional): Output embedding dimension. Defaults to 128.
        """
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
