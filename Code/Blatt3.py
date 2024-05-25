import torch
import numpy as np




class FCLayer(torch.nn.Module):
    """torch.nn.Module inheritance so that the parameters
        are automatically registered and can be accessed via model.parameters()"""
    def __init__(self, shape_in: int, shape_out: int, activation: str, bias: bool = True):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.weights = torch.nn.Parameter(torch.rand(shape_out, shape_in))
        torch.nn.init.xavier_uniform_(self.weights)
        # automatically requires grad
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(shape_out, 1))
            torch.nn.init.xavier_uniform_(self.bias)
        else:
            self.bias = torch.zeros(shape_out, 1)
        self.activation = activation
        match activation:
            case "relu":
                # own implementation would just be torch.maximum(0,x)
                self.activation = torch.relu
            case "sigmoid":
                self.activation = torch.sigmoid
            case "tanh":
                self.activation = torch.tanh

    def propagate_forward(self, x: torch.Tensor) -> torch.Tensor:
        # output z
        z = x @ self.weights.t() + self.bias.t()
        return self.activation(z)
