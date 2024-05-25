import torch
import numpy as np




class FCLayer(torch.nn.Module):
    """torch.nn.Module inheritance so that the parameters
        are automatically registered and can be accessed via model.parameters()"""
    def __init__(self, shape_in: int, shape_out: int, activation: str, bias: bool = True):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.weights = torch.nn.Parameter(torch.empty(shape_in, shape_out))
        # automatically requires grad
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(shape_out))
        else:
            self.bias = torch.zeros(shape_out)
        self.acti_vation = activation
        match activation:
            case "relu":
                # own implementation would just be torch.maximum(0,x)
                self.acti_vation = torch.relu
                torch.nn.init.kaiming_uniform_(self.weights, nonlinearity='relu')
            case "sigmoid":
                self.acti_vation = torch.sigmoid
            case "tanh":
                self.acti_vation = torch.tanh
            case _:
                self.acti_vation = lambda x: x # id
                torch.nn.init.xavier_uniform_(self.weights)

    def propagate_forward(self, x: torch.Tensor) -> torch.Tensor:
        # output z
        z = x @ self.weights + self.bias #xA^T + b, x is a row Vector
        return self.acti_vation(z)
