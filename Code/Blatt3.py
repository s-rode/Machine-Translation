import torch


class FCLayer(torch.nn.Module):
    def __init__(self, shape_in: int, shape_out: int, weight_init: str = "xavier", bias: bool = True):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        # parameter automatically requires grad
        self.weights = torch.nn.Parameter(torch.empty(shape_in, shape_out))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(shape_out))
        else:
            self.bias = torch.zeros(shape_out)
        # normal distribution init
        match weight_init:
            case "he":
                torch.nn.init.kaiming_uniform_(self.weights, nonlinearity="relu")
            case "xavier":
                torch.nn.init.xavier_uniform_(self.weights)
            case _:
                print("WARNING: random weight init!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # xA^T + b, x is a row Vector
        """"
        A*x = (A*x^tr)^tr = (x*A^tr)
        """
        z = x @ self.weights + self.bias
        return z
