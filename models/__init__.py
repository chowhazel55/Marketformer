
from .Marketformer import Marketformer
from .LSTM import LSTM
from .GRU import GRU
from .Transformer import Transformer
from .MLP import MLP
from .DLinear import DLinear


def model_select(name):
    name=name.upper()

    if name in ("MARKETFORMER"):
        return Marketformer
    elif name in ("GRU"):
        return GRU
    elif name in ("LSTM"):
        return LSTM
    elif name in ("MLP"):
        return MLP
    elif name in ("TRANSFORMER"):
        return Transformer
    elif name in ("DLINEAR"):
        return DLinear
    else:
        raise NotImplementedError