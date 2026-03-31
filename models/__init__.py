
from .Marketformer import Marketformer


def model_select(name):
    name=name.upper()

    if name in ("MARKETFORMER"):
        return Marketformer
    else:
        raise NotImplementedError