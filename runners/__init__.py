from .NASDAQRunner import NASDAQRunner
from .CSIRunner import CSIRunner

def runner_select(name):
    name = name.upper()
    if name == "BASIC2":
        return NASDAQRunner
    if name =="CSI":
        return CSIRunner
    else:
        raise NotImplementedError
