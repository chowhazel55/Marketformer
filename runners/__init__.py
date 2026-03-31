from .NASDAQRunner import NASDAQRunner

def runner_select(name):
    name = name.upper()
    if name == "BASIC2":
        return NASDAQRunner
    else:
        raise NotImplementedError
