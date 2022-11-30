"""
Some hook modules for torch nn.Module

Ref:
- How to use pytorch hooks: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
"""
import torch.nn as nn


def hook_show_module_parameters(module: nn.Module, input, output):
    """Ref:https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/"""
    print(module)
    print("---------------Module Parameters---------------")
    neles = sum([param.nelement() for param in module.parameters()])
    print(neles)
    print("---------------Module input Gradients---------------")
    for in_ in input:
        try:
            print(f"in_:{in_.shape}")
        except AttributeError:
            print("None found for Gradient")
    print("---------------Module Outputs---------------")
    for out in output:
        try:
            print(f"out shape: {out.shape}")
        except AttributeError:
            print("None found for Output")
