"""
A script to demo the different of torch.tensor and torch.nn.Parameter in torch.nn.Module
Ref: https://blog.csdn.net/qq_43391414/article/details/120484239


torch.nn.Parameter doc: https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
- A kind of Tensor that is to be considered a module parameter.
- have a very special property when used with Module s 
- When used with a Module, the Parameter automatically gets added to the list of its `parameters`
- It will appear e.g. in parameters() iterator. Assigning a Tensor doesn’t have such effect. 


torch.dot doc: https://pytorch.org/docs/stable/generated/torch.dot.html
- Computes the dot product of two 1D tensors.
"""
from operator import mod
import torch 
import torch.nn as nn 


class ModuleA(nn.Module):
    """Define a class to demo the difference between torch.tensor and torch.nn.Parameter in a nn.Module"""
    def __init__(self):
        super(ModuleA, self).__init__()
        # Directly assign a torch.tensor
        # This tensor would not be regiterd as a parameter in the Module
        self.w1 = torch.tensor([1,2], dtype=torch.float32, requires_grad=True)
        # Pack the tensor into a nn.Parameter
        # The tensor packed with `nn.Parameter` would be regiterd as a parameter of the Module
        a = torch.tensor([3,4], dtype=torch.float32)
        self.w2 = nn.Parameter(a)

    def forward(self, inputs):
        o1 = torch.dot(self.w1, inputs)
        o2 = torch.dot(self.w2, inputs)
        return o1 + o2


if __name__ == "__main__":
    
    # Initialize a ModuleA
    module = ModuleA()

    # print the list of parameters
    for p in module.parameters():
        print(p)
    inputs = torch.tensor([1,1], dtype=torch.float32)
    outputs = module(inputs)
    print(f"outputs: {outputs}")