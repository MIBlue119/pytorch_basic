# pytorch_basic

A repo to record the practice of pytorch 
- [torch trainig pipeline](./notebooks/torch_pipeline/README.md)
- [stft transform practice](./transform/README.md): Use a `torch.nn.Module` to construct a torch class with STFT and ISTFT for audio processing.

## Installation

```
$ pip install -r requirements.txt
```

## Others 
- Install the torch/torchaudio with cuda version
```
$pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
```    
- Check the torch version at https://download.pytorch.org/whl/torch_stable.html

## Resources
- [pytorch style guide](https://github.com/IgorSusmelj/pytorch-styleguide)
- [dive into deeplearning with pyotrch](https://github.com/d2l-ai/d2l-en)