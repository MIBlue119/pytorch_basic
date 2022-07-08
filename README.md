# pytorch_basic

A repo to record the practice of pytorch 
- [torch trainign pipeline](./notebooks/README.md)
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
    - Check the version at `https://download.pytorch.org/whl/torch_stable.html `