"""Define  STFT/ ISTFT class."""
from typing import Optional, Tuple, Union
import torch 
from torch import nn.Module 

class STFT(nn.Module):
    def __init__(
        self, 
        window: Optional[str] = 'hann', 
        fft_size: int = 512,         
        window_size: int = None, 
        hop_size: int = 128
    ):
        super(STFT, self).__init__()
        self.window = window 
        self.fft_size = fft_size
        if window_size is None:
            self.window_size = fft_size
        else:
            self.window_size = window_size 
        self.hop_size = hop_size
    

    def forward(self, x):