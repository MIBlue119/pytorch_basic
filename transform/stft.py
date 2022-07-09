"""Define  STFT/ ISTFT class."""
from importlib.metadata import requires
from typing import Optional, Tuple, Union
import torch 
from torch import Tensor 
import torch.nn as nn 

class STFT(nn.Module):
    def __init__(
        self, 
        n_fft: int = 512,         
        win_length: int = None, 
        hop_length: int = 128,
        window: Optional[nn.Parameter] = None,
        center: bool = True
    ):
        super(STFT, self).__init__()
        self.window = window 
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length 
        self.hop_length = hop_length

        if window is None: 
            self.window = nn.Parameter(
                # Defualt window is a hann window: https://pytorch.org/docs/stable/generated/torch.hann_window.html
                torch.hann_window(
                    window_length = self.win_length,
                    requires_grad = False
                )
            )
        else:
            self.window = window

        # Define whether to pad input on both sides so that the ttt-th frame is centered at time 
        self.center = center

    def forward(self, x: Tensor):
        """STFT forward path
        Args:
            x (Tensor): audio waveform of shape (batch_size, n_frames, n_channels)
        
        Returns:
            output: (batch_size, n_channels,n_freq,n_frames)
        """    
        # Extract the shape of the input signal
        shape = x.size()

        # Unwrap the shape 
        nb_batches, nb_channels, nb_samples = shape 

        # Use torch.view to reshape the input signal to pack the batch dimension
        # Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        # view(-1) mean the shape would be inferred from the shape of the input
        # Example: 
        # x = torch.randn(1, 2, 16000)
        # shape = x.shape
        # x.view(-1, shape[-1]) -> shape size = (2, 16000)  
        x = x.view(-1, shape[-1])

        # torch.stft doc: https://pytorch.org/docs/stable/generated/torch.stft.html
        # Input must be either a 1-D time sequence or a 2-D batch of time sequences.
        
        # shape of input x: (batch_size, n_samples)
        comlex_stft = torch.stft(
            input = x, 
            n_fft = self.n_fft, 
            hop_length = self.hop_length,
            window = self.window,
            center = self.center,
            normalized = False,
            onesided = True,    #  Controls whether to return half of results to avoid redundancy for real inputs.
            pad_mode = "reflect",
            return_complex = True
        )
        # shape of complex_stft: (batch_size, n_freq, n_frames)
        # Use `torch.view_as_real` to convert the complex tensor to real tensor
        # where the last dimension of size 2 represents the real and imaginary components of complex numbers.
        # https://pytorch.org/docs/stable/generated/torch.view_as_real.html
        real_tensor_stft = torch.view_as_real(comlex_stft)
        # real_tensor_stft shape: (batch_size, n_freq, n_frames, 2=real+imag)

        # Unpack the batch 
        # Combine the shape : torch.size(nb_batches, nb_channels)+ torch.size(n_freq, n_frames, 2=real+imag)
        output = output.view(shape[:-1], real_tensor_stft.shape[-3:])
        # shape of output: (nb_batches, nb_channels, n_freq, n_frames, 2=real+imag)

        return output


