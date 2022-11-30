"""Define  STFT/ ISTFT class."""
import os
import pathlib
import sys
from importlib.metadata import requires
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

path_of_utilities = pathlib.Path(__file__).parent.parent.absolute() / "utilities"
print(path_of_utilities)
sys.path.append(str(path_of_utilities))
# from hooks import hook_show_module_parameters


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[nn.Parameter] = None,
        center: bool = True,
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
                torch.hann_window(window_length=self.win_length, requires_grad=False)
            )
        else:
            self.window = window

        # Define whether to pad input on both sides so that the ttt-th frame is centered at time
        self.center = center

    def forward(self, x: Tensor):
        """STFT forward path
        Args:
            x (Tensor): audio waveform of shape (batch_size, n_channels, n_samples)

        Returns:
            output: (batch_size, n_channels,n_freq,n_frames,2=real+imag)
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
        # print(x.shape)

        # torch.stft doc: https://pytorch.org/docs/stable/generated/torch.stft.html
        # Input must be either a 1-D time sequence or a 2-D batch of time sequences.

        # shape of input x: (batch_size, n_samples)
        comlex_stft = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,  # Controls whether to return half of results to avoid redundancy for real inputs.
            pad_mode="reflect",
            return_complex=True,
        )
        # shape of complex_stft: (batch_size, n_freq, n_frames)
        # Use `torch.view_as_real` to convert the complex tensor to real tensor
        # where the last dimension of size 2 represents the real and imaginary components of complex numbers.
        # https://pytorch.org/docs/stable/generated/torch.view_as_real.html
        real_tensor_stft = torch.view_as_real(comlex_stft)
        # real_tensor_stft shape: (batch_size, n_freq, n_frames, 2=real+imag)

        # Unpack the batch
        # Combine the shape : torch.size(nb_batches, nb_channels)+ torch.size(n_freq, n_frames, 2=real+imag)
        # output = real_tensor_stft.view(shape[:-1], real_tensor_stft.shape[-3:])
        output = real_tensor_stft.view(
            -1,
            nb_channels,
            real_tensor_stft.size(-3),
            real_tensor_stft.size(-2),
            real_tensor_stft.size(-1),
        )
        # shape of output: (nb_batches, nb_channels, n_freq, n_frames, 2=real+imag)

        return output


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[nn.Parameter] = None,
        center: bool = True,
    ):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = self.n_fft
        else:
            self.win_length = win_length

        self.hop_length = hop_length

        if window is None:
            self.window = nn.Parameter(
                torch.hann_window(window_length=self.win_length, requires_grad=False)
            )
        else:
            self.window = window

        # Define whether to pad input on both sides so that the ttt-th frame is centered at time
        self.center = center

    def forward(self, X: Tensor, length: Optional[int] = None):
        """ISTFT forward path
        Args:
            X (Tensor): complex stft  shape (batch_size, n_channels, n_freq, n_frames, 2=real+imag)
        """
        # Extract the shape of the complex stft
        shape = X.size()

        # torch.istft document: https://pytorch.org/docs/stable/generated/torch.istft.html#torch.istft
        # The expected tensor shape can either be complex( n_channels, n_freq, n_frame), or real(n_channels, n_freq, n_frame)
        # Reshape the input complex stft to desired shape
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

        # Convert it back to complex
        # https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
        X_complex = torch.view_as_complex(X)

        reconstruct_signal = torch.istft(
            X_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )

        # Reshape the shape to (batch_size, n_channels, n_samples)
        reconstruct_signal = reconstruct_signal.reshape(
            shape[:-3] + reconstruct_signal.shape[-1:]
        )

        return reconstruct_signal


if __name__ == "__main__":
    import torchaudio

    file_path = (
        pathlib.Path(__file__).parent.parent.absolute()
        / "sounds"
        / "speech-female-44.1k.wav"
    )
    input_tensor, fs = torchaudio.load(file_path)
    # The shape of the audio load by torchaudio: (n_channels, n_samples)
    print(input_tensor.shape)
    # Extend the shape with batch size
    input_tensor = input_tensor[None]
    print(input_tensor.shape)

    # Initiate a tensor with shape (batch_size, n_channels, n_samples)
    # fs = 16000
    # input_tensor = torch.randn(1, 2, 16000)

    stft = STFT()
    # Iterate over the children of the module to register the forward hook
    # for i in stft.children():
    #     print(i.register_forward_hook(hook_show_module_parameters))
    output = stft(input_tensor)

    # Plot the spectrogram
    print("Shape of spectrogram: ", output.shape)

    nb_batches, nb_channels, n_freq, n_frames, _ = output.shape

    # Extract the time period series of the output frames
    frame_time = stft.hop_length * np.arange(n_frames) / float(fs)
    # Extract the frequency series of the output frequncy bins
    bin_freq = np.arange(output[0, 0, :, 0, 0].size(0)) * float(fs) / stft.n_fft

    # Extract the real/image part of the spectrogram
    real_part = output[0, 0, :, :, 0]
    imag_part = output[0, 0, :, :, 1]

    # Calculate the magnitude of the spectrogram
    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    # Calculate the phase of the spectrogram
    phase = torch.atan2(imag_part, real_part)

    # Convert the magnitude to dB
    magnitude_db = 20 * torch.log10(magnitude)

    plt.pcolormesh(frame_time, bin_freq, magnitude_db.detach().numpy())

    # Set the x-axis label
    plt.xlabel("Time [s]")
    # Set the y-axis label
    plt.ylabel("Frequency [Hz]")

    plt.title("Spectorgram of the STFT")
    plt.autoscale(tight=True)

    plt.show()

    # Initiate a ISTFT
    istft = ISTFT()
    reconstructded_signal = istft(output)

    # Reshape the reconstructded_signal to shape (n_channels, n_samples)
    reconstructded_signal = reconstructded_signal[0, ...]

    # Export the recontructed signal
    # https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/7303ce3181f4dbc9a50bc1ed5bb3218f/audio_preprocessing_tutorial.ipynb#scrollTo=jsWAdN3JYDeC
    exported_file_path = "reconstucted.wav"
    torchaudio.save(exported_file_path, reconstructded_signal, fs)
