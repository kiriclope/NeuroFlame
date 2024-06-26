import torch
import numpy as np

def decode_bump_torch(signal, axis=-1):
    """
    Decode a signal to a phase and magnitude representation using PyTorch.

    Parameters:
    signal (Tensor): Input signal to decode.
    axis (int): The axis along which the operation is performed. Default is -1.

    Returns:
    tuple: Returns a tuple of three elements (m0, m1, phi) where:
             m0 is the mean of the signal,
             m1 is magnitude of the Fourier transform of the signal,
             phi is the phase of the Fourier transform of the signal.
    """

    # Ensuring the input is a Tensor
    signal_copy = signal.clone().to(torch.cfloat)

    # Swapping axes if necessary
    if axis != -1 and signal_copy.ndim != 1:
        signal_copy = signal_copy.movedim(axis, -1)

    # Calculating the mean along the specified (or last) axis
    m0 = torch.nanmean(signal_copy, dim=-1).real

    length = signal_copy.shape[-1]
    dPhi = 2.0 * torch.pi / length

    # Creating the complex exponential
    exp_vals = torch.exp(1.0j * torch.arange(length).to(signal.device) * dPhi).type(torch.cfloat)

    # Computing the discrete Fourier transform manually
    dft = signal_copy @ exp_vals

    # If the input signal had more than one dimension and was swapped, swap back
    if axis != -1 and signal_copy.ndim != 1:
        dft = dft.movedim(-1, axis)

    # Magnitude of the DFT, adjusted by the signal length
    m1 = 2.0 * torch.abs(dft) / length

    # Phase of the DFT
    phi = torch.atan2(dft.imag, dft.real) % (2 * torch.pi)

    return m0, m1, phi


def decode_bump(signal, axis=-1):
    """
    Decode a signal to a phase and magnitude representation.

    Parameters:
    signal (ndarray): Input signal to decode.
    axis (int): The axis along which the operation is performed. Default is -1.

    Returns:
    tuple: Returns a tuple of three elements (m0, m1, phi) where:
             m0 is the mean of the signal,
             m1 is magnitude of the Fourier transform of the signal,
             phi is the phase of the Fourier transform of the signal.
    """

    signal_copy = signal.copy()
    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal_copy, axis, -1)

    m0 = np.nanmean(signal_copy, -1)

    length = signal_copy.shape[-1]
    dPhi = 2.0 * np.pi / length

    dft = np.dot(signal_copy, np.exp(1.0j * np.arange(length) * dPhi))

    if axis != -1 and signal.ndim != 1:
        dft = np.swapaxes(dft, axis, -1)

    m1 = 2.0 * np.absolute(dft) / length
    phi = (np.arctan2(dft.imag, dft.real)) % (2 * np.pi)

    return m0, m1, phi


def circcvl(signal, windowSize=10, axis=-1):
    """
    Compute the circular convolution of a signal with a smooth kernel.

    Parameters:
    signal (ndarray): The input signal.
    windowSize (int): The length of the smoothing window. Defaults to 10.
    axis (int): The axis along which the operation is performed. Default is -1.

    Returns:
    ndarray: Returns the smoothed signal after circular convolution.
    """

    signal_copy = signal

    if axis != -1 and signal.ndim != 1:
        signal_copy = np.swapaxes(signal, axis, -1)

    ker = np.concatenate(
        (np.ones((windowSize,)), np.zeros((signal_copy.shape[-1] - windowSize,)))
        )

    smooth_signal = np.real(
        np.fft.ifft(
            np.fft.fft(signal_copy, axis=-1) * np.fft.fft(ker, axis=-1), axis=-1
        )
    ) * (1.0 / float(windowSize))

    if axis != -1 and signal.ndim != 1:
        smooth_signal = np.swapaxes(smooth_signal, axis, -1)

    return smooth_signal
