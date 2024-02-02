import numpy as np


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
    dPhi = np.pi / length

    dft = np.dot(signal_copy, np.exp(-2.0j * np.arange(length) * dPhi))

    if axis != -1 and signal.ndim != 1:
        dft = np.swapaxes(dft, axis, -1)
    
    m1 = 2.0 * np.absolute(dft) / length
    phi = np.arctan2(dft.imag, dft.real) % (2 * np.pi)
    
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
