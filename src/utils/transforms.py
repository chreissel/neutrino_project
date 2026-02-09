import numpy as np
from scipy.fft import fft, fftfreq

def fft_and_freq(signal, T, w=1):
    N = len(signal)
    yf = fft(signal*w)
    xf = fftfreq(N, T)[:N//2]
    return xf[1:N//2], 2.0/N * np.abs(yf[1:N//2])

def fft_func(signal):
    N = len(signal)
    yf = fft(signal)
    return 2.0/N * np.abs(yf[1:N//2])

def fft_func_IQ_abs(signal_I, signal_Q):
    signal = signal_I + 1j*signal_Q
    N = len(signal)
    yf = fft(signal)
    return np.abs(yf)

def fft_func_IQ_complex_channels(signal_I, signal_Q):
    signal = signal_I + 1j*signal_Q
    N = len(signal)
    yf = fft(signal)
    real_part = yf.real
    imag_part = yf.imag
    return real_part, imag_part

def q_transform_fast(signal, fs, fmin, fmax, q_value=5.0, num_freqs=100):
    
    n = len(signal)
    times = np.arange(n) / fs
    frequencies = np.logspace(np.log10(fmin), np.log10(fmax), num_freqs)

    q_transform_result = np.zeros((n, num_freqs), dtype=complex)

    for i, f0 in enumerate(frequencies):
        sigma_t = q_value / (2 * np.pi * f0)
        window_samples = int(6 * sigma_t * fs)
        window_samples = min(window_samples, n)

        # Create wavelet
        t_window = np.arange(-window_samples//2, window_samples//2) / fs
        window = np.exp(-0.5 * (t_window / sigma_t)**2)
        window /= np.sqrt(np.sum(window**2))
        carrier = np.exp(-2j * np.pi * f0 * t_window)
        wavelet = window * carrier

        # FFT-based convolution
        signal_fft = fft(signal, n=n + len(wavelet) - 1)
        wavelet_fft = fft(np.conj(wavelet[::-1]), n=n + len(wavelet) - 1)
        convolved = ifft(signal_fft * wavelet_fft)

        # Extract valid portion
        q_transform_result[:, i] = convolved[len(wavelet)//2:len(wavelet)//2 + n]

    return q_transform_result, frequencies, times


def qtransform_func_IQ_complex_channels(ts_I, ts_Q, fs=200e6, fmin=1e6, fmax=100e6, 
                                        q_value=5.0, num_freqs=100):
    complex_signal = ts_I + 1j * ts_Q
    q_result, frequencies, times = q_transform_fast(
        complex_signal, fs, fmin, fmax, q_value, num_freqs
    )
    real_part = np.real(q_result).flatten()
    imag_part = np.imag(q_result).flatten()
    
    return real_part, imag_part
