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
