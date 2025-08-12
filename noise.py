import numpy as np
from scipy.signal import butter, filtfilt

# Basic Gaussian noise model for time series
# Noise should be added separately for each channel (i.e., I and Q)
# The default noise level is a current best-guess for CCA SNR
# Changing the constant will change the noise level relative to the nominal
#
# INPUTS
#    len_array: the length of the noise array desired (length of time series)
#    constant: multiplicative constant with respect to the nominal case
#
# OUTPUTS
#    Noise array of length len_array
def noise_model(len_array, constant=1): 
    sampling_freq = 403E6
    mu = 0
    R = 50 #ohms                                                                                                                                                                                                                                                                                                                                                            
    sigma = np.sqrt(R * 0.5) * np.sqrt(2.2e-13 * constant * sampling_freq) # for correct SNR, per Penny                                                                                                                                                                                                                                                                     
    return np.random.normal(mu, sigma, len_array)

def bandpass_filter(data, lowcut=0.5E8, highcut=0.8E8, order=4):
    fs = 403E6
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
