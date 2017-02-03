"""
Analysis of the song are taken from SAT: http://soundanalysispro.com/matlab-sat
"""

import numpy as np
import libtfr
from scipy.stats import gmean
import json
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
params = json.load(open(os.path.join(dir_path, 'params.json')))

def get_power(signal):
    D = libtfr.mfft_dpss(len(signal), 1.5, 3, len(signal))
    P = D.mtpsd(signal)
    return P

def wiener_entropy(power, freq_range=None):
    # Taken from SAT
    # Ignore low frequency power, starting only at the 10th element
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    sumlog = np.sum(np.log(power[9:freq_range]))
    logsum = np.log(np.sum(power[9:freq_range])/(freq_range - 10))
    return sumlog/(freq_range - 10) - logsum

def frequence_modulation(window):
    D = libtfr.mfft_dpss(len(window), 1.5, 3)
    Z = D.mtfft(window)
    time_der = -Z[:freq_range, 0].real * Z[:freq_range, 1].real \
                - Z[:freq_range, 0].imag * Z[:freq_range, 1].imag
    freq_der = Z[:freq_range, 0].imag * Z[:freq_range, 1].real \
                - Z[:freq_range, 0].real * Z[:freq_range, 1].imag
    return np.atan(np.max(time_der) / np.max(freq_der))

def amplitude_modulation(power, freq_range=None):
    if freq_range is None:
        freq_range=np.floor(params['FFT']*params['Frequency_range']/2)
    logsum = np.log(np.sum(power[9:freq_range, :])/(freq_range - 10))
    amplitude = 10*(np.log10(logsum)+7);
