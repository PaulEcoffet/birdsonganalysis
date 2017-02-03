"""
Analysis of the song are taken from SAT: http://soundanalysispro.com/matlab-sat
"""

import numpy as np
import libtfr
from scipy.stats import gmean

import copy
import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
params = json.load(open(os.path.join(dir_path, 'params.json')))

def get_power(signal):
    D = libtfr.mfft_dpss(len(signal), 1.5, 2, len(signal))
    P = D.mtpsd(signal)
    return P

def get_mtfft(window):
    D = libtfr.mfft_dpss(len(window), 1.5, 2, len(window))
    return D.mtfft(window)

def time_der(Z, freq_range):
    return -Z[:freq_range, 0].real * Z[:freq_range, 1].real \
                - Z[:freq_range, 0].imag * Z[:freq_range, 1].imag

def freq_der(Z, freq_range):
    return Z[:freq_range, 0].imag * Z[:freq_range, 1].real \
                - Z[:freq_range, 0].real * Z[:freq_range, 1].imag

def wiener_entropy(power, freq_range=None):
    # Taken from SAT
    # Ignore low frequency power, starting only at the 10th element
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    sumlog = np.sum(np.log(power[9:freq_range]))
    logsum = np.log(np.sum(power[9:freq_range])/(freq_range - 10))
    return sumlog/(freq_range - 10) - logsum

def frequence_modulation(window=None, freq_range=None):
    """
    """
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    if window.ndim == 1:
        Z = get_mtfft(window)
    else:
        Z = window
    td = time_der(Z, freq_range)
    fd = freq_der(Z, freq_range)
    return np.arctan(np.max(td) / (np.max(fd)+np.finfo(np.double).eps))

def amplitude_modulation(power, freq_range=None):
    if freq_range is None:
        freq_range = int(params['FFT']*params['Frequency_range']/2)
    logsum = np.log(np.sum(power[9:freq_range, :])/(freq_range - 10))
    amplitude = 10*(np.log10(logsum)+7);

def spectral_derivs(song, freq_range=None, **ov_params):
    song = np.array(song, dtype=np.double)
    song = song / (np.max(song) - np.min(song))
    p = copy.deepcopy(params)
    p.update(ov_params)
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    fft_size = p['FFT_size']*2
    fft_step = p['FFT_step']
    nb_windows = int((len(song) - fft_size) // fft_step)
    td = np.zeros((nb_windows, freq_range))
    fd = np.zeros((nb_windows, freq_range))
    fm = np.zeros(nb_windows)
    for i in range(0, nb_windows-1):
        j = i * fft_step
        window = song[j:j+fft_size]
        Z = get_mtfft(window)
        td[i, :] = time_der(Z, freq_range)
        fd[i, :] = freq_der(Z, freq_range)
        fm[i] = np.arctan(np.max(td[i, :]) / np.max(fd[i, :]))  # TODO vectorize
    cfm = np.cos(fm)
    sfm = np.sin(fm)
    return cfm[:, np.newaxis] * td  + sfm[:, np.newaxis] * fd
