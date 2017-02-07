"""
Analysis of the song are taken from SAT: http://soundanalysispro.com/matlab-sat
"""

import numpy as np
import libtfr
from scipy.stats import gmean
from aubio import pitch

import copy
import json
import os

EPS = np.finfo(np.double).eps
dir_path = os.path.dirname(os.path.realpath(__file__))
params = json.load(open(os.path.join(dir_path, 'params.json')))

def get_power(signal, n):
    size = len(signal)
    if size < n:
        signal = np.concatenate((signal, np.zeros(n - len(signal))))
    signal = signal[:n]
    D = libtfr.mfft_dpss(size, 1.5, 2, size)
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
    if np.all(power == 0):
        return 0
    sumlog = np.sum(np.log(power[9:freq_range]))
    logsum = np.log(np.sum(power[9:freq_range])/(freq_range - 10))
    return sumlog/(freq_range - 10) - logsum

def frequency_modulation(window=None, freq_range=None):
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
    return np.arctan(np.max(td) / (np.max(fd)+EPS))

def amplitude_modulation(power, freq_range=None):
    if freq_range is None:
        freq_range = int(params['FFT']*params['Frequency_range']/2)
    logsum = np.sum(power[9:freq_range])
    if logsum > 0:
        return 10*(np.log10(logsum)+7)
    else:
        return 0

def spectral_derivs(song, freq_range=None, ov_params=None):
    song = np.array(song, dtype=np.double)
    song = song / (np.max(song) - np.min(song))
    p = copy.deepcopy(params)
    if ov_params is not None:
        p.update(ov_params)
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    fft_size = p['FFT_size']*2
    fft_step = p['FFT_step']
    song = np.concatenate((np.zeros(fft_size), song, np.zeros(fft_size)))
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
        fm[i] = np.arctan(np.max(td[i, :]) / (np.max(fd[i, :]) + EPS))  # TODO vectorize
    cfm = np.cos(fm)
    sfm = np.sin(fm)
    return cfm[:, np.newaxis] * td  + sfm[:, np.newaxis] * fd


def song_frequency_modulation(song, freq_range=None, ov_params=None):
        song = np.array(song, dtype=np.double)
        song = song / (np.max(song) - np.min(song))
        p = copy.deepcopy(params)
        if ov_params is not None:
            p.update(ov_params)
        if freq_range is None:
            freq_range=int(params['FFT']*params['Frequency_range']/2)
        fft_size = p['FFT_size']*2
        fft_step = p['FFT_step']
        song = np.concatenate((np.zeros(fft_size), song, np.zeros(fft_size)))
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
            fm[i] = np.arctan(np.max(td[i, :]) / (np.max(fd[i, :]) + EPS))  # TODO vectorize
        return fm

def song_amplitude_modulation(song, freq_range=None, ov_params=None):
    song = np.array(song, dtype=np.double)
    song = song / (np.max(song) - np.min(song))
    p = copy.deepcopy(params)
    if ov_params is not None:
        p.update(ov_params)
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    fft_size = p['FFT_size']*2
    fft_step = p['FFT_step']
    song = np.concatenate((np.zeros(fft_size), song, np.zeros(fft_size)))
    nb_windows = int((len(song) - fft_size) // fft_step)
    am = np.zeros(nb_windows)
    for i in range(0, nb_windows-1):
        j = i * fft_step
        window = song[j:j+fft_size]
        P = get_power(window, p['FFT'])
        am[i] = amplitude_modulation(P, freq_range)
    return am

def song_pitch(song, sr, threshold=0.7, freq_range=None, ov_params=None):
    song = np.array(song, dtype=np.float32)
    song = song / (np.max(song) - np.min(song))
    p = copy.deepcopy(params)
    if ov_params is not None:
        p.update(ov_params)
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    tolerance = 0.8
    win_s = p['FFT_size']*4
    step = p['FFT_step']
    pitch_o = pitch("yin", 2048, win_s, sr)
    pitch_o.set_unit("freq")
    pitch_o.set_tolerance(tolerance)
    song = np.concatenate((np.zeros(win_s), song, np.zeros(win_s)))
    nb_windows = int((len(song) - win_s) // step)
    pitches = np.zeros(nb_windows)
    for i in range(0, nb_windows-1):
        j = i * step
        window = song[j:j+win_s].astype(np.float32)
        pitches[i] = pitch_o(window)[0]
    return pitches

def song_wiener_entropy(song, freq_range=None, ov_params=None):
    if freq_range is None:
        freq_range=int(params['FFT']*params['Frequency_range']/2)
    windows = get_windows(song)
    wiener = np.zeros(windows.shape[0])
    for i, window in enumerate(windows):
        P = get_power(window, params['FFT'])
        wiener[i] = wiener_entropy(P, freq_range)
    return wiener

def get_windows(song, ov_params=None):
    song = np.array(song, dtype=np.double)
    song = 2*song / (np.max(song) - np.min(song))
    p = copy.deepcopy(params)
    if ov_params is not None:
        p.update(ov_params)
    fft_size = p['FFT_size']
    fft_step = p['FFT_step']
    size = len(song)
    padsize = fft_size
    song = np.concatenate((np.zeros(padsize), song, np.zeros(padsize)))
    wave_smp = range(fft_step//2, size, fft_step)
    nb_windows = len(wave_smp)
    windows = np.zeros((nb_windows, fft_size))
    for i, smp in enumerate(wave_smp):
        begin = smp - fft_size//2 + padsize
        windows[i, :] = song[begin:begin+ fft_size]
    return windows
