"""
Analyses the features of bird song.

Analysis of the song are taken from SAT: http://soundanalysispro.com/matlab-sat
"""

import json
import os
from collections import defaultdict

import numpy as np
from aubio import pitch as aubio_pitch

import libtfr

from .utils import get_windows, cepstrum

EPS = np.finfo(np.double).eps
dir_path = os.path.dirname(os.path.realpath(__file__))

def get_power(signal):
    """
    Return the power of the signal.

    This method uses multitaper fast fourier transform with two tapers as
    it is commonly done with bird song analysis.
    """
    D = libtfr.mfft_dpss(len(signal), 1.5, 2, len(signal))
    P = D.mtpsd(signal)
    return P


def get_mtfft(signal):
    """Return the two multitaper FFT of the signal."""
    D = libtfr.mfft_dpss(len(signal), 1.5, 2, len(signal))
    return D.mtfft(signal)


def time_der(Z, freq_range):
    """
    Compute the time derivative of a signal.

    It computes only one value of the time derivative. Z is supposed to be from
    a windowed signal.
    Z - The MTFFTs of the signal window computed with `get_mtfft`.
    """
    return -Z[:freq_range, 0].real * Z[:freq_range, 1].real \
        - Z[:freq_range, 0].imag * Z[:freq_range, 1].imag


def freq_der(Z, freq_range):
    """
    Compute the frequency derivative of a signal.

    Z - The MTFFTs of the signal window computed with `get_mtfft`.
    """
    return Z[:freq_range, 0].imag * Z[:freq_range, 1].real \
        - Z[:freq_range, 0].real * Z[:freq_range, 1].imag


def wiener_entropy(power, freq_range=None):
    """
    Compute the wiener entropy of a signal.

    power - The power of the signal computed with `get_power`

    It computes only one value of the wiener entropy. Usually, you want to
    computes the Wiener entropy of a whole song using `song_wiener_entropy`.
    """
    # Taken from SAT
    # Ignore low frequency power, starting only at the 10th element
    if freq_range is None:
        freq_range = 256
    if np.all(power == 0):
        return 0
    sumlog = np.sum(np.log(power[9:freq_range]))
    logsum = np.log(np.sum(power[9:freq_range])/(freq_range - 10))
    return sumlog/(freq_range - 10) - logsum


def amplitude_modulation(window, freq_range=None):
    """
    Compute the amplitude modulation of a signal.

    window - The signal or the MTFFTs of the signal

    It computes only one value of the amplitude modulation. Usually, you want
    to computes the Wiener entropy of a whole song using
    `song_amplitude_modulation`.
    """
    if freq_range is None:
        freq_range = 256
    if window.ndim == 1:
        Z = get_mtfft(window)
    else:
        Z = window
    td = time_der(Z, freq_range)
    return np.sum(td)


def frequency_modulation(window, freq_range=None):
    """
    Compute the frequency modulation of a signal.

    window - The signal or the MTFFTs of the signal

    It computes only one value of the frequency modulation. Usually, you want
    to computes the frequency modulation of a whole song using
    `song_frequency_modulation`.
    """
    if freq_range is None:
        freq_range = 256
    if window.ndim == 1:
        Z = get_mtfft(window)
    else:
        Z = window
    td = time_der(Z, freq_range)
    fd = freq_der(Z, freq_range)
    return np.arctan(np.max(td) / (np.max(fd)+EPS))


def amplitude(power, freq_range=None):
    """
    Compute the amplitude of a signal.

    power - The power of the signal computed by `get_power`.

    It computes only one value of the amplitude. Usually, you want
    to computes the amplitude of a whole song using
    `song_amplitude`.
    """
    if freq_range is None:
        freq_range = 256
    logsum = np.sum(power[9:freq_range])
    if logsum > 0:
        return 10*(np.log10(logsum)+7)
    else:
        return 0


def goodness(signal, freq_range=None, D=None):
    """Compute the goodness of pitch of a signal."""
    if D is None:
        D = libtfr.dpss(len(signal), 1.5, 1)[0]
    signal = signal * D[0, :]
    if freq_range is None:
        freq_range = 256
    if np.all(signal == 0):
        return 0
    else:
        return np.max(cepstrum(signal)[25:freq_range])


def spectral_derivs(song, freq_range=None, fft_step=None, fft_size=None):
    """
    Return the spectral derivatives of a song.

    The spectral derivatives are usefull to have a nice representation of
    the song. To get this plot, use `spectral_derivs_plot`.
    """
    if freq_range is None:
        freq_range = 256
    windows = get_windows(song, fft_step, fft_size)
    nb_windows = windows.shape[0]
    td = np.zeros((nb_windows, freq_range))
    fd = np.zeros((nb_windows, freq_range))
    fm = np.zeros(nb_windows)
    D = libtfr.mfft_dpss(windows.shape[1], 1.5, 2, windows.shape[1])
    for i, window in enumerate(windows):
        Z = D.mtfft(window)
        td[i, :] = time_der(Z, freq_range)
        fd[i, :] = freq_der(Z, freq_range)
        fm[i] = np.arctan(np.max(td[i, :]) / (np.max(fd[i, :]) + EPS))  # TODO vectorize
    cfm = np.cos(fm)
    sfm = np.sin(fm)
    return cfm[:, np.newaxis] * td + sfm[:, np.newaxis] * fd


def song_frequency_modulation(song, freq_range=None, fft_step=None,
                              fft_size=None):
    """Return the whole song frequency modulations array."""
    if freq_range is None:
        freq_range = 256
    windows = get_windows(song, fft_step, fft_size)
    nb_windows = windows.shape[0]
    td = np.zeros((nb_windows, freq_range))
    fd = np.zeros((nb_windows, freq_range))
    fm = np.zeros(nb_windows)
    for i, window in enumerate(windows):
        Z = get_mtfft(window)
        td[i, :] = time_der(Z, freq_range)
        fd[i, :] = freq_der(Z, freq_range)
        fm[i] = np.arctan(np.max(td[i, :]) / (np.max(fd[i, :]) + EPS))  # TODO vectorize
    return fm


def song_amplitude(song, freq_range=None, fft_step=None, fft_size=None):
    """Return an array of amplitude values for the whole song."""
    windows = get_windows(song, fft_step, fft_size)
    nb_windows = windows.shape[0]
    amp = np.zeros(nb_windows)
    D = libtfr.mfft_dpss(windows.shape[1], 1.5, 2, windows.shape[1])
    for i, window in enumerate(windows):
        P = D.mtpsd(window)
        amp[i] = amplitude(P, freq_range)
    return amp


def song_pitch(song, sr, threshold=None, freq_range=None, fft_step=None,
               fft_size=None):
    """Return an array of pitch values for the whole song."""
    if threshold is None:
        threshold = 0.8
    windows = get_windows(song, fft_step, fft_size)
    nb_windows = windows.shape[0]
    win_s = windows.shape[1]
    pitch_o = aubio_pitch("yin", 2048, win_s, sr)
    pitch_o.set_unit("freq")
    pitch_o.set_tolerance(threshold)
    pitches = np.zeros(nb_windows)
    for i, window in enumerate(windows):
        pitches[i] = pitch_o(window.astype(np.float32))
    pitches[pitches > sr / 2] = 0
    return pitches


def song_wiener_entropy(song, freq_range=None, fft_step=None, fft_size=None):
    """Return an array of wiener entropy values for the whole song."""
    if freq_range is None:
        freq_range = 256
    windows = get_windows(song, fft_step, fft_size)
    wiener = np.zeros(windows.shape[0])
    for i, window in enumerate(windows):
        P = get_power(window)
        wiener[i] = wiener_entropy(P, freq_range)
    return wiener


def song_amplitude_modulation(song, freq_range=None, fft_step=None,
                              fft_size=None):
    """Return an array of amplitude modulation for the whole song."""
    if freq_range is None:
        freq_range = 256
    windows = get_windows(song, fft_step, fft_size)
    am = np.zeros(windows.shape[0])
    for i, window in enumerate(windows):
        am[i] = amplitude_modulation(window)
    return am


def song_goodness(song, freq_range=None, fft_step=None, fft_size=None):
    """Return an array of goodness of pitch for the whole song."""
    if freq_range is None:
        freq_range = 256
    windows = get_windows(song, fft_step, fft_size)
    good = np.zeros(windows.shape[0])
    for i, window in enumerate(windows):
        good[i] = goodness(window, freq_range)
    return good


def all_song_features(song, sr, pitch_method=None,
                      pitch_threshold=None, freq_range=None,
                      fft_step=None, fft_size=None):
    """Return all the song features in a `dict`."""
    windows = get_windows(song, fft_step, fft_size)
    out = defaultdict(lambda: np.zeros(windows.shape[0], dtype=float))
    D = libtfr.mfft_dpss(windows.shape[1], 1.5, 2, windows.shape[1])
    for i, window in enumerate(windows):
        Z = D.mtfft(window)
        P = D.mtpsd(window)
        out['goodness'][i] = goodness(window, freq_range, D=D.tapers)
        out['am'][i] = amplitude_modulation(Z, freq_range)
        out['fm'][i] = frequency_modulation(Z, freq_range)
        out['amplitude'][i] = amplitude(P, freq_range)
        out['entropy'][i] = wiener_entropy(P, freq_range)
        if pitch_method == 'fft':
            out['pitch'][i] = np.argmax(P[0:freq_range])/len(P) * sr
    if pitch_method != 'fft':
        out['pitch'] = song_pitch(song, sr, pitch_threshold)
    else:
        silent_th = np.percentile(out['amplitude'], 20)
        out['pitch'][out['amplitude'] < silent_th] = 0
    return out
