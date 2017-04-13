"""Utility functions for birdsonganalysis."""

import numpy as np


# TODO med and mad should be easily changeable
med = {
    'am': 0.00039675611898210353,
    'amplitude': 82.958846230933801,
    'entropy': -3.4923664803371448,
    'fm': 0.84876612812883656,
    'goodness': 0.10728456589138036,
    'pitch': 3042.5634765625}

mad = {
    'am': 3.1437898652827876,
    'amplitude': 6.4795818349700909,
    'entropy': 0.93547788888390127,
    'fm': 0.36812686183136534,
    'goodness': 0.026704864227088371,
    'pitch': 554.991455078125}


def set_med_mad(med_, mad_):
    """Set the new values of med and mad for features."""
    global med, mad
    med.update(med_)
    mad.update(mad_)


def get_windows(song, fft_step=None, fft_size=None):
    r"""
    Build the windows of the song for analysis.

    The windows are separeted by
    `fft_step` and are of the size `fft_size`. The windows are centered on the
    actual fft_step. Therefore, with default parameters, they go
    song[-400:399] (centered on 0); song[-360:439] (centered on 40); etc.
    out of range indices (including negative indices) are filled with zeros.
    For example, the first window will be
    ```
    [0 0 0 ... 0 0 0 0.48 0.89 0.21 -0.4]
     \  400 times  / ^- The first recording of the signal
    ```
    and the last window
    ```
     [0.54 0.12 -0.25 0 0 0 ... 0 0 0]
    ```
    """
    if fft_step is None:
        fft_step = 40
    if fft_size is None:
        fft_size = 1024
    song = np.array(song, dtype=np.double)
    song = 2*song / (np.max(song) - np.min(song))
    size = len(song)
    padsize = fft_size
    song = np.concatenate((np.zeros(padsize), song, np.zeros(padsize)))
    wave_smp = range(fft_step//2, size, fft_step)
    nb_windows = len(wave_smp)
    windows = np.zeros((nb_windows, fft_size))
    for i, smp in enumerate(wave_smp):
        begin = smp - fft_size//2 + padsize
        windows[i, :] = song[begin:begin + fft_size]
    return windows


def distbroad(X, Y):
    """
    Compute the squared dist between two features vector element by element.

    It computes the matrix:
    ```
    (X[0] - Y[0])**2 ; (X[0] - Y[1])**2 ; ... ; (X[0] - Y[m])**2
    (X[1] - Y[0])**2 ; (X[1] - Y[1])**2 ; ... ; (X[1] - Y[m])**2
    ...
    (X[n] - Y[0])**2 ; (X[n] - Y[1])**2 ; ... ; (X[n] - Y[m])**2
    """
    return (X[:, np.newaxis] - Y)**2


def normalize_features(song_features):
    """
    Normalize the features of a song.

    It normalizes using the median and the MAD.
    """
    adj_song_features = dict()
    for fname in song_features:
        adj_song_features[fname] = ((song_features[fname] - med[fname])
                                    / mad[fname])
        adj_song_features[fname][np.isnan(adj_song_features[fname])] = 0

    return adj_song_features


def calc_dist_features(feats1, feats2, feat_names=None):
    """Compute the distance between two dict of features."""
    if feat_names is None:
        feat_names = feats1.keys()
    out = dict()
    for fname in feat_names:
        out[fname] = distbroad(feats1[fname], feats2[fname])
    return out


def cepstrum(signal, n=None):
    """Compute the real cepstrum of a signal."""
    spectrum = np.fft.fft(signal, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real
    return ceps
