"""
Analysis of the song are taken from SAT: http://soundanalysispro.com/matlab-sat
"""

import numpy as np
import libtfr
from scipy.stats import gmean

def wiener_entropy(power, freq_range=1024):
    # Taken from SAT
    # Ignore low frequency power, starting only at the 10th element
    sumlog = np.sum(np.log(power[9:freq_range, :]))
    logsum = np.log(np.sum(power[9:freq_range, :])/(freq_range - 10))
    return sumlog/(freq_range - 10) - logsum

def song_similarity(song, reference_song):
    """
    Song similarity is asymmetric
    """
    raise NotImplemented

def frequence_modulation(window):
    D = libtfr.mfft_dpss(len(window), 1.5, 3)
    Z = D.mtfft(window)
    time_der = -Z[:]
