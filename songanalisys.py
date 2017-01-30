import numpy as np
from scipy.stats import gmean

def wiener_entropy(signal):
    return np.ln(gmean(signal)/np.mean(signal))

def song_similarity(song, reference_song):
    """
    Song similarity is asymmetric
    """
    raise NotImplemented
