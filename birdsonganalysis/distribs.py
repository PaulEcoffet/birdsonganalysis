"""
This module computes G² and L² distributions.

To compute similarity, a distribution of errors must be built with unrelated
songs.

Beware, this is a very computationally expensive process.
"""

import numpy as np
import itertools as it

from birdsonganalysis.utils import get_windows, normalize_features, \
                                   calc_dist_features
from birdsonganalysis.songfeatures import all_song_features

def get_distribs(songs, samplerate=44100, T=70):
    allG = np.array([0], dtype=float)
    allL = np.array([0], dtype=float)
    for song, refsong in it.combinations(songs, 2):
        song_win = get_windows(song)
        refsong_win = get_windows(refsong)
        song_features = all_song_features(song, samplerate,
                                          without="amplitude")
        refsong_features = all_song_features(refsong, samplerate,
                                             without="amplitude")
        adj_song_features = normalize_features(song_features)
        adj_refsong_features = normalize_features(refsong_features)
        #################################
        # Compute the L matrix (step 3) #
        #################################
        # L2 = L²
        local_dists = calc_dist_features(adj_song_features,
                                         adj_refsong_features)
        L2 = np.mean(
            np.array([local_dists[fname] for fname in local_dists.keys()]),
            axis=0)
        # avoid boundaries effect
        # maxL2 = np.max(L2)
        # L2[:T//2, :] = maxL2
        # L2[-(T//2):, :] = maxL2
        # L2[:, :T//2] = maxL2
        # L2[:, -(T//2):] = maxL2
        G2 = np.zeros((song_win.shape[0], refsong_win.shape[0]))  # G2 = G²
        #############################
        # Compute G Matrix (step 4) #
        #############################
        for i in range(song_win.shape[0]):
            for j in range(refsong_win.shape[0]):
                imin = max(0, (i-T//2))
                imax = min(G2.shape[0], (i+T//2))
                jmin = max(0, (j-T//2))
                jmax = min(G2.shape[1], (j+T//2))
                G2[i, j] = np.mean(L2[imin:imax, jmin:jmax])
        allG = np.concatenate((allG, G2.flatten()))
        allL = np.concatenate((allL, L2.flatten()))
    return allG, allL
