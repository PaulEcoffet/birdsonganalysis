"""
This module computes G² and L² distributions.

To compute similarity, a distribution of errors must be built with unrelated
songs.

Beware, this is a very computationally expensive process.
"""

import numpy as np
import itertools as it
import time
import datetime

from birdsonganalysis.utils import get_windows, normalize_features, \
                                   calc_dist_features
from birdsonganalysis.songfeatures import all_song_features, song_amplitude


def get_distribs(songs, samplerate=44100, T=70, verbose=True):
    allG = np.array([], dtype=float)
    allL = np.array([], dtype=float)
    total = len(list(it.combinations(songs, 2)))
    start = time.time()
    for i, (song, refsong) in enumerate(it.combinations(songs, 2)):
        print("{}/{}".format(i+1, total))
        if i >= 1:
            elapsed = time.time() - start
            print('elapsed: {}, Total: {}'.format(
                datetime.timedelta(seconds=elapsed),
                datetime.timedelta(seconds=(elapsed / i * total))))
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
                if not np.all(np.isnan(np.diag(L2[imin:imax, jmin:jmax]))):
                    G2[i, j] = np.nanmean(np.diag(L2[imin:imax, jmin:jmax]))
                else:
                    G2[i, j] = np.nan
        amp_song = song_amplitude(song)
        threshold = np.percentile(amp_song, 15)
        L2[amp_song < threshold, :] = np.nan
        G2[amp_song < threshold, :] = np.nan
        amp_song = song_amplitude(refsong)
        threshold = np.percentile(amp_song, 15)
        L2[:, amp_song < threshold] = np.nan
        G2[:, amp_song < threshold] = np.nan
        allG = np.concatenate((allG, G2.flatten()))
        allL = np.concatenate((allL, L2.flatten()))
    return allG[~ np.isnan(allG)], allL[~ np.isnan(allL)]
