"""Compute the similarity between two songs."""

import numpy as np

import .songfeatures as sf


def pdist2(X, Y):
    """
    Compute the distance between two matrices.

    Quick'n'dirty implementation of matlab pdist2, which has no equivalent in
    numpy/scipy.
    """
    out = np.zeros((X.shape[0], Y.shape[0]))
    for i, rowX in enumerate(X):
            out[i] = np.linalg.norm(Y - rowX, axis=1)
    return out


def similarity(song, refsong):
    """
    Compute similarity between song.

    Compute the similarity between the song `song` and a reference song
    `refsong` using the method described in Tchernichovski, Nottebohm,
    Ho, Pesaran, & Mitra (2000).

    ### References:

    Tchernichovski, O., Nottebohm, F., Ho, C. E., Pesaran, B., & Mitra,
    P. P. (2000). A procedure for an automated measurement of song similarity.
    Animal Behaviour, 59(6), 1167–1176. https://doi.org/10.1006/anbe.1999.1416
    """
    feature_names = ['am', 'fm', 'pitch', 'entropy', 'amplitude']

    song_win = get_windows(song)
    refsong_win = get_windows(song)
    L = np.zeros((song_win.shape[0], refsong_win.shape[0]))
    G = np.zeros((song_win.shape[0], refsong_win.shape[0]))

    song_features = all_song_features(song)
    refsong_features = all_song_features(refsong)
    adj_song_features = dict()
    adj_refsong_features = dict()
    L_feat = dict()
    for fname in feature_names:
        adj_song_features[fname] = (song_features[fname] - med[fname]
                                    / mad[fname])
        adj_refsong_features[fname] = (refsong_features[fname] - med[fname]
                                       / mad[fname])
        L_feat = pdist2()
