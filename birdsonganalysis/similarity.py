"""Compute the similarity between two songs."""

import numpy as np
from scipy.stats import norm

import .songfeatures as sf


def distbroad(X, Y):
    """
    Compute the dist between two features vector element by element.

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
        adj_song_features[fname] = (song_features[fname] - med[fname]
                                    / mad[fname])
    return adj_song_features


def calc_dist_features(feats1, feats2, feat_names=None):
    """Compute the distance between two dict of features."""
    if feat_names is None:
        feat_names = feats.key()
    out = dict()
    for fname in feat_names:
        out[fname] = distbroad(feats1[fname], feats2[fname])
    return out


def identify_blocks(similarity):
    """Identify the blocks of similarity in a song."""
    pass


def similarity(song, refsong, T=70, threshold=0.01):
    """
    Compute similarity between two songs.

    song - The song to compare
    refsong - The reference song (tutor song)
    T - The number of windows to compute global average. According to
        Tchernichovski et al. 2000, the average must cover around 50ms
        of song. With 70 windows, spaced by 40 samples, and at samplerate of
        44100, the windows cover 63ms. It is also the default value used by
        Sound Analysis Toolbox.

    Compute the similarity between the song `song` and a reference song
    `refsong` using the method described in Tchernichovski, Nottebohm,
    Ho, Pesaran, & Mitra (2000).
    All the methods are detailed in this article, in the Appendix section.
    The paper is available in open access on several plateforms.

    This implementation follow the rules in the paper and not the ones in SAT.

    ### References:

    Tchernichovski, O., Nottebohm, F., Ho, C. E., Pesaran, B., & Mitra,
    P. P. (2000). A procedure for an automated measurement of song similarity.
    Animal Behaviour, 59(6), 1167–1176. https://doi.org/10.1006/anbe.1999.1416
    """
    song_win = get_windows(song)
    refsong_win = get_windows(song)
    #########################################################################
    # Compute sound features and scale them (step 2 of Tchernichovski 2000) #
    #########################################################################
    song_features = all_song_features(song)
    refsong_features = all_song_features(refsong)
    adj_song_features = normalize_features(song_features)
    adj_refsong_features = normalize_features(refsong_features)
    #################################
    # Compute the L matrix (step 3) #
    #################################
    # L2 = L²
    local_dists = calc_dist_features(adj_song_features, adj_refsong_features)
    L2 = np.mean(
        np.array([local_dists[fname] for fname in local_dists.keys()]),
        axis=0)
    G2 = np.zeros((song_win.shape[0], refsong_win.shape[0]))  # G2 = G²
    #############################
    # Compute G Matrix (step 4) #
    #############################
    for i in range(song_win.shape[0]):
        for j in range(refsong_win.shape[0]):
            G2[i, j] = np.mean(L2[(i-T//2):(i+T//2)])
    ####################################################################
    # Compute P value and reject similarity hypothesis (steps 5 and 6) #
    ####################################################################
    similarity = np.where(p_val_err_global(np.sqrt(G2)) < threshold,
                          1 - p_val_err_local(np.sqrt(L2)),
                          0)
    #######################################
    # Identify similarity blocks (step 7) #
    #######################################
    blocks = identify_blocks(similarity)


def p_val_err_local(x):
    """Give the probability that the local error could be `x` or less."""
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(x, 2)  # TODO Assumed to be gaussian, not the case
    # assumed mean error would be 2 MAD, with std 1 MAD.


def p_val_err_global(x):
    """Give the probability that the global error could be `x` or less.""""
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(x, 2)  # TODO Assumed to be gaussian, not the case
    # assumed mean error would be 2 MAD, with std 1 MAD
