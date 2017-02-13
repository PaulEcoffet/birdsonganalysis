"""Compute the similarity between two songs."""

#################################################################
#                                                               #
# This code may be quite hard to understand, feel free to send  #
# me mails if you need some help.                               #
# ecoffet.paul@gmail.com                                        #
#                                                               #
#################################################################

import numpy as np
import os
from scipy.stats import norm

from .songfeatures import all_song_features, get_windows


# TODO med and mad should not be hard coded
med = {
    'pitch': 688,
    'fm': 0.64,
    'am': 0,
    'entropy': -1.8
}

mad = {
    'pitch': 200,
    'fm': 0.34,
    'am': 2.1,
    'entropy': 0.94
}


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
        adj_song_features[fname] = ((song_features[fname] - med[fname])
                                    / mad[fname])
        assert not np.any(np.isnan(adj_song_features[fname])), \
            'nan in {}'.format(fname)
    return adj_song_features


def calc_dist_features(feats1, feats2, feat_names=None):
    """Compute the distance between two dict of features."""
    if feat_names is None:
        feat_names = feats1.keys()
    out = dict()
    for fname in feat_names:
        out[fname] = distbroad(feats1[fname], feats2[fname])
    return out


def identify_sections(similarity):
    """
    Identify the blocks of similarity in a song.

    This algorithm is written in step 7 of the appendix of Tchernichovski 2000.
    """
    directions = [(1, 0), (0, 1), (1, 1)]
    len_refsong = similarity.shape[1]
    sections = []
    visited = np.full(similarity.shape, False, dtype=bool)
    for i, j in zip(*np.where(similarity > 0)):
        if j > i or visited[i, j]:  # do not visit if j > i because symmetrical
            continue
        locvisited = np.full(similarity.shape, False, dtype=bool)
        # `locvisited` represents the element of the matrix visited
        # during the creation of one specific section
        # locvisited is recreated every new section because one element
        # can be in two different sections. Though, if an element is
        # already in a section, any section with this element in the upper
        # left corner will be a subsection of the sections containing
        # the element, and therefore, this subsection will be less good.
        #  . . . . . . . . .
        #  . . + - - - - + .
        #  . . | . . . . | .
        #  . + + - - + . | .
        #  . | | . * | . | .
        #  . + + - - + . | .
        #  . . | . . . . | .
        #  . . + - - - - + .
        # The star can be both in the big section or the small section.
        # Thus they need their own locvisited for the flooding.
        # But the section starting from the star will necesserarly be
        # less good than any other section which started before.
        # Therefore it is no use to take the star as the `beg` coordinate
        # of a section. And every element which is already in a section
        # does not need to be taken as `beg`.
        # if it is not clear, just send me a mail ecoffet.paul@gmail.com
        locvisited[i, j] = True
        beg = (i, j)
        end = (i, j)
        flood_stack = [beg]
        # use a flood algorithm to find the boundaries of the section
        # as stated in step 7 of Tchernichovski 2000
        while flood_stack:
            cur = flood_stack.pop()
            locvisited[cur] = True
            # extend the boundaries of the section
            end = (max(end[0], cur[0]), max(end[1], cur[1]))
            for diri, dirj in directions:
                new_coord = (cur[0] + diri, cur[1] + dirj)
                if new_coord[0] < similarity.shape[0] \
                        and new_coord[1] < similarity.shape[1] \
                        and similarity[new_coord] > 0 \
                        and not locvisited[new_coord]:
                    locvisited[new_coord] = True
                    flood_stack.append((new_coord))
        if end[0] - beg[0] > 4 and end[1] - beg[1] > 4:
            P = np.sum(np.max(similarity[beg[0]:end[0]+1, beg[1]:end[1]+1],
                              axis=0)) / len_refsong
            sections.append({'beg': beg, 'end': end, 'P': P})
        # If it is already part of a section, it is no use to
        # start exploring from this point, so we put locvisited as
        # visited
        visited = visited | locvisited
    sections.sort(key=lambda x: x['P'], reverse=True)
    return sections


def similarity(song, refsong, T=70, threshold=0.01, samplerate=44100):
    """
    Compute similarity between two songs.

    song - The song to compare
    refsong - The reference song (tutor song)
    T - The number of windows to compute global average. According to
        Tchernichovski et al. 2000, the average must cover around 50ms
        of song. With 70 windows, spaced by 40 samples, and at samplerate of
        44100, the windows cover 63ms. It is also the default value used by
        Sound Analysis Toolbox.

    Return a dict with the keys :
    similarity - a float between 0 and 1
    sim_matrix - a 2D-array of the similarity
    sections - The sections that are similar and their scores

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
    refsong_win = get_windows(refsong)
    #########################################################################
    # Compute sound features and scale them (step 2 of Tchernichovski 2000) #
    #########################################################################
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
    local_dists = calc_dist_features(adj_song_features, adj_refsong_features)
    L2 = np.mean(
        np.array([local_dists[fname] for fname in local_dists.keys()]),
        axis=0)
    # avoid boundaries effect
    maxL2 = np.max(L2)
    L2[:T//2, :] = maxL2
    L2[-(T//2):, :] = maxL2
    L2[:, :T//2] = maxL2
    L2[:, -(T//2):] = maxL2
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

    ####################################################################
    # Compute P value and reject similarity hypothesis (steps 5 and 6) #
    ####################################################################
    similarity = np.where(p_val_err_global(np.sqrt(G2)) < threshold,
                          1 - p_val_err_local(np.sqrt(L2)),
                          0)
    #########################################
    # Identify similarity sections (step 7) #
    #########################################
    all_sections_found = False
    sections = []
    wsimilarity = np.copy(similarity)
    while not all_sections_found:
        cur_sections = identify_sections(wsimilarity)
        if cur_sections:
            best = cur_sections[0]
            sections.append(best)
            wsimilarity[best['beg'][0]:best['end'][0]+1, :] = 0
            wsimilarity[:, best['beg'][1]:best['end'][1]] = 0
        else:
            all_sections_found = True
    out = {'similarity': np.sum([section['P'] for section in sections]),
           'sim_matrix': similarity,
           'sections': sections
           }
    return out


def p_val_err_local(x):
    """Give the probability that the local error could be `x` or less."""
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(x, 6, 1)  # TODO Assumed to be gaussian, not the case
    # assumed mean error would be 2 MAD, with std 0.5 MAD.


def p_val_err_global(x):
    """Give the probability that the global error could be `x` or less."""
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(x, 6, 1)  # TODO Assumed to be gaussian, not the case
    # assumed mean error would be 2 MAD, with std 1 MAD
