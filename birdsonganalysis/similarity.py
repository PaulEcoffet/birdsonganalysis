"""Compute the similarity between two songs."""

#################################################################
#                                                               #
# This code may be quite hard to understand, feel free to send  #
# me mails if you need some help.                               #
# ecoffet.paul@gmail.com                                        #
#                                                               #
#################################################################
import numpy as np
from scipy.stats import norm

from .songfeatures import all_song_features
from .utils import calc_dist_features, normalize_features, get_windows


def identify_sections(similarity):
    """
    Identify the blocks of similarity in a song.

    This algorithm is written in step 7 of the appendix of Tchernichovski 2000.
    """
    directions = [(1, 0), (0, 1), (1, 1)]
    sections = []
    visited = np.full(similarity.shape, False, dtype=bool)
    for i, j in sorted(zip(*np.where(similarity > 0))):
        if visited[i, j]:
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
            sections.append({'beg': beg, 'end': end})
        # If it is already part of a section, it is no use to
        # start exploring from this point, so we put locvisited as
        # visited
        visited = visited | locvisited
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
            G2[i, j] = np.mean(np.diag(L2[imin:imax, jmin:jmax]))

    ####################################################################
    # Compute P value and reject similarity hypothesis (steps 5 and 6) #
    ####################################################################
    glob = 1 - p_val_err_global(G2)
    similarity = np.where(glob > (1 - threshold),
                          1 - p_val_err_local(L2),
                          0)
    #########################################
    # Identify similarity sections (step 7) #
    #########################################
    len_refsong = similarity.shape[1]
    sections = []
    wsimilarity = np.copy(similarity)
    while True:
        cur_sections = identify_sections(wsimilarity)
        if len(cur_sections) == 0:
            break  # Exit the loop if there is no more sections
        for section in cur_sections:
            beg, end = section['beg'], section['end']
            section['P'] = (np.sum(np.max(similarity[beg[0]:end[0]+1,
                                                     beg[1]:end[1]+1], axis=0))
                            / len_refsong)
        cur_sections.sort(key=lambda x: x['P'])
        best = cur_sections.pop()
        wsimilarity[best['beg'][0]:best['end'][0]+1, :] = 0
        wsimilarity[:, best['beg'][1]:best['end'][1]+1] = 0
        sections.append(best)
    out = {'similarity': np.sum([section['P'] for section in sections]),
           'sim_matrix': similarity,
           'glob_matrix': glob,
           'sections': sections
           }
    return out


def p_val_err_local(x):
    """
    Give the probability that the local error could be `x` or less.

    See the notebook `Distrib` to understand the mean and std used.
    """
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(np.log(x + 0.01), 2.5212985606078586, 1.9138147753306101)

def p_val_err_global(x):
    """
    Give the probability that the global error could be `x` or less.

    The fit is done using 4 songs, it is available in the notebook `Distrib`
    """
    assert np.all(x >= 0), 'Errors must be positive.'
    return norm.cdf(np.log(x + 0.01), 3.6359735324494631, 1.6034598153962765)
