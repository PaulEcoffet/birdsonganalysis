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

from .songfeatures import all_song_features, song_amplitude
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


def similarity(song, refsong, threshold=0.01, ignore_silence=True,
               T=70, samplerate=44100):
    """
    Compute similarity between two songs.

    song - The song to compare
    refsong - The reference song (tutor song)
    threshold - The probability that the global error has this value even if
                the two songs are unrelated. The smaller the threshold, the
                less tolerant is the similar section identification. See
                Tchernichovski et al. 2000, appendix for more details.
    ignore_silence - Should the silence part be taken into account in the
                     similarity measurement.
    T - The number of windows to compute global average. According to
        Tchernichovski et al. 2000, the average must cover around 50ms
        of song. With 70 windows, spaced by 40 samples, and at samplerate of
        44100, the windows cover 63ms. It is also the default value used by
        Sound Analysis Toolbox.


    Return a dict with the keys :
    similarity - a float between 0 and 1
    sim_matrix - a 2D-array of the similarity probability
    glob_matrix - a 2D-array of the global similarity probability
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
    if ignore_silence:
        amp_song = song_amplitude(song)
        amp_refsong = song_amplitude(refsong)
        # Do not take into account all sounds that are in the first 20
        # percentile. They are very likely to be silent.
        silence_th = np.percentile(amp_song, 15)
        similarity[amp_song < silence_th, :] = 0
        silence_th = np.percentile(amp_refsong, 15)
        similarity[:, amp_refsong < silence_th] = 0
        len_refsong = similarity.shape[1] - np.sum(amp_refsong < silence_th)
    else:
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
           'sections': sections,
           'G2': G2,
           'L2': L2
           }
    return out


def p_val_err_local(x):
    """
    Give the probability that the local error could be `x` or less.

    See the notebook `Distrib` to understand the mean and std used.
    """
    assert np.all(x >= 0), 'Errors must be positive.'
    p = np.zeros(x.shape)
    for i in range(len(percentile_L)):
        p[np.where(x > percentile_L[i])] = (i+1)/100
    return p
    # return norm.cdf(np.log(x + 0.01), 2.0893176665431645, 1.3921924227352549)


def p_val_err_global(x):
    """
    Give the probability that the global error could be `x` or less.

    The fit is done using 4 songs, it is available in the notebook `Distrib`
    """
    assert np.all(x >= 0), 'Errors must be positive.'
    p = np.zeros(x.shape)
    for i in range(len(percentile_G)):
        p[np.where(x > percentile_G[i])] = (i+1)/100
    return p
#    return norm.cdf(np.log(x + 0.01), 2.6191330043001892, 1.6034598153962765)


percentile_G = np.array(
        [1.89770356e+00,   2.27389503e+00,   2.54968227e+00,
         2.78131417e+00,   2.98486788e+00,   3.17123629e+00,
         3.34525077e+00,   3.51025085e+00,   3.66863695e+00,
         3.82109806e+00,   3.96861476e+00,   4.11290115e+00,
         4.25507455e+00,   4.39585569e+00,   4.53567679e+00,
         4.67515723e+00,   4.81488665e+00,   4.95563525e+00,
         5.09726824e+00,   5.23960718e+00,   5.38356220e+00,
         5.52779327e+00,   5.67294862e+00,   5.81901377e+00,
         5.96695865e+00,   6.11639770e+00,   6.26676849e+00,
         6.41763517e+00,   6.57075117e+00,   6.72572460e+00,
         6.88269590e+00,   7.04167035e+00,   7.20257536e+00,
         7.36647737e+00,   7.53412098e+00,   7.70433931e+00,
         7.87775660e+00,   8.05426529e+00,   8.23487026e+00,
         8.41928846e+00,   8.60884119e+00,   8.80291199e+00,
         9.00161567e+00,   9.20586305e+00,   9.41598225e+00,
         9.63201208e+00,   9.85583601e+00,   1.00877407e+01,
         1.03268864e+01,   1.05746760e+01,   1.08332251e+01,
         1.11015503e+01,   1.13791127e+01,   1.16684179e+01,
         1.19724341e+01,   1.22872865e+01,   1.26182066e+01,
         1.29681133e+01,   1.33353626e+01,   1.37201490e+01,
         1.41253741e+01,   1.45520608e+01,   1.49994858e+01,
         1.54707340e+01,   1.59606214e+01,   1.64689445e+01,
         1.69961200e+01,   1.75393722e+01,   1.81075811e+01,
         1.87041123e+01,   1.93296962e+01,   1.99928838e+01,
         2.06844122e+01,   2.14121442e+01,   2.21794347e+01,
         2.29795787e+01,   2.38141760e+01,   2.46865149e+01,
         2.55932126e+01,   2.65494145e+01,   2.75655017e+01,
         2.86727643e+01,   2.98850104e+01,   3.12387528e+01,
         3.27591829e+01,   3.44543271e+01,   3.64565123e+01,
         3.89429186e+01,   4.21844383e+01,   4.68543550e+01,
         5.41365744e+01,   6.34448203e+01,   8.26659737e+01,
         1.24112847e+02,   3.68541621e+02,   5.15148887e+02,
         7.18761009e+02,   1.19109255e+03,   2.52444106e+03,
         3.78843768e+04])


percentile_L = np.array(
        [4.94400157e-01,   7.04678964e-01,   8.74760001e-01,
         1.02491632e+00,   1.16317131e+00,   1.29351061e+00,
         1.41786443e+00,   1.53807643e+00,   1.65482839e+00,
         1.76914126e+00,   1.88122073e+00,   1.99158336e+00,
         2.10059395e+00,   2.20871828e+00,   2.31636712e+00,
         2.42374115e+00,   2.53106505e+00,   2.63841006e+00,
         2.74614272e+00,   2.85414560e+00,   2.96278730e+00,
         3.07159799e+00,   3.18122139e+00,   3.29188653e+00,
         3.40361070e+00,   3.51656705e+00,   3.63102319e+00,
         3.74742232e+00,   3.86571756e+00,   3.98566492e+00,
         4.10804084e+00,   4.23264788e+00,   4.35999560e+00,
         4.48984574e+00,   4.62256738e+00,   4.75786253e+00,
         4.89671603e+00,   5.03873215e+00,   5.18408394e+00,
         5.33289804e+00,   5.48588200e+00,   5.64291387e+00,
         5.80465582e+00,   5.97106640e+00,   6.14271235e+00,
         6.31926721e+00,   6.50173997e+00,   6.68944974e+00,
         6.88350258e+00,   7.08414562e+00,   7.29124030e+00,
         7.50539333e+00,   7.72683083e+00,   7.95538380e+00,
         8.19146265e+00,   8.43629850e+00,   8.68879790e+00,
         8.95035416e+00,   9.22046036e+00,   9.50127596e+00,
         9.79355365e+00,   1.00970491e+01,   1.04137992e+01,
         1.07442769e+01,   1.10897138e+01,   1.14523194e+01,
         1.18340159e+01,   1.22380678e+01,   1.26663879e+01,
         1.31213365e+01,   1.36066483e+01,   1.41231152e+01,
         1.46759899e+01,   1.52680318e+01,   1.59031359e+01,
         1.65881319e+01,   1.73323807e+01,   1.81441121e+01,
         1.90330985e+01,   2.00097591e+01,   2.10898332e+01,
         2.22911065e+01,   2.36330590e+01,   2.51415355e+01,
         2.68259084e+01,   2.87142336e+01,   3.08684567e+01,
         3.33497007e+01,   3.62897803e+01,   3.98433359e+01,
         4.42306413e+01,   4.95221700e+01,   5.61038620e+01,
         6.46724830e+01,   7.57920701e+01,   9.17127595e+01,
         1.20529332e+02,   1.88840899e+02,   1.75695874e+03,
         9.50517349e+04])
