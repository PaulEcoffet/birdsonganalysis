import numpy as np


"""
translated from https://github.com/ashokfernandez/Yin-Pitch-Tracking/blob/master/Yin.c

De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator
for speech and music. The Journal of the Acoustical Society of America, 111(4),
1917-1930.
http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf
"""

def yin_diff(buff):
    half = len(buff)//2
    buff = np.array(buff, dtype=float)
    yin = np.zeros(half, dtype=float)
    for tau in range(0, half):
        deltas = buff[:half] - buff[tau:(half+tau)]
        yin[tau] = np.sum(np.square(deltas))
    return yin

def cumulative_mean_normalized_diff(yin):
    yin = np.copy(yin)
    yin[0] = 1
    running_sum = np.cumsum(yin)
    yin *= np.arange(0, len(yin))/running_sum
    return yin

def absolute_threshold(yin, threshold):
    found = False
    tau = 2
    while not found and tau < len(yin):
        if (yin[tau] < threshold):
            found = True
            while yin[tau + 1] < yin[tau]:
                tau += 1
        else:
            tau += 1
    if found:
        return tau
    else:
        raise Exception("No pitch found")

def parabolic_interpolation(yin, tau):
    x0 = tau - 1
    x2 = min(tau + 1, len(yin)-1)
    s0 = yin[x0]
    s1 = yin[tau]
    s2 = yin[x2]
    better_tau = tau + (s2 - s0) / (2 * ((2 * s1) - s2 - s0))
    return better_tau

def pitch(signal, samplerate, threshold=0.10):
    yin = cumulative_mean_normalized_diff(yin_diff(signal))
    tau = absolute_threshold(yin, threshold)
    better_tau = parabolic_interpolation(yin, tau)
    return samplerate/better_tau
