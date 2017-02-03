import numpy as np
import unittest
import birdsonganalysis.songfeatures as sf
import libtfr
from scipy.io import wavfile


class FeaturesTest(unittest.TestCase):

    def test_wiener_perfect(self):
        """
        Test if wiener entropy is -inf for pure tone
        """
        for goal in [440, 220]:
            signal = np.sin(np.linspace(0, 2048/44100, 2048) * 2 * np.pi * goal)
            entropy = sf.wiener_entropy(sf.get_power(signal))
            noise = np.random.normal(size=len(signal))
            entropy2 = sf.wiener_entropy(sf.get_power(noise))
            self.assertGreater(entropy2, entropy)
