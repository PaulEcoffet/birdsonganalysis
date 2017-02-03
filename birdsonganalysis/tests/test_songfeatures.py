import numpy as np
import unittest
import birdsonganalysis.songfeatures as sf
import libtfr
from scipy.io import wavfile


class FeaturesTest(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(20170203)

    def test_wiener_perfect(self):
        """
        Test if wiener entropy is less for pure signal than for noisy
        """
        for goal in [440, 220]:
            signal = np.sin(np.linspace(0, 2048/44100, 2048) * 2 * np.pi * goal)
            entropy = sf.wiener_entropy(sf.get_power(signal))
            noise = self.rng.normal(size=len(signal))
            entropy2 = sf.wiener_entropy(sf.get_power(noise))
            self.assertGreater(entropy2, entropy)

    def test_frequence_modulation_perfect(self):
        """
        Test if frequence modulation is 0 for pure tone
        """
        for goal in [440, 220]:
            signal = np.sin(np.linspace(0, 2048/44100, 2048) * 2 * np.pi * goal)
            fm = sf.frequence_modulation(signal)
            self.assertAlmostEqual(fm, 0, places=1)

    def test_frequence_modulation_fast(self):
        """
        Test if frequence modulation is not 0 for accelerating frequence
        """
        for goal in [440, 220]:
            signal = np.sin(np.linspace(0, 2048/44100, 2048)**2 * 2 * np.pi * goal)
            fm = sf.frequence_modulation(signal)
            self.assertNotAlmostEqual(fm, 0, places=1)
