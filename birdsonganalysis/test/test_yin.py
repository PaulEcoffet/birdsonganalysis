import numpy as np
import unittest
import yin_pitch
from scipy.io import wavfile


class YinTest(unittest.TestCase):

    def test_yin_perfect(self):
        """
        Test if relative error is less than 5%
        """
        for goal in [20, 440, 880, 2000, 4000]:
            signal = np.sin(np.linspace(0, 1, 44100) * 2 * np.pi * goal)
            self.assertLess(abs((goal - yin_pitch.pitch(signal, 44100)) / goal),
                            0.05)

    def test_yin_guitar(self):
        sr, signal = wavfile.read('test/OpenE.wav')
        goal = 82.41
        self.assertLess(abs((goal - yin_pitch.pitch(signal, sr)) / goal),
                            0.05)
