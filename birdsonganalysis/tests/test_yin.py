import numpy as np
import unittest
from birdsonganalysis import yin_pitch
from scipy.io import wavfile
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

class YinTest(unittest.TestCase):

    def test_yin_perfect(self):
        """
        Test if yin pitch estimate relative error is less than 1% for pure tone
        """
        for goal in [20, 440, 880, 2000, 4000]:
            signal = np.sin(np.linspace(0, 1, 44100) * 2 * np.pi * goal)
            pitch = yin_pitch.pitch(signal, 44100)
            self.assertLess(abs((goal - pitch) / goal),
                            0.01)

    def test_yin_guitar(self):
        """
        Test if yin pitch estimate relative error is less than 1% for guitar
        """
        sr, signal = wavfile.read(os.path.join(dir_path, 'OpenE.wav'))
        goal = 82.41
        pitch = yin_pitch.pitch(signal, sr)
        self.assertLess(abs((goal - pitch) / goal),
                            0.01)
