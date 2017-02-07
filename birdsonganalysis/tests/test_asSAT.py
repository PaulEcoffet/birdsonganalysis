import numpy as np
import unittest
import birdsonganalysis.songfeatures as sf
import libtfr
from scipy.io import wavfile, matlab

import os
dir_path = os.path.dirname(os.path.realpath(__file__))



class FeaturesTest(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(20170203)
        self.mat = matlab.loadmat(os.path.join(dir_path,'simple.mat'),
                                  squeeze_me=True, struct_as_record=False)['simple']
        self.sr, self.song = wavfile.read(
            os.path.join(dir_path, '../../songs/simple.wav'))

    def test_wiener(self):
        wiener = sf.song_wiener_entropy(self.song)
        np.testing.assert_allclose(wiener, self.mat.entropy)
