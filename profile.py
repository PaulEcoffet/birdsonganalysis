import pstats, profile

from birdsonganalysis import all_song_features, spectral_derivs
from birdsonganalysis.similarity import similarity

from scipy.io import wavfile

sr, song = wavfile.read("songs/bells.wav")
sr, repr = wavfile.read("songs/bells_reproduction.wav")

if False:
    profile.runctx("all_song_features(song, sr, 256, 40, 1024)", globals(), locals(), 'all_song_features.prof')
    s = pstats.Stats("all_song_features.prof")
    s.strip_dirs().sort_stats("time").print_stats()


    profile.runctx("spectral_derivs(song, 256, 40, 1024)", globals(), locals(), 'spec_derivs.prof')
    s = pstats.Stats("spec_derivs.prof")
    s.strip_dirs().sort_stats("time").print_stats()

profile.runctx("similarity(repr[:20000], song[:20000])", globals(), locals(), 'sim.prof')

s = pstats.Stats("sim.prof")
s.strip_dirs().sort_stats("time").print_stats()
