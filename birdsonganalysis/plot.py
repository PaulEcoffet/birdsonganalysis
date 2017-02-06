import matplotlib.pyplot as plt
import seaborn as sns

from .songfeatures import spectral_derivs

def spectral_derivs_plot(spec_der, contrast=0.1, ax=None, freq_range=None,
                         ov_params=None):
    """
    Plot the spectral derivatives of a song in a grey scale.
    spec_der - The spectral derivatives of the song (computed with
               `spectral_derivs`) or the song itself
    contrast - The contrast of the plot
    ax - The matplotlib axis where the plot must be drawn, if None, a new axis
         is created
    freq_range - The amount of frequency to plot, usefull only if `spec_der` is
                 a song. Given to `spectral_derivs`
    ov_params - The Parameters to override, passed to `spectral_derivs`
    """
    if spec_der.ndim == 1:
        spec_der = spectral_derivs(spec_der, freq_range, ov_params)
    ax = sns.heatmap(spec_der.T, yticklabels=100, xticklabels=100,
                     vmin=-contrast, vmax=contrast, ax=ax, cmap='Greys',
                     cbar=False)
    ax.invert_yaxis()
    return ax
