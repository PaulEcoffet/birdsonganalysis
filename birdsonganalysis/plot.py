import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def plot_over_spec(data, ax, freq_range=256, **plot_params):
    """
    Plot the feature over the spectral derivatives plot

    The data are first normalized then rescale to fit the ylim of the axis.
    """
    ndata = data / (np.max(data) - np.min(data))
    # We plot -ndata because the yaxis is inverted (see `spectral_derivs_plot`)
    # We take for abscisse axis 95% of freq_range
    # We rescale the data so that they take 75% of the graph
    ax.plot(95/100 * freq_range - 75/100 * freq_range * (ndata - np.min(ndata)),
            **plot_params)
    return ax
