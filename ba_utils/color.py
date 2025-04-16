from abc import ABC, abstractmethod
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ColorMapper(ABC):
    @abstractmethod
    def get_color_by_value(self, value):
        pass

class LinearHSVColorMapper(ColorMapper):
    '''
        linear normalization between a min_val and max_val
    '''
    def __init__(self, max_val, min_val=0, colormap='turbo'):
        """
        Initializes the LinearHSVColorMapper with a specified matplotlib colormap.
        colormap: Name of the colormap to use (default: 'turbo')
        """
        self.colormap = cm.get_cmap(colormap)
        self.min_val = min_val
        self.max_val = max_val

    def get_color_by_value(self, value):
        """
        Maps a value linearly to the color map.
        value: float between 0 and 1
        return: tuple (r, g, b, a)
        """
        if not self.min_val <= value <= self.max_val:
            raise ValueError(f"Value must be between {self.min_val} and {self.max_val}")
        
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return self.colormap(normalized)

class BinnedPercentileColorMapper(ColorMapper):
    def __init__(self, bins=10, colormap='bwr', min_val=0, max_val=1):
        """
        Initializes the BinnedPercentileColorMapper with specified bins and colormap.
        bins: Number of bins to divide the range into (default: 10)
        colormap: Name of the colormap to use (default: 'bwr')
        min_val: Minimum value of the range (default: 0)
        max_val: Maximum value of the range (default: 1)
        """
        self.bins = bins
        self.colormap = cm.get_cmap(colormap)
        self.min_val = min_val
        self.max_val = max_val

    def get_color_by_value(self, value):
        """
        Maps a value to a color based on bins.
        value: float between min_val and max_val
        return: tuple (r, g, b, a)
        """
        if not self.min_val <= value <= self.max_val:
            raise ValueError(f"Value must be between {self.min_val} and {self.max_val}.")
        
        normalized_value = (value - self.min_val) / (self.max_val - self.min_val)
        bin_index = int(normalized_value * self.bins)
        bin_index = min(bin_index, self.bins - 1)  # Ensure it stays within bounds
        bin_value = bin_index / (self.bins - 1)
        return self.colormap(bin_value)

def get_color_by_id(artist_idx, total_artists):
    cmap = plt.colormaps['Greys']  # Greyscale colormap
    norm = mcolors.Normalize(vmin=0, vmax=total_artists - 1)  # Normalization 
    return cmap(norm(artist_idx))

def get_color_by_id2(artist_idx, total_artists):
    cmap = plt.colormaps['bwr']  # Blue-White-Red colormap
    norm = mcolors.Normalize(vmin=0, vmax=total_artists - 1)
    return cmap(norm(artist_idx))

def get_color_by_id3(artist_idx, total_artists):
    cmap = plt.colormaps['turbo']  # bright and perceptually improved rainbow scale, other option 'gist_rainbow'
    norm = mcolors.Normalize(vmin=0, vmax=total_artists - 1)
    return cmap(norm(artist_idx))

def get_color_by_id4(artist_idx, total_artists):
    #cmap = plt.colormaps['RdYlGn']  # Red-Yellow-Green colormap
    cmap = plt.colormaps['RdYlBu']  # Red-Yellow-Blue colormap
    norm = mcolors.Normalize(vmin=0, vmax=total_artists - 1)
    return cmap(norm(artist_idx))

def get_color_by_exhibitions(num_exhibitions, max_exhibitions, linear_mapper):
    return linear_mapper.get_color_by_value(num_exhibitions / max_exhibitions)