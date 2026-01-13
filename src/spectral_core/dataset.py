from typing import Optional, List, Dict, Union, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import pandas as pd

try:
    from brokenaxes import brokenaxes
    BROKENAXES_AVAILABLE = True
except ImportError:
    BROKENAXES_AVAILABLE = False

from .models import SpectralData
from .loaders import SpectralDataLoader


class SpectralDataset:
    """
    High-level wrapper for SpectralData.
    """
    
    def __init__(self, data: SpectralData):
        """
        Initialize directly from a SpectralData object.
        Use class methods (from_csv, etc.) for loading from files.
        """
        self._data = data

    @classmethod
    def from_csv(cls, file_path: Union[str, Path], wavenumbers_path: Union[str, Path], **kwargs):
        """Initialize directly from CSV files."""
        data = SpectralDataLoader.from_csv(file_path, wavenumbers_path, **kwargs)
        return cls(data)

    @classmethod
    def from_dataframe(cls, df, wavenumbers, **kwargs):
        """Initialize directly from DataFrame."""
        data = SpectralDataLoader.from_dataframe(df, wavenumbers, **kwargs)
        return cls(data)

    @classmethod
    def from_npy(cls, spectra_path, wavenumbers_path, features_paths):
        """Initialize directly from NPY files."""
        data = SpectralDataLoader.from_npy(spectra_path, wavenumbers_path, features_paths)
        return cls(data)

    @classmethod
    def from_arrays(cls, spectra, wavenumbers, features, metadata=None):
        """Initialize directly from Numpy arrays."""
        data = SpectralDataLoader.from_arrays(spectra, wavenumbers, features, metadata)
        return cls(data)

    @property
    def data(self) -> SpectralData:
        return self._data
    
    @property
    def spectra(self) -> np.ndarray:
        return self._data.spectra
    
    @property
    def wavenumbers(self) -> np.ndarray:
        return self._data.wavenumbers
    
    @property
    def features(self) -> Dict[str, np.ndarray]:
        return self._data.features
    
    @property
    def n_samples(self) -> int:
        return self._data.n_samples
    
    @property
    def n_features(self) -> int:
        return self._data.n_features

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic access to feature arrays. 
        Allows dataset.HbA1c instead of dataset.features['HbA1c'].
        """
        if name in {'_data', 'features'}:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        if name in self._data.features:
            return self._data.features[name]
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute or feature '{name}'")

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx) -> tuple:
        return self._data[idx]

    def __repr__(self) -> str:
        return (f"<SpectralDataset: {self.n_samples} samples, {self.n_features} wn, "
                f"Features: {list(self.features.keys())}>")

    def summary(self) -> pd.DataFrame:
            """
            Displays metadata and returns feature statistics as a DataFrame.
            """
            try:
                from IPython.display import display, Markdown
                header = (
                    f"### Spectral Dataset Summary\n"
                    f"- **Samples:** {self.n_samples}\n"
                    f"- **Features:** {self.n_features} wavenumbers\n"
                    f"- **Range:** {self.wavenumbers.min():.1f} - {self.wavenumbers.max():.1f} cm⁻¹"
                )
                display(Markdown(header))
            except ImportError:
                print("=" * 50)
                print(f"Spectral Dataset: {self.n_samples} samples | {self.n_features} features")
                print(f"Range: {self.wavenumbers.min():.1f} - {self.wavenumbers.max():.1f} cm⁻¹")
                print("=" * 50)

            stats = {
                name: {
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Max': values.max()
                }
                for name, values in self.features.items()
            }
            
            return pd.DataFrame(stats).T
    
    def plot_spectra(
            self, 
            feature: str = None, 
            alpha: float = 0.7, 
            figsize: tuple = (10, 5), 
            cmap: str = 'viridis'
        ) -> tuple:
        """
        Plot spectra. If feature is provided, color lines by feature value.
        Automatically uses broken axes if regions are present in metadata.
        """
        if feature is None and self.features:
            feature = list(self.features.keys())[0]
        elif feature is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.wavenumbers, self.spectra.T, color='k', alpha=0.1)
            return fig, ax

        if feature not in self.features:
            raise ValueError(f"Feature '{feature}' not found. Available: {list(self.features.keys())}")
        
        feature_values = self.features[feature]
        norm = Normalize(vmin=feature_values.min(), vmax=feature_values.max())
        colormap = cm.get_cmap(cmap)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        
        regions = self._data.metadata.get('selected_regions', None)
        
        if regions is None or len(regions) == 1:
            fig, ax = plt.subplots(figsize=figsize)
            for idx in range(self.n_samples):
                ax.plot(
                    self.wavenumbers,
                    self.spectra[idx],
                    color=colormap(norm(feature_values[idx])),
                    alpha=alpha,
                    linewidth=0.5
                )
            
            ax.set_xlabel('Wavenumbers (cm⁻¹)')
            ax.set_ylabel('Absorbance')
            ax.set_title(f'Spectra colored by {feature}')
            
            if regions and len(regions) == 1:
                ax.set_xlim([regions[0][0], regions[0][1]])
            
            fig.colorbar(sm, ax=ax, label=feature)
            return fig, ax
        
        if not BROKENAXES_AVAILABLE:
            raise ImportError(
                "brokenaxes package is required for plotting multiple regions. "
                "Install with: pip install brokenaxes"
            )
        
        fig = plt.figure(figsize=figsize)
        xlims = tuple((start, end) for start, end in regions)
        bax = brokenaxes(xlims=xlims, hspace=0.1, despine=False)
        
        for idx in range(self.n_samples):
            bax.plot(
                self.wavenumbers,
                self.spectra[idx],
                color=colormap(norm(feature_values[idx])),
                alpha=alpha,
                linewidth=0.5
            )
        
        bax.set_xlabel('Wavenumbers (cm⁻¹)')
        bax.set_ylabel('Absorbance')
        fig.suptitle(f'Spectra colored by {feature}')
        
        # Add colorbar to the last axis
        fig.colorbar(sm, ax=bax.axs[-1], label=feature)
        
        return fig, bax