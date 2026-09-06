from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict
from enum import Enum
import numpy as np


class NormalizationType(Enum):
    """Enumeration of available normalization methods."""
    AMIDE = "amide"
    VECTOR = "vector"
    SNV = "snv"

@dataclass(frozen=True)
class SpectralRegion:
    """Represents a spectral region with start and end wavenumbers."""
    start: float
    end: float
    
    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError(f"Start wavenumber ({self.start}) must be less than end ({self.end})")
    
    def contains(self, wavenumber: float) -> bool:
        """Check if a wavenumber falls within this region."""
        return self.start <= wavenumber <= self.end
    
    def to_mask(self, wavenumbers: np.ndarray) -> np.ndarray:
        """Create a boolean mask for wavenumbers in this region."""
        return (wavenumbers >= self.start) & (wavenumbers <= self.end)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for spectral data preprocessing."""
    baseline_correction: bool = False
    savgol_window: Optional[int] = None
    savgol_polyorder: Optional[int] = None
    savgol_deriv: int = 0
    normalization: Optional[NormalizationType] = None
    regions: Optional[List[SpectralRegion]] = None
    
    def __post_init__(self):
        if self.savgol_window is not None:
            if self.savgol_polyorder is None:
                raise ValueError("savgol_polyorder must be specified if savgol_window is set")
            if self.savgol_window <= self.savgol_polyorder:
                raise ValueError("savgol_window must be greater than savgol_polyorder")
            if self.savgol_window % 2 == 0:
                raise ValueError("savgol_window must be odd")


@dataclass(frozen=True)
class DataSplitConfig:
    """Configuration for train/test/validation splitting."""
    test_size: float = 0.2
    val_size: Optional[float] = None
    stratify_bins: int = 8
    random_state: int = 34
    
    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.val_size is not None and not 0 < self.val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if self.val_size is not None and (self.test_size + self.val_size) >= 1:
            raise ValueError("test_size + val_size must be less than 1")


@dataclass
class SpectralData:
    """
    Immutable container for spectral data with metadata.
    
    Attributes:
        spectra: 2D array of shape (n_samples, n_features)
        wavenumbers: 1D array of wavenumber values
        features: Dictionary mapping feature names to 1D arrays (e.g., {'HbA1c': array, 'Age': array})
        metadata: Optional dictionary for additional information
    """
    spectra: np.ndarray
    wavenumbers: np.ndarray
    features: Dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data shapes and convert to numpy arrays if needed."""

        self.spectra = np.asarray(self.spectra)
        self.wavenumbers = np.asarray(self.wavenumbers).flatten()
        
        features_dict = {}
        for key, value in self.features.items():
            features_dict[key] = np.asarray(value).flatten()
        object.__setattr__(self, 'features', features_dict)
        
        n_samples = self.spectra.shape[0]

        if self.spectra.shape[1] != len(self.wavenumbers):
            raise ValueError("Number of wavenumbers must match spectra features")
        
        for feature_name, feature_values in self.features.items():
            if len(feature_values) != n_samples:
                raise ValueError(f"Feature '{feature_name}' length must match number of samples")
    
    @property
    def n_samples(self) -> int:
        """Number of spectral samples."""
        return self.spectra.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of spectral features (wavenumbers)."""
        return self.spectra.shape[1]
    
    def get_feature(self, name: str):
        return self.features.get(name, None)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: Union[int, slice, np.ndarray, List[int]]) -> 'SpectralData':
        """
        Get subset of data by index, returning new SpectralData instance.
        
        Args:
            idx: Integer, slice, or boolean/integer array for indexing
            
        Returns:
            New SpectralData instance with selected samples
        """
        if isinstance(idx, (int, np.integer)):
            idx = slice(idx, idx + 1)
        
        indexed_features = {}
        for key, values in self.features.items():
            indexed_features[key] = values[idx]
        
        return SpectralData(
            spectra=self.spectra[idx],
            wavenumbers=self.wavenumbers,
            features=indexed_features,
            metadata=self.metadata.copy()
        )
    
    def copy(self) -> 'SpectralData':
        """Create a deep copy of this SpectralData instance."""
        features_copy = {key: values.copy() for key, values in self.features.items()}
        return SpectralData(
            spectra=self.spectra.copy(),
            wavenumbers=self.wavenumbers.copy(),
            features=features_copy,
            metadata=self.metadata.copy()
        )


@dataclass(frozen=True)
class DataSplit:
    """
    Container for train/test/validation splits.

    Attributes:
        train: Training samples
        test: Test samples
        val: Validation samples, if a three-way split was requested
        train_idx: Positions of the training samples in the data that was split
        test_idx: Positions of the test samples in the data that was split
        val_idx: Positions of the validation samples in the data that was split
    """
    train: SpectralData
    test: SpectralData
    val: Optional[SpectralData] = None
    train_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    val_idx: Optional[np.ndarray] = None
