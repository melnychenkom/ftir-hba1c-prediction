from typing import Optional, List, Sequence, Tuple, Union, overload
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from spectral_core.dataset import SpectralDataset

try:
    from boxsers.preprocessing import rubberband_baseline_cor
    BOXSERS_AVAILABLE = True
except ImportError:
    BOXSERS_AVAILABLE = False

from .models import SpectralData, NormalizationType, SpectralRegion


class PreprocessingStep:
    """Base class for preprocessing steps."""
    
    def apply(self, data: SpectralData) -> SpectralData:
        """
        Apply preprocessing step to data.
        
        Args:
            data: Input SpectralData
            
        Returns:
            New SpectralData instance with preprocessing applied
        """
        raise NotImplementedError


class BaselineCorrection(PreprocessingStep):
    """Rubberband baseline correction."""
    
    def __init__(self):
        if not BOXSERS_AVAILABLE:
            raise ImportError(
                "boxsers package is required for baseline correction. "
                "Install with: pip install boxsers"
            )
    
    def apply(self, data: SpectralData) -> SpectralData:
        """Apply rubberband baseline correction to each spectrum."""
        corrected_spectra = np.apply_along_axis(
            lambda row: rubberband_baseline_cor(row).squeeze(),
            1,
            data.spectra
        )
        
        return SpectralData(
            spectra=corrected_spectra,
            wavenumbers=data.wavenumbers,
            features=data.features,
            metadata={**data.metadata, 'baseline_corrected': True}
        )


class SavitzkyGolayFilter(PreprocessingStep):
    """Savitzky-Golay smoothing/derivative filter."""
    
    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0):
        """
        Initialize Savitzky-Golay filter.
        
        Args:
            window_length: Length of filter window (must be odd)
            polyorder: Order of polynomial for fitting
            deriv: Order of derivative (0 for smoothing)
        """
        if window_length <= polyorder:
            raise ValueError("window_length must be greater than polyorder")
        
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
    
    def apply(self, data: SpectralData) -> SpectralData:
        """Apply Savitzky-Golay filter to each spectrum."""
        filtered_spectra = np.apply_along_axis(
            lambda row: savgol_filter(
                row,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv
            ),
            1,
            data.spectra
        )
        
        metadata = {
            **data.metadata,
            'savgol_window': self.window_length,
            'savgol_polyorder': self.polyorder,
            'savgol_deriv': self.deriv
        }
        
        return SpectralData(
            spectra=filtered_spectra,
            wavenumbers=data.wavenumbers,
            features=data.features,
            metadata=metadata
        )


class Normalization(PreprocessingStep):
    """Spectral normalization with various methods."""
    
    def __init__(self, method: Union[NormalizationType, str], margin: Optional[int] = None):
        """
        Initialize normalization.
        
        Args:
            method: Normalization method - can be NormalizationType enum or string ('amide', 'vector', 'snv')
            margin: Number of wavenumber points to exclude from each end when finding max for AMIDE normalization.
                   This helps avoid normalizing by noisy edge regions (e.g., 600-800 cm⁻¹) that may exceed 
                   the Amide I band intensity. Default is None (use all points).
        """
        if isinstance(method, str):
            try:
                self.method = NormalizationType(method.lower())
            except ValueError:
                valid_methods = [m.value for m in NormalizationType]
                raise ValueError(f"Invalid normalization method '{method}'. Valid options: {valid_methods}")
        elif isinstance(method, NormalizationType):
            self.method = method
        else:
            raise TypeError(f"method must be a string or NormalizationType, got {type(method)}")
        
        if margin is not None and margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = margin
    
    def apply(self, data: SpectralData) -> SpectralData:
        """Apply normalization to spectra."""
        if self.method == NormalizationType.AMIDE:
            if self.margin is not None and self.margin > 0:
                spectra_for_max = data.spectra[:, self.margin:-self.margin]
                max_values = spectra_for_max.max(axis=1, keepdims=True)
            else:
                max_values = data.spectra.max(axis=1, keepdims=True)
            normalized = data.spectra / max_values
        
        elif self.method == NormalizationType.VECTOR:
            norms = np.linalg.norm(data.spectra, axis=1, keepdims=True)
            normalized = data.spectra / norms
        
        elif self.method == NormalizationType.SNV:

            means = data.spectra.mean(axis=1, keepdims=True)
            stds = data.spectra.std(axis=1, keepdims=True)
            normalized = (data.spectra - means) / stds
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        metadata = {**data.metadata, 'normalization': self.method.value}
        if self.margin is not None:
            metadata['normalization_margin'] = self.margin
        
        return SpectralData(
            spectra=normalized,
            wavenumbers=data.wavenumbers,
            features=data.features,
            metadata=metadata
        )


class RegionSelector(PreprocessingStep):
    """Select specific wavenumber regions."""
    
    def __init__(
        self,
        regions: Sequence[Union[SpectralRegion, Tuple[float, float]]]
    ):
        """
        Initialize region selector.
        
        Args:
            regions: List of SpectralRegion objects or (start, end) tuples to select
        """
        normalized_regions = []
        
        for region in regions:
            if isinstance(region, SpectralRegion):
                if region.start >= region.end:
                    raise ValueError(f"Region start ({region.start}) must be less than end ({region.end})")
                normalized_regions.append(region)
            elif isinstance(region, (tuple, list)) and len(region) == 2:
                start, end = region[0], region[1]
                if start >= end:
                    raise ValueError(f"Region start ({start}) must be less than end ({end})")
                normalized_regions.append(SpectralRegion(start=start, end=end))
            else:
                raise ValueError(
                    f"Region must be a SpectralRegion or (start, end) tuple/list, got {type(region)}"
                )
        
        if not normalized_regions:
            raise ValueError("At least one region must be specified")
        
        self.regions = normalized_regions
    
    def apply(self, data: SpectralData) -> SpectralData:
        """Select specified wavenumber regions."""

        mask = np.zeros_like(data.wavenumbers, dtype=bool)
        for region in self.regions:
            mask |= region.to_mask(data.wavenumbers)
        
        selected_spectra = data.spectra[:, mask]
        selected_wavenumbers = data.wavenumbers[mask]
        
        metadata = {
            **data.metadata,
            'selected_regions': [(r.start, r.end) for r in self.regions]
        }
        
        return SpectralData(
            spectra=selected_spectra,
            wavenumbers=selected_wavenumbers,
            features=data.features,
            metadata=metadata
        )


class SampleFilter(PreprocessingStep):
    """Filter samples based on criteria."""
    
    def __init__(
        self,
        feature_min: Optional[dict] = None,
        feature_max: Optional[dict] = None,
        exclude_indices: Optional[List[int]] = None,
        max_absorbance: Optional[float] = None
    ):
        """
        Initialize sample filter.
        
        Args:
            feature_min: Dict mapping feature names to minimum values (e.g., {'HbA1c': 5.0, 'Age': 18})
            feature_max: Dict mapping feature names to maximum values (e.g., {'HbA1c': 12.0, 'Age': 80})
            exclude_indices: List of sample indices to exclude
            max_absorbance: Keep only samples with max absorbance >= this value
        """
        self.feature_min = feature_min or {}
        self.feature_max = feature_max or {}
        self.exclude_indices = exclude_indices or []
        self.max_absorbance = max_absorbance
    
    def apply(self, data: SpectralData) -> SpectralData:
        """Apply filters to data."""
        mask = np.ones(len(data), dtype=bool)
        
        for feature_name, min_value in self.feature_min.items():
            if feature_name in data.features:
                mask &= data.features[feature_name] >= min_value
        
        for feature_name, max_value in self.feature_max.items():
            if feature_name in data.features:
                mask &= data.features[feature_name] <= max_value
        
        if self.exclude_indices:
            mask[self.exclude_indices] = False
        
        if self.max_absorbance is not None:
            max_abs = data.spectra.max(axis=1)
            mask &= max_abs >= self.max_absorbance
        
        return data[mask]


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline with method chaining.
    
    Example:
        pipeline = PreprocessingPipeline()
        pipeline.add_step(SavitzkyGolayFilter(32, 2, 1))
        pipeline.add_step(Normalization(NormalizationType.VECTOR))
        pipeline.add_step(RegionSelector([SpectralRegion(800, 1800)]))
        
        processed_data = pipeline.apply(raw_data)
    """
    
    def __init__(self, steps: Optional[Sequence[PreprocessingStep]] = None):
        """
        Initialize pipeline with optional list of steps.
        
        Args:
            steps: Optional list of PreprocessingStep instances
        """
        self.steps: List[PreprocessingStep] = list(steps) if steps is not None else []
    
    def add_step(self, step: PreprocessingStep) -> 'PreprocessingPipeline':
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            step: PreprocessingStep instance
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self
    
    @overload
    def apply(self, data: SpectralDataset) -> SpectralDataset: ...

    @overload
    def apply(self, data: SpectralData) -> SpectralData: ...

    def apply(self, data):
        """
        Apply all preprocessing steps in sequence.
        
        Args:
            data: Input SpectralData or SpectralDataset
            
        Returns:
            Processed SpectralData or SpectralDataset (matching input type)
        """
        
        is_dataset = isinstance(data, SpectralDataset)
        result = data.data if is_dataset else data
        
        for step in self.steps:
            result = step.apply(result)
        
        return SpectralDataset(result) if is_dataset else result
    
    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self.steps)
    
    def __repr__(self) -> str:
        """String representation of pipeline."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"PreprocessingPipeline({' -> '.join(step_names)})"
