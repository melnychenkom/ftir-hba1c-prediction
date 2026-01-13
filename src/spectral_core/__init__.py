"""
Spectral Core - Modern spectral data processing library.

This package provides clean, composable tools for loading, preprocessing,
and analyzing spectral data, particularly FTIR spectra for HbA1c prediction.
"""

# Core data models
from .models import (
    SpectralData,
    SpectralRegion,
    NormalizationType,
    PreprocessingConfig,
    DataSplitConfig,
    DataSplit,
)

# Data loading
from .loaders import SpectralDataLoader

# Preprocessing
from .preprocessing import (
    PreprocessingPipeline,
    PreprocessingStep,
    BaselineCorrection,
    SavitzkyGolayFilter,
    Normalization,
    RegionSelector,
    SampleFilter,
)

# Data splitting
from .splitting import DataSplitter

# Dataset wrapper with visualization
from .dataset import SpectralDataset

# PLSR modeling
from .plsr import PLSRComponents


__version__ = '2.0.0'

__all__ = [
    'SpectralData',
    'SpectralRegion',
    'NormalizationType',
    'PreprocessingConfig',
    'DataSplitConfig',
    'DataSplit',
    'SpectralDataLoader',
    'PreprocessingPipeline',
    'PreprocessingStep',
    'BaselineCorrection',
    'SavitzkyGolayFilter',
    'Normalization',
    'RegionSelector',
    'SampleFilter',
    'DataSplitter',
    'SpectralDataset',
    'PLSRComponents',
]
