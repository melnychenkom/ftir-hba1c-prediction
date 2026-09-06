from typing import Optional, Union
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from spectral_core.models import SpectralData, DataSplit
from spectral_core.dataset import SpectralDataset

class DataSplitter:
    """
    Handles splitting of spectral data with stratification.
    
    Provides methods for creating train/test and train/val/test splits
    with stratification based on feature variable bins.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_bins: int = 8,
        random_state: int = 34
    ):
        """
        Initialize data splitter.
        
        Args:
            test_size: Proportion of data for test set (0 < test_size < 1)
            val_size: Proportion of data for validation set (optional)
            stratify_bins: Number of bins for stratification
            random_state: Random seed for reproducibility
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if val_size is not None and not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1")
        if val_size is not None and (test_size + val_size) >= 1:
            raise ValueError("test_size + val_size must be less than 1")
        
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_bins = stratify_bins
        self.random_state = random_state
    
    def train_test_split(
        self,
        data: Union[SpectralData, SpectralDataset],
        feature: str = 'HbA1c'
    ) -> DataSplit:
        """
        Split data into train and test sets with stratification.
        
        Args:
            data: SpectralData to split
            feature: Name of feature column for stratification (default: 'HbA1c')
            
        Returns:
            DataSplit with train and test sets
        """
        if feature not in data.features:
            raise ValueError(f"Feature '{feature}' not found in data. Available: {list(data.features.keys())}")
        
        feature_values = data.features[feature]
        
        discretizer = KBinsDiscretizer(
            n_bins=self.stratify_bins,
            encode='ordinal',
            strategy='uniform',
            random_state=self.random_state
        )
        categories = discretizer.fit_transform(feature_values.reshape(-1, 1)).flatten()
        
        indices = np.arange(len(data))
        
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=categories,
            random_state=self.random_state
        )
        train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
        
        return DataSplit(
            train=data[train_idx],
            test=data[test_idx],
            train_idx=train_idx,
            test_idx=test_idx
        )
    
    def train_val_test_split(
        self,
        data: Union[SpectralData, SpectralDataset],
        feature: str = 'HbA1c'
    ) -> DataSplit:
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            data: SpectralData to split
            feature: Name of feature column for stratification (default: 'HbA1c')
            
        Returns:
            DataSplit with train, val, and test sets
        """
        if self.val_size is None:
            raise ValueError("val_size must be specified for train/val/test split")
        
        if feature not in data.features:
            raise ValueError(f"Feature '{feature}' not found in data. Available: {list(data.features.keys())}")
        
        feature_values = data.features[feature]
        
        discretizer = KBinsDiscretizer(
            n_bins=self.stratify_bins,
            encode='ordinal',
            strategy='uniform',
            random_state=self.random_state
        )
        categories = discretizer.fit_transform(feature_values.reshape(-1, 1)).flatten()
        
        indices = np.arange(len(data))
        
        temp_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=categories,
            random_state=self.random_state
        )
        temp_idx, test_idx = np.asarray(temp_idx), np.asarray(test_idx)
        
        val_size_relative = self.val_size / (1 - self.test_size)
        
        train_idx, val_idx = train_test_split(
            temp_idx,
            test_size=val_size_relative,
            stratify=categories[temp_idx],
            random_state=self.random_state
        )
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        
        return DataSplit(
            train=data[train_idx],
            val=data[val_idx],
            test=data[test_idx],
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )
    
    @staticmethod
    def save_split(
        split: DataSplit,
        output_dir: str,
        prefix: str = "",
        feature: str = 'HbA1c'
    ) -> None:
        """
        Save data split to .npy files.

        Writes spectra, the selected feature and the wavenumber axis for every
        subset, plus the positions each subset occupied in the data that was
        split, so the same split can be recovered without re-running it.

        Args:
            split: DataSplit to save
            output_dir: Directory to save files
            prefix: Optional prefix for filenames (e.g., 'cnn_', 'plsr_')
            feature: Feature column name to save (default: 'HbA1c')
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if feature not in split.train.features:
            raise ValueError(f"Feature '{feature}' not found. Available: {list(split.train.features.keys())}")
        
        np.save(output_path / f"{prefix}X_train.npy", split.train.spectra)
        np.save(output_path / f"{prefix}y_train.npy", split.train.features[feature])
        
        np.save(output_path / f"{prefix}X_test.npy", split.test.spectra)
        np.save(output_path / f"{prefix}y_test.npy", split.test.features[feature])
        
        if split.val is not None:
            np.save(output_path / f"{prefix}X_val.npy", split.val.spectra)
            np.save(output_path / f"{prefix}y_val.npy", split.val.features[feature])
        
        np.save(output_path / f"{prefix}wavenumbers.npy", split.train.wavenumbers)

        for name, indices in (
            ('train', split.train_idx),
            ('val', split.val_idx),
            ('test', split.test_idx)
        ):
            if indices is not None:
                np.save(output_path / f"{prefix}{name}_idx.npy", indices)

        print(f"Data split saved to {output_dir}")
        print(f"  Train samples: {len(split.train)}")
        if split.val is not None:
            print(f"  Val samples: {len(split.val)}")
        print(f"  Test samples: {len(split.test)}")
