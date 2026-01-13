from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import pandas as pd
import numpy as np

from .models import SpectralData


class SpectralDataLoader:
    """
    Factory class for loading spectral data from various sources.
    
    Provides static methods for loading data from:
    - CSV files
    - DataFrames
    - NumPy arrays
    - SPA format files
    """
    
    @staticmethod
    def _detect_feature_columns(df: pd.DataFrame) -> List[str]:
        """
        Auto-detect feature columns by checking if column names can be cast to float.
        Assumes wavenumber columns are numeric (can be cast to float).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that are likely feature variables
        """
        feature_cols = []
        for col in df.columns:
            try:
                float(col)
            except (ValueError, TypeError):
                feature_cols.append(col)
        return feature_cols
    
    @staticmethod
    def from_csv(
        file_path: Union[str, Path],
        wavenumbers_path: Union[str, Path],
        feature_columns: Union[str, List[str]] = 'auto',
        **kwargs
    ) -> SpectralData:
        """
        Load spectral data from CSV files.
        
        Args:
            file_path: Path to CSV file with spectra and features
            wavenumbers_path: Path to CSV file with wavenumber values
            feature_columns: Either 'auto' to auto-detect, or list of feature column names.
                          If 'auto', detects columns whose names can't be cast to float.
                          Default: 'auto'
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            SpectralData instance
            
        Raises:
            FileNotFoundError: If files don't exist
            ValueError: If columns are missing or invalid
            
        Examples:
            # Auto-detect feature columns (columns with non-numeric names)
            data = SpectralDataLoader.from_csv('spectra.csv', 'wavenumbers.csv')
            
            # Explicitly specify feature columns
            data = SpectralDataLoader.from_csv(
                'spectra.csv', 
                'wavenumbers.csv',
                feature_columns=['HbA1c', 'Age', 'Gender']
            )
        """
        file_path = Path(file_path)
        wavenumbers_path = Path(wavenumbers_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if not wavenumbers_path.exists():
            raise FileNotFoundError(f"Wavenumbers file not found: {wavenumbers_path}")
        
        data = pd.read_csv(file_path, **kwargs)
        
        if feature_columns == 'auto':
            feature_cols = SpectralDataLoader._detect_feature_columns(data)
            if not feature_cols:
                raise ValueError("No feature columns detected. All columns appear to be numeric (wavenumbers).")
        else:
            feature_cols = feature_columns if isinstance(feature_columns, list) else [feature_columns]
            for col in feature_cols:
                if col not in data.columns:
                    raise ValueError(f"Feature column '{col}' not found in data")
        
        features_dict = {}
        for col in feature_cols:
            features_dict[col] = data[col].to_numpy()
        
        spectra_cols = [col for col in data.columns if col not in feature_cols]
        spectra = data[spectra_cols].to_numpy()
        
        wavenumbers = pd.read_csv(wavenumbers_path, **kwargs).to_numpy().flatten()
        
        if len(wavenumbers) != spectra.shape[1]:
            raise ValueError(
                f"Wavenumber count ({len(wavenumbers)}) doesn't match "
                f"spectra features ({spectra.shape[1]})"
            )
        
        metadata = {
            'source': 'csv',
            'file_path': str(file_path),
            'wavenumbers_path': str(wavenumbers_path),
            'feature_columns': feature_cols
        }
        
        return SpectralData(
            spectra=spectra,
            wavenumbers=wavenumbers,
            features=features_dict,
            metadata=metadata
        )
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        wavenumbers: Union[np.ndarray, pd.Series],
        feature_columns: Union[str, List[str]] = 'auto'
    ) -> SpectralData:
        """
        Load spectral data from a pandas DataFrame.
        
        Args:
            df: DataFrame with spectra and feature columns
            wavenumbers: Array or Series of wavenumber values
            feature_columns: Either 'auto' to auto-detect, or list of feature column names
            
        Returns:
            SpectralData instance
            
        Examples:
            # Auto-detect (columns with non-numeric names)
            data = SpectralDataLoader.from_dataframe(df, wavenumbers)
            
            # Custom features
            data = SpectralDataLoader.from_dataframe(
                df, wavenumbers, 
                feature_columns=['HbA1c', 'Age', 'BMI']
            )
        """
        if feature_columns == 'auto':
            feature_cols = SpectralDataLoader._detect_feature_columns(df)
            if not feature_cols:
                raise ValueError("No feature columns detected. All columns appear to be numeric (wavenumbers).")
        else:
            feature_cols = feature_columns if isinstance(feature_columns, list) else [feature_columns]
            for col in feature_cols:
                if col not in df.columns:
                    raise ValueError(f"Feature column '{col}' not found")
        
        features_dict = {col: df[col].to_numpy() for col in feature_cols}
        
        spectra_cols = [col for col in df.columns if col not in feature_cols]
        spectra = df[spectra_cols].to_numpy()
        
        return SpectralData(
            spectra=spectra,
            wavenumbers=np.asarray(wavenumbers).flatten(),
            features=features_dict,
            metadata={'source': 'dataframe', 'feature_columns': feature_cols}
        )
    
    @staticmethod
    def from_arrays(
        spectra: np.ndarray,
        wavenumbers: np.ndarray,
        features: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> SpectralData:
        """
        Load spectral data from numpy arrays.
        
        Args:
            spectra: 2D array of spectra (n_samples, n_features)
            wavenumbers: 1D array of wavenumber values
            features: Dictionary mapping feature names to 1D arrays (e.g., {'HbA1c': array, 'Age': array})
            metadata: Optional metadata dictionary
            
        Returns:
            SpectralData instance
            
        Example:
            data = SpectralDataLoader.from_arrays(
                spectra=X,
                wavenumbers=wn,
                features={'HbA1c': y_hba1c, 'Age': y_age, 'BMI': y_bmi}
            )
        """
        if metadata is None:
            metadata = {'source': 'arrays'}
        else:
            metadata = {'source': 'arrays', **metadata}
        
        return SpectralData(
            spectra=spectra,
            wavenumbers=wavenumbers,
            features=features,
            metadata=metadata
        )
    
    @staticmethod
    def from_npy(
        spectra_path: Union[str, Path],
        wavenumbers_path: Union[str, Path],
        features_paths: Dict[str, Union[str, Path]]
    ) -> SpectralData:
        """
        Load spectral data from .npy files.
        
        Args:
            spectra_path: Path to .npy file with spectra
            wavenumbers_path: Path to .npy file with wavenumbers
            features_paths: Dictionary mapping feature names to .npy file paths
                         (e.g., {'HbA1c': 'y_hba1c.npy', 'Age': 'y_age.npy'})
            
        Returns:
            SpectralData instance
            
        Example:
            data = SpectralDataLoader.from_npy(
                spectra_path='X.npy',
                wavenumbers_path='wn.npy',
                features_paths={'HbA1c': 'y_hba1c.npy', 'Age': 'y_age.npy'}
            )
        """
        spectra = np.load(spectra_path)
        wavenumbers = np.load(wavenumbers_path)
        
        features_dict = {}
        for feature_name, feature_path in features_paths.items():
            features_dict[feature_name] = np.load(feature_path)
        
        metadata = {
            'source': 'npy',
            'spectra_path': str(spectra_path),
            'wavenumbers_path': str(wavenumbers_path),
            'features_paths': {k: str(v) for k, v in features_paths.items()}
        }
        
        return SpectralData(
            spectra=spectra,
            wavenumbers=wavenumbers,
            features=features_dict,
            metadata=metadata
        )
    
    @staticmethod
    def from_spa(file_path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a *.spa file and return spectra and wavenumbers.
        
        Note: SPA files don't contain HbA1c/age metadata, so this returns
        raw arrays that can be combined with metadata separately.
        
        Args:
            file_path: Path to the *.spa file
            
        Returns:
            Tuple of (wavenumbers, spectra) as 1D arrays
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"SPA file not found: {file_path}")
        
        with open(file_path, 'rb') as file:
            file.seek(564)
            spectrum_points = np.fromfile(file, np.int32, 1)[0]
            
            file.seek(576)
            max_wavenumber = np.fromfile(file, np.single, 1)[0]
            min_wavenumber = np.fromfile(file, np.single, 1)[0]
            
            wavenumbers = np.flip(np.linspace(min_wavenumber, max_wavenumber, spectrum_points))
            
            file.seek(288)
            flag = 0
            while flag != 3:
                flag = np.fromfile(file, np.uint16, 1)
            
            data_position = np.fromfile(file, np.uint16, 1)
            file.seek(data_position[0])
            
            spectra = np.fromfile(file, np.single, spectrum_points)
        return wavenumbers, spectra
