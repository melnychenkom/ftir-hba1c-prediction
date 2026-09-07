# Machine Learning Methods for HbA1c Prediction

This repository contains code accompanying the preprint [_"Decoupling Accuracy
and Explainability: Machine Learning Strategies for HbA1c Prediction and
Biomarker Discovery in Blood FTIR Spectroscopy"_](https://www.medrxiv.org/content/10.64898/2026.01.26.26344831v1).

<p align="center">
  <img src="spectrum.png" alt="FTIR Spectrum" width="800"/>
</p>

## Installation

```bash
git clone https://github.com/melnychenkom/ftir-hba1c-prediction.git
cd ftir-hba1c-prediction
```

### Option A: Poetry (Recommended)

Installs CPU-only versions automatically.

```bash
poetry install
```

### Option B: Manual / GPU

Use `pyproject.toml` for dependency reference.

```bash
pip install -e .
```

**Note**: This installation includes CPU-only versions of PyTorch and JAX. For GPU support, you'll need to install the appropriate CUDA-enabled versions separately.

**JAXFit**: This repository uses a personal fork of JAXFit with fixes for compatibility with newer JAX versions.

## Quick Start

### Loading Data

```python
from spectral_core import SpectralDataset

dataset = SpectralDataset.from_csv(
    file_path='data/dataset_681.csv',
    wavenumbers_path='data/wavenumbers.csv'
)

dataset.summary()

dataset.plot_spectra(feature='HbA1c')
```

### Preprocessing

```python
from spectral_core import (
    PreprocessingPipeline,
    SavitzkyGolayFilter,
    Normalization,
    RegionSelector,
    SampleFilter
)

pipeline = PreprocessingPipeline([
    SampleFilter(feature_max={"HbA1c": 14.0}, exclude_indices=[287, 636]),
    SavitzkyGolayFilter(window_length=33, polyorder=2, deriv=1),
    Normalization(method='vector'),
    RegionSelector(regions=[(800, 1800), (2800, 3400)])
])

preprocessed_data: SpectralData = pipeline.apply(dataset)
```

### Data Splitting

```python
from spectral_core import DataSplitter

splitter = DataSplitter(
    test_size=0.2,
    val_size=0.2,
    random_state=34
)

split = splitter.train_val_test_split(preprocessed_data)
splitter.save_split(split, output_dir='data/processed/')
```

## Models

### PLSR

```python
from spectral_core import PLSRComponents

plsr = PLSRComponents(target_name='HbA1c')
n_components = plsr.fit(
    calibration.spectra,
    calibration.get_feature('HbA1c'),
    ncomp=20,
    cv=10
)

fig, ax = plsr.evaluate(
    split.test.spectra,
    split.test.get_feature('HbA1c'),
    text_pos=(5.0, 12.0)
)
```

### 1D-CNN

For detailed CNN training workflow, see `notebooks/cnn.ipynb`.

#### Training

The CNN implementation uses PyTorch Lightning with a flexible CLI-based training system.

```bash
# Train with config file
python -m spectral_core.cnn1d.main fit --config src/spectral_core/cnn1d/config/train_config.yaml

# Or customize training parameters
python -m spectral_core.cnn1d.main fit \
    --model.input_size 3319 \
    --model.learning_rate 0.0001 \
    --data.data_folder data/processed/ \
    --trainer.max_epochs 300
```

### Curve Fitting with SpectraFit

For spectral curve fitting and peak analysis, see `notebooks/curvefit.ipynb`

The `SpectraFit` class uses JAXFit library for modeling spectral data. Peak positions should be identified beforehand.

```python
from spectral_core.curvefit.curvefit import SpectraFit

params = {
    "center": {"min": 3, "max": 3},
    "fwhm": {"value": 3},
    "eta": {"value": 0.5}
}

model = SpectraFit()
model.fit(x_values, y_values, peaks, param_dict=params, ftol=1e-12)

fig = model.plot_fit_plotly()
fig.show()
print(f"R²: {model.r2:.5f}, Discrepancy: {model.discrepancy:.3e}")
```

## Reproducibility

All stochastic steps are seeded. The values below are the ones used for the
published results; each is set in the file listed, not passed in at run time.

### Random states

| Stage                           | Value                                           | Set in                                                                                              |
| ------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Train/val/test split            | `random_state=34`                               | `notebooks/preprocessing.ipynb`; default in `DataSplitter` (`splitting.py`)                         |
| Stratification bins             | 8, uniform                                      | `DataSplitter(stratify_bins=8)`                                                                     |
| CNN training                    | seeds `0`–`9`; **seed 0 is the reported model** | `notebooks/cnn.ipynb` (`SEEDS`, `REPORTED_SEED`), applied via `seed_everything(seed, workers=True)` |
| CNN config default              | `seed_everything: 0`                            | `cnn1d/config/train_config.yaml`, `test_config.yaml`                                                |
| Benchmark grid search           | `random_state=34` (`KFold(shuffle=True)`)       | `notebooks/model_comparison.ipynb` (`RANDOM_STATE`)                                                 |
| Random Forest, XGBoost          | `random_state=34`                               | `baselines.py` (`DEFAULT_RANDOM_STATE`)                                                             |
| Bootstrap confidence intervals  | `np.random.default_rng(34)`                     | `notebooks/model_comparison.ipynb`                                                                  |
| Repeated-split robustness check | seeds `0`–`4`                                   | `notebooks/model_comparison.ipynb` (`REPEAT_SEEDS`)                                                 |
| H2O AutoML                      | `seed=1234`, `nfolds=5`                         | `notebooks/automl.ipynb`, `notebooks/automl_curvefit.py`                                            |

## Citation

Melnychenko, M., Makhnii, T., Midlovets, K., Dmyterchuk, B. & Krasnienkov, D.
Decoupling accuracy and explainability: machine learning strategies for HbA1c
prediction and biomarker discovery in blood FTIR spectroscopy. _medRxiv_
2026.01.26.26344831 (2026). <https://doi.org/10.64898/2026.01.26.26344831>

```bibtex
@article{melnychenko2026decoupling,
  title   = {Decoupling Accuracy and Explainability: Machine Learning Strategies
             for HbA1c Prediction and Biomarker Discovery in Blood FTIR Spectroscopy},
  author  = {Melnychenko, Mykola and Makhnii, Tatiana and Midlovets, Konstantin
             and Dmyterchuk, Bogdan and Krasnienkov, Dmytro},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.01.26.26344831},
  url     = {https://www.medrxiv.org/content/10.64898/2026.01.26.26344831v1},
  note    = {Preprint, posted 28 January 2026}
}
```
