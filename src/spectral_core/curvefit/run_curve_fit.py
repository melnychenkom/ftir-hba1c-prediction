import sys
sys.path.append('../preprocessing')

import argparse
import os
import pandas as pd
import json
from tqdm.auto import tqdm
from curvefit import SpectraFit
from dataset import DatasetSpectra
import matplotlib.pyplot as plt
import time


CONFIG = {
    "START": 1000,
    "END": 1720,
    "MAX_HBA1C": 14,
    "NORMALIZATION": 'amide',
    "START_INDEX": 0,
    "SAVGOL_PARAMS": {
        "window_length": 5,
        "polyorder": 3
    }
}

def main(data_path, domain_path, peaks_path, output_dir, fit_sample, data_dir):
    data_path = os.path.expanduser(os.path.join(data_dir, data_path))
    domain_path = os.path.expanduser(os.path.join(data_dir, domain_path))
    peaks_path = os.path.expanduser(os.path.join(data_dir, peaks_path))
    output_dir = os.path.expanduser(os.path.join(data_dir, output_dir))

    peaks = pd.read_csv(peaks_path).to_numpy().reshape(-1)

    dataset = DatasetSpectra(data_path, domain_path)
    dataset.select_max_hba1c(CONFIG["MAX_HBA1C"])
    dataset.savgol_filter(**CONFIG["SAVGOL_PARAMS"])
    dataset.baseline_corr()
    dataset.normalization(kind=CONFIG["NORMALIZATION"])
    dataset.select_region([CONFIG["START"], CONFIG["END"]])

    x_values = dataset.get_wavenumbers()

    model = SpectraFit()

    if fit_sample:
        spectra, hba1c, age = dataset[fit_sample]
        model.fit(x_values, spectra, peaks)
        print(f"HbA1c: {hba1c}, Age: {int(age)}")
        model.plot_fit()
        plt.show()
        sys.exit(0)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    x_values = dataset.get_wavenumbers()

    params = {
        "center": {"min": 3, "max": 3},
        "fwhm": {"value": 3},
        "eta": {"value": 0.5}
    }

    metadata = {
        "Normalization": CONFIG["NORMALIZATION"],
        "Selected region": f"{CONFIG['START']} to {CONFIG['END']}",
        "Peaks": peaks.tolist(),
        "Parameters": params
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    fit_info_path = os.path.join(output_dir, 'fit_info.txt')

    if not CONFIG["START_INDEX"]:
        with open(fit_info_path, 'a') as fit_info_file:
            fit_info_file.write(f"Index\tr2\tDiscrepancy\tHbA1c\tAge\tTime\n")

    total_samples = dataset.n_samples - CONFIG["START_INDEX"]

    spectra = dataset.spectra.mean(axis=0)
    pmodel = SpectraFit()
    pmodel.fit(x_values, spectra, peaks, params, ftol=1e-12)
    pmodel.params.to_csv(os.path.join(output_dir, 'mean_spectra.csv'), index=None)

    with tqdm(total=total_samples) as pbar:
        for i in range(CONFIG["START_INDEX"], len(dataset)):
            spectra, hba1c, age = dataset[i]

            model = SpectraFit()
            model.load_model(pmodel)

            try:
                start_time = time.time()
                model.fit(x_values, spectra, peaks, params)
                end_time = time.time()
                fit_time = end_time - start_time 

                filename = f"{i}_fit_{hba1c}_{age}.csv"
                output_path = os.path.join(output_dir, filename)
                model.params.to_csv(output_path, index=None)

                with open(fit_info_path, 'a') as fit_info_file:
                    fit_info_file.write(f"{i}\t{model.r2:.5f}\t{model.discrepancy:.5e}\t{hba1c}\t{age}\t{fit_time:.2f}\n")

            except Exception as e:
                message = f"Error at index {i}: {e}"
                with open(fit_info_path, 'a') as fit_info_file:
                    fit_info_file.write(f"{i}\tNone\tNone\tNone\tNone\tNone\n") 
                print(message)

            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spectral data.")
    parser.add_argument('--data_dir', type=str, default='~/projects/ml_methods_hba1c/data/', help="Path to the data")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--domain_path', type=str, required=True, help="Path to the domain CSV file")
    parser.add_argument('--peaks_path', type=str, required=True, help="Path to the peaks CSV file")
    parser.add_argument('--output_dir', type=str, default='fits/', help="Directory to save fit info")
    parser.add_argument('--fit_sample', type=int, default=None, help="Number of the sample to fit")
    
    args = parser.parse_args()
    main(args.data_path, args.domain_path, args.peaks_path, args.output_dir, args.fit_sample, args.data_dir)