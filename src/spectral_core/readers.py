import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split

# TODO: implement more flexible reader with options to select which parameters to load
class FitReader:
    """
    Class for fit reading produced by SpectraFit class
    """
    def __init__(self, dir_path, peaks_path):
        self.dir_path = dir_path
        self.peaks_path = peaks_path
        self._load_fits()

    def _load_fits(self):
        self.wavenumbers = []
        self.fwhm = []
        self.amplitude = []
        self.eta = []
        self.height = []
        self.area = []
        self.area_abs = []
        self.hba1c = []
        self.ages = []

        for filename in os.listdir(self.dir_path):
            name, extension = os.path.splitext(filename)
            if extension == '.csv' and name[0].isdigit():
                target = name.split("_")[2]
                age = name.split("_")[3]

                df = pd.read_csv(os.path.join(self.dir_path, filename))

                self.hba1c.append(float(target))
                self.ages.append(float(age))

                self.wavenumbers.append(df['wavenumber'].values)
                self.fwhm.append(df['FWHM'].values)
                self.amplitude.append(df['amplitude'].values)
                self.eta.append(df['eta'].values)
                self.height.append(df['height'].values)
                self.area.append(df['area'].values)
                self.area_abs.append(df['area_absolute'].values)

        try:
            self.mean_spectra = pd.read_csv(os.path.join(self.dir_path, 'mean_spectra.csv'))
        except FileNotFoundError:
             print("mean_spectra.csv has not been found. Shiftws relative to the mean spectra will not be created.")
             self.mean_spectra = None
    
        self.fit_quality = pd.read_csv(os.path.join(self.dir_path, 'fit_info.txt'), sep='\t')

        self.hba1c = np.array(self.hba1c)

        peaks = pd.read_csv(self.peaks_path).values.reshape(-1).astype(np.int32)

        self.wavenumbers = pd.DataFrame(self.wavenumbers, columns=peaks)
        self.wavenumbers['HbA1c'] = self.hba1c
        self.wavenumbers['Age'] = self.ages

        if self.mean_spectra is not None:
            self.wavenumber_shifts = pd.DataFrame(self.wavenumbers.iloc[:, :-2] - \
                                                  self.mean_spectra['wavenumber'].values, columns=peaks)
            self.wavenumber_shifts['HbA1c'] = self.hba1c
            self.wavenumber_shifts['Age'] = self.ages
        
        # TODO: refactor repetitive code
        self.fwhm = pd.DataFrame(self.fwhm, columns=peaks)
        self.fwhm['HbA1c'] = self.hba1c
        self.fwhm['Age'] = self.ages

        self.amplitude = pd.DataFrame(self.amplitude, columns=peaks)
        self.amplitude['HbA1c'] = self.hba1c
        self.amplitude['Age'] = self.ages

        self.eta = pd.DataFrame(self.eta, columns=peaks)
        self.eta['HbA1c'] = self.hba1c
        self.eta['Age'] = self.ages

        self.height = pd.DataFrame(self.height, columns=peaks)
        self.height['HbA1c'] = self.hba1c
        self.height['Age'] = self.ages

        self.areas = pd.DataFrame(self.area, columns=peaks)
        self.areas['HbA1c'] = self.hba1c
        self.areas['Age'] = self.ages

        self.areas_abs = pd.DataFrame(self.area_abs, columns=peaks)
        self.areas_abs['HbA1c'] = self.hba1c
        self.areas_abs['Age'] = self.ages

        return None

    def dataset_split(
            self, 
            dataframe_name='areas', 
            normalization=True,
            bins=8, 
            test_size=0.2, 
            target='HbA1c', 
            random_state=34
            ):
            """
            Splits a specified dataset inside self.<dataframe> into training and testing sets.
            """

            data = getattr(self, dataframe_name)

            X = data.drop(columns=['HbA1c', 'Age'])
            y = data[target]

            discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform', random_state=random_state)
            categories = discretizer.fit_transform(data[target].values.reshape(-1, 1))

            if normalization:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size=test_size, 
                stratify=categories, 
                random_state=random_state
            )

            return X_train, X_test, y_train, y_test

    # TODO: reimplement with a generlized method for filtering by any feature
    def select_max_hba1c(self, hba1c):
        self.wavenumbers = self.wavenumbers[self.wavenumbers['HbA1c'] <= hba1c]
        self.fwhm = self.fwhm[self.fwhm['HbA1c'] <= hba1c]
        self.amplitude = self.amplitude[self.amplitude['HbA1c'] <= hba1c]
        self.eta = self.eta[self.eta['HbA1c'] <= hba1c]
        self.height = self.height[self.height['HbA1c'] <= hba1c]
        self.areas = self.areas[self.areas['HbA1c'] <= hba1c]
        self.areas_abs = self.areas_abs[self.areas_abs['HbA1c'] <= hba1c]
        self.fit_quality = self.fit_quality[self.fit_quality['HbA1c'] <= hba1c]

        if self.mean_spectra is not None:
            self.wavenumber_shifts = self.wavenumber_shifts[self.wavenumber_shifts['HbA1c'] <= hba1c]

        return None

# TODO: why is this function here? move to another file?
def read_spa(filepath: str) -> np.ndarray:
    """
    Read Thermo Nicolet .spa file format.
    
    Args:
        filepath: Path to .spa file
        
    Returns:
        2D array with shape (n_points, 2) where columns are [wavenumbers, spectra]
    """
    with open(filepath, 'rb') as file:
        file.seek(564)
        spectrum_points = np.fromfile(file, np.int32, 1)[0]
        
        file.seek(30)
        spectra_titles = np.fromfile(file, np.uint8, 255)
        spectra_titles = ''.join([chr(x) for x in spectra_titles if x != 0])

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
        
    return np.stack((wavenumbers, spectra), axis=1)