import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

class SpectraDataset(Dataset):
    def __init__(self, spectra, target):
        self.spectra = spectra
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.spectra[idx], self.target[idx]
    

class SpectraDataModule(L.LightningDataModule):
    def __init__(self, data_folder, batch_size=32):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_data(self, filename):
        file_path = os.path.join(self.data_folder, filename)
        return np.load(file_path).astype(np.float32)

    def setup(self, stage=None):
        data_files = {
            "X_train": "X_train.npy",
            "y_train": "y_train.npy",
            "X_val": "X_val.npy",
            "y_val": "y_val.npy",
            "X_test": "X_test.npy",
            "y_test": "y_test.npy",
        }

        X_train = self._load_data(data_files["X_train"])
        y_train = self._load_data(data_files["y_train"])
        X_val = self._load_data(data_files["X_val"])
        y_val = self._load_data(data_files["y_val"])
        X_test = self._load_data(data_files["X_test"])
        y_test = self._load_data(data_files["y_test"])

        self.train_dataset = SpectraDataset(X_train, y_train)
        self.val_dataset = SpectraDataset(X_val, y_val)
        self.test_dataset = SpectraDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
