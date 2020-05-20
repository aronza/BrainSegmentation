import logging

import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from .dataset import BrainDataset


def pick_files(file_range, picked_files, ratio, name):

    if picked_files is not None:
        other_files = [i for i in file_range if i not in picked_files]
    elif ratio >= 1.0:
        picked_files = file_range
        other_files = []
    elif ratio > 0.0:
        other_files, picked_files = train_test_split(file_range, test_size=ratio)
    else:
        other_files = file_range

    if picked_files is not None:
        logging.info(f'{name} images: {picked_files}')
    else:
        logging.info(f'{name} skipped')

    return picked_files, other_files


def expand_file_indices(files, num_slices):
    if files is None:
        return []
    return [idx for file_no in files for idx in range(file_no * num_slices, (file_no + 1) * num_slices)]


class BrainLoaders:

    def __init__(self, dataset: BrainDataset, ratios=None, files=None):
        """
                Splits the data set into training, validation and testing based on the given ratios and returns
                    data loaders pointing to these subsets.

                    :param validation_ratio: Percentages of whole data set to be used for validation
                    :param test_ratio: Percentages of whole data set to be used for testing
                    :param test_files: Use given files as test.
                    :param shuffle: If the data in loader will be shuffled

                    :return: List of torch.utils.data.DataLoader in order of training, validation and testing.
        """

        if ratios is None:
            validation_ratio, test_ratio = [0.0, 0.0]
        else:
            validation_ratio, test_ratio = ratios

        if files is None:
            validation_files, test_files = [None, None]
        else:
            validation_files, test_files = files

        num_files = range(dataset.num_files())
        test_files, train_files = pick_files(num_files, test_files, test_ratio, "Test")

        if test_ratio != 1:
            val_train_ratio = validation_ratio / (1 - test_ratio)
        else:
            val_train_ratio = 0
        validation_files, train_files = pick_files(train_files, validation_files, val_train_ratio, "Validation")

        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset

    def train_loader(self) -> DataLoader:
        train_indices = expand_file_indices(self.train_files, self.dataset.num_slices())
        self.dataset.set_training_set(train_indices)
        return DataLoader(Subset(self.dataset, train_indices),
                          batch_size=1,
                          shuffle=True,
                          num_workers=0 if self.device.type == 'cuda' else 8,
                          pin_memory=self.device.type != 'cuda')

    def validation_loader(self) -> DataLoader:
        return DataLoader(Subset(self.dataset, expand_file_indices(self.validation_files, self.dataset.num_slices())),
                          batch_size=self.dataset.num_slices(),
                          shuffle=False,
                          num_workers=0 if self.device.type == 'cuda' else 8,
                          pin_memory=self.device.type != 'cuda')

    def test_loader(self) -> DataLoader:
        return DataLoader(Subset(self.dataset, expand_file_indices(self.test_files, self.dataset.num_slices())),
                          batch_size=self.dataset.num_slices(),
                          shuffle=False,
                          num_workers=0 if self.device.type == 'cuda' else 8,
                          pin_memory=self.device.type != 'cuda')
