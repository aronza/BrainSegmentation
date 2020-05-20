import logging
from glob import glob
from os import listdir
from os.path import join

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader

from unet3d.model import UNet3D
from .slicer import build_slices
from .transforms import RandomRotate, RandomFlip, RandomContrast

GLOBAL_RANDOM_STATE = np.random.RandomState(47)


def get_nii_files(directory, tag):
    path = join(directory, tag) + '*'
    # mask_path = self.masks_dir + tag.replace(self.img_prefix, self.mask_prefix) + '*'

    file = glob(path)

    assert len(file) == 1, \
        f'Either no file or multiple files found for the ID {directory}: {file}'

    return nib.load(file[0])

    # return img.dataobj


def standardize(m, mean, std):
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    """
    return (m - mean) / np.clip(std, a_min=1e-6, a_max=None)


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img for img in images]
    )
    return np.mean(flat), np.std(flat), (np.count_nonzero(flat) * 100.0) / flat.size


class BrainDataset(Dataset):
    def __init__(self, imgs_dir, img_postfix='T1', masks_dir=None, stack_size=16, stride=14, mask_net=False,
                 verbose=False):
        self.img_filenames = listdir(imgs_dir)

        tags = [file[:file.find(img_postfix)] if file.find(img_postfix) != -1 else file[:file.find('.')]
                for file in self.img_filenames if not file.startswith('.')]

        self.img_nibs = [get_nii_files(imgs_dir, tag) for tag in tags]

        self.img_files = [img.dataobj for img in self.img_nibs]
        self.mask_files = None if masks_dir is None else [get_nii_files(masks_dir, tag).dataobj for tag in tags]

        self.input_shape = self.img_files[0].shape
        patch_shape = (stack_size, self.input_shape[1], self.input_shape[2])
        stride_shape = (stride, self.input_shape[1], self.input_shape[2])
        self.train_indices = []

        logging.info(f'Input shape: {self.input_shape}')
        logging.info(f'Patch shape: {patch_shape}')

        self.slices = build_slices(self.input_shape,
                                   patch_shape=patch_shape,
                                   stride_shape=stride_shape)

        logging.info(f'Creating dataset from {imgs_dir} and {masks_dir}'
                     f'\nWith {len(self.img_files)} examples and {len(self.slices)} slices each')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.length = len(self.img_files) * len(self.slices)
        self.verbose = verbose
        if mask_net:
            logging.info(f'Loading Pre-process Network from {mask_net}')
            pre_net = UNet3D(in_channels=1, out_channels=2, final_sigmoid=False, testing=True)
            pre_net.load_state_dict(torch.load(mask_net, map_location=self.device))
            pre_net.to(device=self.device)
            logging.info(f'Pre-process Network Loaded')
            self.mask_net = pre_net
        else:
            self.mask_net = None

    def num_slices(self):
        return len(self.slices)

    def num_files(self):
        return len(self.img_files)

    def set_training_set(self, train_indices):
        self.train_indices = train_indices

    def __len__(self):
        return self.length

    def get_nib_file(self, file_idx):
        return self.img_nibs[file_idx]

    @classmethod
    def pre_process(cls, img_nd, is_label, device, phase='test', random_state=None):
        if not is_label:
            img_nd = np.expand_dims(img_nd, axis=0)

        if random_state is not None and phase == 'train':
            transforms = [RandomFlip(random_state), RandomRotate(random_state), RandomContrast(random_state)]

            for transform in transforms:
                img_nd = transform(img_nd)

        img_nd = torch.from_numpy(img_nd.copy())

        img_nd = img_nd.to(device=device, dtype=torch.int64 if is_label else torch.float32)  #

        return img_nd

    @classmethod
    def post_process(cls, img: torch.Tensor):
        assert img.size()[0] == 1

        class_dim = img.dim() - 4
        return img.argmax(class_dim).byte().squeeze()

    def __getitem__(self, idx):
        file_idx = idx // len(self.slices)
        slice_idx = idx % len(self.slices)

        if self.verbose:
            logging.info(f"{idx}: Getting slice {slice_idx} from image {file_idx}")
        phase = 'train' if idx in self.train_indices else 'val'
        seed = GLOBAL_RANDOM_STATE.randint(10000000)

        _slice = self.slices[slice_idx]

        #  Only loads the sliced img to data
        img = self.pre_process(self.img_files[file_idx][_slice], is_label=False, device=self.device,
                               phase=phase, random_state=np.random.RandomState(seed))

        if self.mask_net is not None:
            with torch.no_grad():
                # count_percentage(img, "Pre Img")
                output = self.mask_net(img.unsqueeze(1))
                pre_mask = self.post_process(output)
                # count_percentage(pre_mask, "Mask")
                img = pre_mask * img
                # count_percentage(img, "Post Img")

        else:
            pre_mask = None

        if self.mask_files is not None:
            mask = self.pre_process(self.mask_files[file_idx][_slice], is_label=True, device=self.device,
                                    phase=phase, random_state=np.random.RandomState(seed))
            if pre_mask is not None:
                mask = pre_mask * mask
            return {'image': img, 'mask': mask, 'file_idx': file_idx, 'slice_idx': slice_idx}

        return {'image': img, 'file_idx': file_idx, 'slice_idx': slice_idx}
