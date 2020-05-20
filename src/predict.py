#!/usr/bin/env python3

import argparse
import logging
import timeit
from os.path import join
from pathlib import Path

import torch

from dataset.dataset import BrainDataset
from dataset.loader import BrainLoaders
from dataset.slicer import Sticher

from unet3d.model import UNet3D
from unet3d.utils import make_dir


def predict(model: UNet3D, brain_loader: BrainLoaders, out_files):
    model.eval()

    loader = brain_loader.test_loader()
    with torch.no_grad():
        for batch in loader:
            epoch_start_time = timeit.default_timer()

            img_batch = batch['image']
            file_idx = batch['file_idx'][0].item()
            slices = batch['slice_idx']

            logging.info("\nPredicting image {} ...".format(file_idx))

            result = Sticher(brain_loader.dataset.input_shape, brain_loader.dataset.slices)

            for img_patch, slice_idx in zip(img_batch, slices):
                img_patch = img_patch.unsqueeze(0)

                output = model(img_patch)

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(model, 'final_activation') and model.final_activation is not None and not model.testing:
                    output = model.final_activation(output)

                output_patch = BrainDataset.post_process(output)

                result.update(output_patch, slice_idx)

            result.save(out_files[file_idx], brain_loader.dataset.get_nib_file(file_idx))
            elapsed = timeit.default_timer() - epoch_start_time
            logging.info(f'Predicted in {elapsed} seconds')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type', '-t', dest='type', type=str,
                        help="mask = Brainmask segementation, wm = WhiteMatter segmentation")
    parser.add_argument('--model', '-m', metavar='FILE', type=str, default=False,
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='directory of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='directory of output files', required=True)
    parser.add_argument('-p', '--patch-size', metavar='P', type=int, default=16,
                        help='Patch Size', dest='patch_size')
    parser.add_argument('-n', '--name', type=str, default='OUT',
                        help='Postfix to append to output filenames', dest='name')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    input_channels = 1
    output_channels = 2
    net = UNet3D(in_channels=input_channels, out_channels=output_channels, testing=True, final_sigmoid=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    logging.info(f'Using device {device}')

    base_path = Path(__file__).parent

    file_path = (base_path / "../models/test.csv").resolve()
    pre_net = False
    if args.type == 'mask':
        model = (base_path / "../models/BrainMask.pth").resolve()
    elif args.type == 'wm':
        model = (base_path / "../models/WhiteMatter.pth").resolve()
        pre_net = (base_path / "../models/BrainMask.pth").resolve()
    else:
        model = args.model

    assert model is not None and model
    logging.info("Loading model {}".format(model))
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("Model loaded!")

    data_set = BrainDataset(args.input, stack_size=args.patch_size, mask_net=pre_net)
    loaders = BrainLoaders(data_set, ratios=[0.0, 1.0])

    make_dir(args.output)
    output_files = {}

    for idx, filename in enumerate(data_set.img_filenames):
        extension_idx = filename.find('.')

        postfix_idx = filename.find('T1')
        if postfix_idx == -1:
            postfix_idx = extension_idx

        output_files[idx] = join(args.output, filename[:postfix_idx] + args.name + filename[extension_idx:])

    predict(model=net, brain_loader=loaders, out_files=output_files)
