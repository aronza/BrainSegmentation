import argparse
from datetime import datetime
import logging
import sys
import torch

from dataset.dataset import BrainDataset
from dataset.loader import BrainLoaders

from unet3d.metrics import PatchValidator, METRICS
from unet3d.model import UNet3D
from unet3d.utils import RunningAverage


def validate(model: UNet3D, brain_loader: BrainLoaders, loss_fnc=None, is_validation=True, quiet=True):
    val_losses = RunningAverage()
    val_scores = {fnc: RunningAverage() for fnc in METRICS}

    loader = brain_loader.validation_loader() if is_validation else brain_loader.test_loader()

    with torch.no_grad():
        for batch in loader:
            img_batch = batch['image']
            mask_batch = batch['mask']
            file_idx = batch['file_idx'][0]

            patch_validator = PatchValidator(device=brain_loader.device, image_shape=brain_loader.dataset.input_shape)
            for slice_no, (img_patch, mask_patch) in enumerate(zip(img_batch, mask_batch)):
                img_patch = img_patch.unsqueeze(0)
                mask_patch = mask_patch.unsqueeze(0)

                # forward pass
                output = model(img_patch)

                # compute the loss
                if loss_fnc is not None:
                    loss = loss_fnc(output, mask_patch)
                    val_losses.update(loss.item(), n=1)

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(model, 'final_activation') and model.final_activation is not None and not model.testing:
                    output = model.final_activation(output)

                slices = brain_loader.dataset.slices
                patch_validator.update(output, mask_patch, patch_slice=slices[slice_no % len(slices)])

            # When we finish an image, calculate the score for it and update mean
            if not quiet:
                logging.info(f'Image:{file_idx}:')
            for fnc in METRICS:
                score = patch_validator.calculate_fnc(fnc)
                val_scores[fnc].update(score, n=1)

                if not quiet:
                    logging.info(f"\t\t{fnc+' Score:':<30}{score}")

    if loss_fnc is not None:
        logging.info(f'Validation: Avg. Loss: {val_losses.avg}.')
    scores = '\n\t'.join([f"{fnc+' Score:':<30}{val_scores[fnc].avg}" for fnc in METRICS])
    logging.info(f'Evaluation Scores: \n\t{scores}')

    return {fnc: v.avg for fnc, v in val_scores.items()}


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', dest='model', type=str, required=True,
                        help='Load model from a .pth file')
    parser.add_argument('-i', '--inputs', dest="input", metavar='INPUT',
                        help='directory of input images', required=True)
    parser.add_argument('-l', '--labels', dest="label", metavar='LABEL',
                        help='directory of ground truth images', required=True)
    parser.add_argument('-t', '--tests', dest='tests', nargs='+', default=False, type=int,
                        help='Files to use as test cases')
    parser.add_argument('-n', '--name', dest='name', type=str, default='U-Net',
                        help='Prefix name to be used in output files')
    parser.add_argument('-o', '--output', dest='output', type=str, default='Testing.log',
                        help='File to write the log')
    parser.add_argument('-q', "--quiet", default=False, action="store_true", help="Run without stdout")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    output_filename = args.output + '_' + args.name + datetime.now().strftime("_%b-%d-%Y_%I-%M-%S_%p")
    handlers = [logging.FileHandler(output_filename)]
    if not args.quiet:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
                        handlers=handlers)

    net = UNet3D(in_channels=1, out_channels=2, testing=True, final_sigmoid=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    logging.info(f'Using device {device}')

    logging.info("Testing model {}".format(args.model))
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info(f"On dataset in {args.input} and {args.label}")
    data_set = BrainDataset(args.input, 'T1', args.label)

    if args.tests:
        data_loader = BrainLoaders(data_set, files=[None, args.tests])
    else:
        data_loader = BrainLoaders(data_set, ratios=[0, 1])

    validate(net, data_loader, is_validation=False, quiet=False)
