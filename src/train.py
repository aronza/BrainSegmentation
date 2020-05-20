#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import timeit
from datetime import datetime

import torch
from torch import optim
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt

from dataset.dataset import BrainDataset
from dataset.loader import BrainLoaders

from eval import validate
# from unet3d.losses import DiceLoss
from unet3d.model import UNet3D
from unet3d.metrics import METRICS
from unet3d.utils import RunningAverage, make_dir

dir_img = '/home/turkmea/training_data/T1/'
dir_mask = '/home/turkmea/training_data/WM/'
dir_checkpoint = '/home/turkmea/checkpoints/'
output_folder = '/home/turkmea/runs/'

# loss_fnc = DiceLoss(sigmoid_normalization=False)
loss_fnc = CrossEntropyLoss()


# loss_fnc = KLDivLoss(reduction='batchmean')

def plot_cost(costs, name, model_name):
    plt.clf()
    plt.title(name + " over Gradient Descent Iterations")
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.plot(range(len(costs)), costs, color='red')  # cost line
    plt.savefig(output_folder + model_name + str(datetime.now()) + '_' + name + '_overtime.png')


def train_net(model: UNet3D,
              epochs=5,
              learning_rate=0.0002,
              val_percent=0.1,
              test_percent=0.1,
              name='U-Net',
              tests=None,
              patch_size=16,
              testing_memory=False,
              mask_model=False):

    data_set = BrainDataset(dir_img, 'T1', dir_mask, stack_size=patch_size, mask_net=mask_model)
    loader = BrainLoaders(data_set, ratios=[val_percent, test_percent], files=[None, tests])

    train_loader = loader.train_loader()
    val_loader = loader.validation_loader()
    test_loader = loader.test_loader()

    num_images = data_set.num_files()
    log_interval = len(train_loader) if num_images < 10 else len(data_set.slices) * (num_images // 10)
    global_step = 0
    logging.info(f'''Starting {name} training:
        Epochs:          {epochs}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader)} slices
        Validation size: {len(val_loader)} images
        Testing size:    {len(test_loader)} images
        Log Interval     {log_interval}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    losses = []
    val_scores = {}
    for fnc in METRICS:
        val_scores[fnc] = []

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_start_time = timeit.default_timer()
        log_start_time = timeit.default_timer()
        log_loss = RunningAverage()
        for batch in train_loader:
            model.train()

            img = batch['image']
            mask = batch['mask']

            masks_pred = model(img)

            loss = loss_fnc(masks_pred, mask)

            epoch_loss += loss.item()
            log_loss.update(loss.item(), n=1)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if testing_memory:  # When testing patch sizes only one iteration is enough
                return

            global_step += 1
            if global_step % log_interval == 0:
                elapsed = timeit.default_timer() - log_start_time
                losses.append(log_loss.avg)
                logging.info(f'I: {global_step}, Avg. Loss: {log_loss.avg} in {elapsed} seconds')
                log_start_time = timeit.default_timer()
                log_loss = RunningAverage()

        scores = validate(model, loader, is_validation=True, loss_fnc=loss_fnc)
        for fnc in METRICS:
            val_scores[fnc].append(scores[fnc])

        make_dir(dir_checkpoint)
        torch.save(model.state_dict(),
                   dir_checkpoint + f'{name}_epoch{epoch + 1}.pth')
        elapsed = timeit.default_timer() - epoch_start_time
        logging.info(f'Epoch: {epoch + 1} Total Loss: {epoch_loss} in {elapsed} seconds')
        logging.info(f'Checkpoint {epoch + 1} saved !')
        plot_cost(losses, name='Loss', model_name=name + str(epoch) + '_')
        for fnc in METRICS:
            plot_cost(val_scores[fnc], name='Validation_' + type(fnc).__name__, model_name=name + str(epoch) + '_')

    logging.info('Starting Testing')
    validate(model, loader, is_validation=False, loss_fnc=loss_fnc, quiet=False)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-p', '--patch-size', metavar='P', type=int, default=16,
                        help='Patch Size', dest='patch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-P', '--premodel', dest='premodel', type=str, default=False,
                        help='Load model to precede current one from a .pth file')
    parser.add_argument('-n', '--name', dest='name', type=str, default='U-Net',
                        help='Prefix name to be used in output files')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--tests', dest='tests', nargs='+', default=None, type=int,
                        help='Files to use as test cases')
    parser.add_argument('-o', '--output', dest='output', type=str, default='Training.log',
                        help='File to write the log')
    parser.add_argument('-q', "--quiet", default=False, action="store_true", help="Run without stdout")
    parser.add_argument('-m', '--memory', dest='memory', type=int, default=False,
                        help='Test patch sizes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    make_dir(output_folder)
    output_filename = output_folder + datetime.now().strftime("%b-%d-%Y_%I-%M-%S_%p") + '_' + args.name + args.output
    handlers = [logging.FileHandler(output_filename)]
    if not args.quiet:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
                        handlers=handlers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    input_channels = 1
    output_channels = 2
    net = UNet3D(in_channels=input_channels, out_channels=output_channels, final_sigmoid=False)
    logging.info(f'Network:\n'
                 f'\t{input_channels} input channels\n'
                 f'\t{output_channels} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True

    try:
        if args.memory:
            stack_size = args.memory
            while True:
                logging.info(f'TESTING with size {stack_size}')
                logger = logging.getLogger()
                logger.disabled = True
                train_net(model=net,
                          epochs=1,
                          learning_rate=args.lr,
                          val_percent=0.1,
                          test_percent=0.8,
                          name=args.name,
                          tests=args.tests,
                          patch_size=stack_size,
                          testing_memory=True,
                          mask_model=args.premodel)
                logger.disabled = False
                logging.info(f'SUCCESS: Size {stack_size}')
                stack_size += 1
        train_net(model=net,
                  epochs=args.epochs,
                  learning_rate=args.lr,
                  val_percent=args.val / 100,
                  name=args.name,
                  tests=args.tests,
                  patch_size=args.patch_size,
                  mask_model=args.premodel)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
