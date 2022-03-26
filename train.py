import os
import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

import projector
import net
from loader import LLFFDataset
from net import (RenderNet)
from loss import VggLoss
from sampler import SubsetSampler

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(args, model, device, train_loader, optimizer,
          epoch, loss_fn, writer, outpath):
    model.train()

    np.random.seed()

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = [x.to(device, non_blocking=True) for x in data]
        target = target.to(device, non_blocking=True)
        rendering, acc_alpha, disp = model(inputs)

        tgt_disp = inputs[-1][:, -1]

        loss = loss_fn(rendering, target)
        loss.backward()

        optimizer.step()

        # Logging
        if idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, idx+1, len(train_loader),
                    100. * (idx+1) / len(train_loader),
                    loss.detach().item()))

        # Tensorboard
        if epoch % args.image_interval == 0 and idx % args.image_iter == 0 and writer != None:

            output_imgs = rendering.detach()
            output_imgs = torch.clamp(output_imgs, 0.0, 1.0)
            gridded_image = vutils.make_grid(output_imgs)
            writer.add_image('rgb_output', gridded_image, idx + 1 + epoch * len(train_loader))

            gridded_image = vutils.make_grid(target)
            writer.add_image('rgb_reference', gridded_image, idx + 1 + epoch * len(train_loader))

            gridded_image = vutils.make_grid(acc_alpha)
            writer.add_image('accumulated_alphas', gridded_image, idx + 1 + epoch * len(train_loader))

            gridded_image = vutils.make_grid(disp, normalize=True)
            writer.add_image('output_disparity', gridded_image, idx + 1 + epoch * len(train_loader))

            gridded_image = vutils.make_grid(tgt_disp, normalize=True)
            writer.add_image('target_disparity', gridded_image, idx + 1 + epoch * len(train_loader))

        if writer != None and idx % args.log_interval == 0:
            writer.add_scalar('Loss', loss.detach().item(), idx + 1 + epoch * len(train_loader))

        # Saving model file
        if idx % args.save_interval == 0 and not args.no_save and epoch % 5 == 0:
            print("Saving model...")
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, outpath + f'model_{epoch+1}_{idx}.pth')

def test(args, model, device, test_loader, loss_fn):

    model.eval()
    test_loss = 0

    with torch.no_grad():

        for data, target in test_loader:
            inputs = [x.to(device) for x in data]
            target = target.to(device)

            output = model(inputs)
            loss = loss_fn(output, target)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    print('\n Test set: Average loss: {:.4f}\n'.format(
        test_loss
    ))

def main():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, metavar='PATH',
                        help='path to the training data',
                        default='/media/ken/Data/unreal_llff/important_data/synthetic_training/unreal/train2/')
    parser.add_argument('--test_path', type=str, metavar='PATH',
                        help='path to the validation data',
                        default='/media/ken/Data/sidewinder_resized_validation/')
    parser.add_argument('--log_path', type=str,
                        metavar='PATH',
                        help='path to log file',
                        default='/home/ken/sidewinder_data/dmpi_net_log/')
    parser.add_argument('--model_path', type=str,
                        metavar='PATH',
                        help='path to model file')
    parser.add_argument('--code_folder', type=str,
                        metavar='PATH',
                        help='path to code folder',
                        default='/home/ken/sidewinder_data/dmpi_net_experiment/')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='N',
                        help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='# of epochs for training')
    parser.add_argument('--image_iter', type=int, default=500, metavar='N',
                        help='# of iterations for saving images')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--valid_interval', type=int, default=1, metavar='N',
                        help='how many iterations between validation')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many iterations between logs')
    parser.add_argument('--save_interval', type=int, default=1000, metavar='N',
                        help='how many iterations between saves')
    parser.add_argument('--image_interval', type=int,
                        default=1,
                        metavar='N',
                        help='how many epochs between saves for images')
    parser.add_argument('--disable_cuda', action='store_true', default=False,
                        help='enable/disable cuda')
    parser.add_argument('--no_save', action='store_true', default=False,
                        help='do not save experiment')
    args = parser.parse_args()
    

    print(args)

    outpath = args.code_folder + 'exp_' + \
            time.strftime("%m_%d_%H%M", time.localtime()) + '/'

    if not args.no_save:
        if not os.path.exists(outpath):
            try:
                os.makedirs(outpath, exist_ok=True)
            except OSError:
                print("Directory creation failed at %s" % outpath)
            else:
                print("Directory created for %s" % outpath)

        os.system('cp *.py %s' % outpath )

    # Set random seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        np.random.seed(int(time.time()))

    # Set device
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
    device = args.device

    # Load dataset
    train_dataset = LLFFDataset(args.path, ret_disp=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=8,
            drop_last=False, pin_memory=True, worker_init_fn=worker_init_fn)

    '''
    # Load test dataset
    test_dataset = MPIDataset(args.test_path, neigh=4, cam_selection='closest')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=8,
            drop_last=False, pin_memory=True)
    '''

    # Initialize logger
    if not args.no_save:
        writer = SummaryWriter(args.log_path + 'exp' +
                time.strftime("%m_%d_%H%M", time.localtime()))
    else:
        writer = None

    # Setting global variable for the functions
    projector.device = device
    net.device = device

    # Create model
    model = nn.DataParallel(RenderNet().to(device))

    # Create loss function
    vgg = VggLoss().to(device)
    loss_fn = vgg

    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load previous model
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("Continue training with previous model file.")
            
            # Load model
            state = model.module.state_dict()           
            # state = model.state_dict()     
            # state.update(torch.load(args.model_path))
            state.update(torch.load(args.model_path)['model_state_dict'])
            model.module.load_state_dict(state)
            # model.load_state_dict(state)
            
            # Load optimizer
            # state = optimizer.state_dict()
            # state.update(torch.load(args.model_path)['optimizer_state_dict'])
            # optimizer.load_state_dict(state)
        else:
            print("No file found...training a new model instead.")
    else:
        print("Start training a new model.")

    # Train the model
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer,
            epoch, loss_fn, writer, outpath)
        # if epoch % args.valid_interval == 0:
        #     test(args, model, device, test_loader, loss_fn)

if __name__ == '__main__':
    main()
