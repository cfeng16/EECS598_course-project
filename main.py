import os
from turtle import color
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import logging
import argparse
from model_me import FSRCNN
from dataset import Trainset, Testset, Valset
from tqdm import tqdm
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--train_directory', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\train_t91_upscale_3.h5')
parser.add_argument('--test_directory', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\val_upscale_3.h5')
parser.add_argument('--scale_factor', type=int, default=3)
parser.add_argument('--mapping_layer', type=int, default=4)
parser.add_argument('--lr_dimension', type=int, default=56)
parser.add_argument('--hr_dimension', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--deconv_lr', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--max_iter', type=int, default=600000)
parser.add_argument('--output_dir', type=str, default=r'C:\Users\33602\Desktop\598_mini_project')
parser.add_argument('--print_period', type=int, default=2000)
args = parser.parse_args()
def main():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    output_path = time.ctime().split(' ')
    output_path_ = ''
    for i in output_path:
        if ':' in i:
            i = i.replace(':', "_")
        output_path_ += '_'
        output_path_ += i
    output_dir = os.path.join(args.output_dir, output_path_)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger_path = os.path.join(output_dir, 'exp.log')
    logger = get_logger(logger_path)
    writer = SummaryWriter(log_dir=output_dir)
    cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_set = Trainset(args.train_directory, args.scale_factor)
    test_set = Valset(args.test_directory, args.scale_factor)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_loader = infinite_loader(train_loader)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    net = FSRCNN(d=args.lr_dimension, s=args.hr_dimension, m=args.mapping_layer, n=args.scale_factor)
    #net = FSRCNN(scale_factor=args.scale_factor).to(device)
    net.to(device)
    criterion = nn.MSELoss()
    params = [
        {'params':net.feat_ext.parameters()},
        {'params':net.shrink.parameters()},
        {'params':net.expand.parameters()},
        {'params':net.deconv.parameters(), 'lr':args.deconv_lr},
        {'params':net.act1.parameters(), "weight_decay": 0},
        {'params':net.act2.parameters(), "weight_decay": 0},
    #    {'params':net.act3.parameters(), "weight_decay": 0},
        {'params':net.act4.parameters(), "weight_decay": 0},
    ]
    for k in range(2*args.mapping_layer):
        if isinstance(net.map[k], nn.Conv2d):
            params.append({'params':net.map[k].parameters()})
        elif isinstance(net.map[k], nn.PReLU):
            params.append({'params':net.map[k].parameters(), "weight_decay": 0})
        else:
            raise Exception('The module should be either nn.Conv2d or nn.PReLU')
    optimizer = torch.optim.Adam(params, lr=args.lr)
    #for epoch in range(args.num_epochs):
        #with tqdm(total=(len(train_set) - len(train_set) % args.batch_size), ncols=100) as t:
    loss_history = []
    psnr_history = []
    with tqdm(total=args.max_iter, ncols=120) as t:
        #train_loss = AverageMeter()
        for iter in range(args.max_iter):
            net.train()
            test_psnr = AverageMeter()
            t.set_description('iteration: {}/{}'.format(iter, args.max_iter - 1))
            train_batch = next(train_loader)
            hr = train_batch['hr'].to(device)
            lr = train_batch['lr'].to(device)
            predict = net(lr)
            loss = criterion(predict, hr)
            #with torch.no_grad():
            #    train_loss.update(loss, n=args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #train_loss.update(loss.item(), len(lr))
            writer.add_scalar('Loss/train', loss.item(), iter)
            if iter % args.print_period == 0 or iter == args.max_iter - 1:
                loss_history.append(loss.item())
            t.set_postfix(loss='{:.6f}'.format(loss.item()))
            #logger.info('Step:[{}/{}]\t loss={:.6f}'.format(step, (args.num_epochs)*len(train_loader), loss.data))
            if iter % args.print_period == 0 or iter == args.max_iter - 1:
                net.eval()
                with torch.no_grad():
                    for test_image in test_loader:
                        hr = test_image['hr'].to(device)
                        lr = test_image['lr'].to(device)
                        predict = net(lr)
                        predict = torch.clamp(predict, min=0, max=1)
                        #test_psnr.update(peak_signal_noise_ratio(hr.cpu().numpy(), predict.cpu().numpy()), n=1)
                        test_psnr.update((10. * torch.log10(1 / torch.mean((predict - hr) ** 2))).item(), n=1)
                    psnr_history.append(test_psnr.avg)
                    logger.info('Iter:[{}/{}]\t PSNR={:.4f}'.format(iter, args.max_iter-1, test_psnr.avg))
                    writer.add_scalar('PSNR/test', test_psnr.avg, iter)
            t.update(1)
    torch.save(net.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    ax1.set_title("Training loss history")
    ax1.set_xlabel(f"Iteration (x {args.print_period})")
    ax1.set_ylabel("Loss")
    ax1.plot(loss_history, color='blue', marker='.')
    fig1.savefig(os.path.join(output_dir, 'train_loss.jpg'))
    ax2.set_title("Val PSNR history")
    ax2.set_xlabel(f"Iteration (x {args.print_period})")
    ax2.set_ylabel("PSNR")
    ax2.plot(psnr_history, color='red', marker='.')
    fig2.savefig(os.path.join(output_dir, 'val_psnr.jpg'))
        

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def infinite_loader(loader):
# The code of this function reference a4_helper.py from Assignment 4.
    while True:
        yield from loader


if __name__ == '__main__':
    main()