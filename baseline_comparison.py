import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from model_me import FSRCNN
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import argparse
from dataset import Testset
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--test_directory', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\test_upscale_4_set14.h5')
parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--model_path', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\_Mon_Apr_25_03_24_02_2022\final_model.pth')
args = parser.parse_args()

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


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = FSRCNN(56, 12, 4, args.scale_factor).to(device)
    net.load_state_dict(torch.load(args.model_path))
    test_set = Testset(args.test_directory, args.scale_factor)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    psnr_bilinear = AverageMeter()
    psnr_bicubic = AverageMeter()
    ssim_bilinear = AverageMeter()
    ssim_bicubic = AverageMeter()
    psnr_fsrcnn = AverageMeter()
    ssim_fsrcnn = AverageMeter()
    for test_image in tqdm(test_loader):
        fsrcnn_result = net(test_image['lr'].to(device))
        fsrcnn_result = torch.clamp(fsrcnn_result, min=0, max=1)
        hr = test_image['hr'].numpy().squeeze(0)
        lr = test_image['lr'].numpy().squeeze(0)
        bilinear_result = torch.from_numpy(resize(lr, hr.shape, order=1))
        bicubic_result = torch.from_numpy(resize(lr, hr.shape, order=3))
        bilinear_result = torch.clamp(bilinear_result, min=0, max=1)
        bicubic_result  = torch.clamp(bicubic_result , min=0, max=1)
        bilinear_result = bilinear_result.numpy()
        bicubic_result = bicubic_result.numpy()
        fsrcnn_result = fsrcnn_result.squeeze(0).cpu().detach().numpy()
        psnr_bilinear.update(peak_signal_noise_ratio(hr*255, bilinear_result*255, data_range=255), n=1)
        psnr_bicubic.update(peak_signal_noise_ratio(hr*255, bicubic_result*255, data_range=255), n=1)
        psnr_fsrcnn.update(peak_signal_noise_ratio(hr*255, fsrcnn_result*255, data_range=255), n=1)
        ssim_bilinear.update(structural_similarity(255*hr.squeeze(0), 255*bilinear_result.squeeze(0), data_range=255), n = 1)
        ssim_bicubic.update(structural_similarity(255*hr.squeeze(0), 255*bicubic_result.squeeze(0), data_range=255), n=1)
        ssim_fsrcnn.update(structural_similarity(255*hr.squeeze(0), 255*fsrcnn_result.squeeze(0), data_range=255), n=1)
    print('bilinear result psnr is {:.4f}'.format(psnr_bilinear.avg))
    print('bilinear result ssim is {:.4f}'.format(ssim_bilinear.avg))
    print('bicubic result psnr is {:.4f}'.format(psnr_bicubic.avg))
    print('bicubic result ssim is {:.4f}'.format(ssim_bicubic.avg))
    print('fsrcnn result psnr is {:.4f}'.format(psnr_fsrcnn.avg))
    print('fsrcnn result ssim is {:.4f}'.format(ssim_fsrcnn.avg))






if __name__ == "__main__":
    main()