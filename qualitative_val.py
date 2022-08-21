import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
from model_me import FSRCNN
from skimage.transform import resize
from PIL import Image
from skimage import color
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\_Mon_Apr_25_03_24_02_2022\final_model.pth')
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--test_directory', type=str, default=r'C:\Users\33602\Desktop\val_2')
args = parser.parse_args()

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = FSRCNN(56, 12, 4, args.scale_factor).to(device)
    net.load_state_dict(torch.load(args.model_path))
    image_list = os.listdir(args.test_directory)
    for i in range(len(image_list)):
        hr = Image.open(os.path.join(args.test_directory, image_list[i])).convert('RGB')
        hr_width = (hr.width // args.scale_factor) * args.scale_factor
        hr_height = (hr.height // args.scale_factor) * args.scale_factor
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // args.scale_factor, hr_height // args.scale_factor), resample=Image.BICUBIC)
        lr_file_name = 'lr_' + str(i) + '_' + '.jpg'
        lr.save(os.path.join(args.test_directory, lr_file_name))
        hr = np.array(hr)
        lr = np.array(lr)
        hr = color.rgb2ycbcr(hr)
        lr = color.rgb2ycbcr(lr)
        hr = hr[:, :, :].astype(np.float32) / 255.
        hr = np.transpose(hr, (2, 0, 1))
        lr = lr[:, :, :].astype(np.float32)  /255.
        lr = np.transpose(lr, (2, 0, 1))
        lr_direct = np.copy(lr)
        lr = lr[0:1, :, :]
        bilinear_result =resize(lr_direct, hr.shape, order=1)
        bicubic_result = resize(lr_direct, hr.shape, order=3)
        fsrcnn_result = net(torch.from_numpy(lr).to(device).unsqueeze(0))
        fsrcnn_result = torch.clamp(fsrcnn_result, 0, 1)
        fsrcnn_result = fsrcnn_result.cpu().detach().numpy().squeeze(0)
        fsrcnn_result = np.concatenate((fsrcnn_result, bicubic_result[1:, :, :]), axis=0)
        fsrcnn_result = color.ycbcr2rgb(fsrcnn_result*255, channel_axis=0)*255
        bilinear_result = color.ycbcr2rgb(bilinear_result*255, channel_axis=0)*255
        bicubic_result = color.ycbcr2rgb(bicubic_result*255, channel_axis=0)*255
        bicubic_result = np.transpose(bicubic_result, (1,2,0))
        bilinear_result = np.transpose(bilinear_result, (1,2,0))
        fsrcnn_result = np.transpose(fsrcnn_result, (1,2,0))
        fsrcnn_result = np.clip(fsrcnn_result, 0, 255)
        bicubic_result = np.clip(bicubic_result, 0, 255)
        bilinear_result = np.clip(bilinear_result, 0, 255)
        image_bicubic = Image.fromarray(bicubic_result.astype('uint8'), 'RGB')
        image_bilinear = Image.fromarray(bilinear_result.astype('uint8'), 'RGB')
        image_fsrcnn = Image.fromarray(fsrcnn_result.astype('uint8'), 'RGB')
        bicubic_file_name = 'bicubic_'+str(i)+'.jpg'
        bilinear_file_name = 'bilinear_'+str(i)+'.jpg'
        fsrcnn_file_name = 'fsrcnn_'+str(i)+'.jpg'
        image_bicubic.save(os.path.join(args.test_directory, bicubic_file_name ))
        image_bilinear.save(os.path.join(args.test_directory, bilinear_file_name))
        image_fsrcnn.save(os.path.join(args.test_directory, fsrcnn_file_name))
if __name__ == "__main__":
    main()