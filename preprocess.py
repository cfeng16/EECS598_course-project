# The code of this part referenced https://github.com/yjn870/FSRCNN-pytorch
import pickle
import numpy as np
from PIL import Image
import os
from skimage import color
from tqdm import tqdm 
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--upscale_factor', type=int, default=2)
parser.add_argument('--patch_size', type=int, default=10)
parser.add_argument('--train_val_ratio', type=float, default=0.9)
parser.add_argument('--train_path', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\T91')
parser.add_argument('--val_path', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\T91')
parser.add_argument('--test_path', type=str, default=r'C:\Users\33602\Desktop\598_mini_project\image_SRF_2')
parser.add_argument('--test_set_name', type=str, default='set14')
args = parser.parse_args()

def train_preprocess(train_dir, scale, patch_size, image_list):
    downscale_list = [1.0, 0.9, 0.8, 0.7, 0.6]
    rotation_list = [0, 90, 180, 270]
    h5_file = h5py.File(r'C:\Users\33602\Desktop\598_mini_project\train_t91_upscale_' + str(args.upscale_factor) + '.h5', 'w')
    lr_patches = []
    hr_patches = []
    #image_list  = os.listdir(train_dir)
    for i in tqdm(range(len(image_list))):
        hr_images = []
        image = Image.open(os.path.join(train_dir, image_list[i])).convert('RGB')
        for d in downscale_list:
            for r in rotation_list:
                hr_image = image.resize((int(image.width * d), int(image.height * d)), resample=Image.BICUBIC)
                hr_image = hr_image.rotate(r, expand=True)
                hr_images.append(hr_image)
        for hr in hr_images:
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
            hr = np.array(hr)
            lr = np.array(lr)
            hr =  color.rgb2ycbcr(hr)
            lr = color.rgb2ycbcr(lr)
            hr = hr[:, :, 0:1].astype(np.float32)
            lr = lr[:, :, 0:1].astype(np.float32)
            for k in range(0, lr.shape[0] - patch_size + 1, scale):
                for q in range(0, lr.shape[1] - patch_size + 1, scale):
                    lr_patches.append(lr[k:k+patch_size, q:q+patch_size])
                    hr_patches.append(hr[k*scale:(k+patch_size)*scale, q*scale:(q+patch_size)*scale])
    #train_set['lr'] = lr_patches
    #train_set['hr'] = hr_patches
    #with open(r'C:\Users\33602\Desktop\598_mini_project\train_upscale_' + str(scale) + '.pkl', 'wb') as file:
    #    pickle.dump(train_set, file)
    #np.save(r'C:\Users\33602\Desktop\598_mini_project\train_lr_upscale_' + str(scale) + '.npy', lr_patches)
    #np.save(r'C:\Users\33602\Desktop\598_mini_project\train_hr_upscale_' + str(scale) + '.npy', hr_patches)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()

# def val_preprocess(train_dir, scale, patch_size, image_list):
#     downscale_list = [1.0, 0.9, 0.8, 0.7, 0.6]
#     rotation_list = [0, 90, 180, 270]
#     h5_file = h5py.File(r'C:\Users\33602\Desktop\598_mini_project\val_t91_upscale_' + str(args.upscale_factor) + '.h5', 'w')
#     lr_patches = []
#     hr_patches = []
#     #image_list  = os.listdir(train_dir)
#     for i in tqdm(range(len(image_list))):
#         hr_images = []
#         image = Image.open(os.path.join(train_dir, image_list[i]))
#         for d in downscale_list:
#             for r in rotation_list:
#                 hr_image = image.resize((int(image.width * d), int(image.height * d)), resample=Image.BICUBIC)
#                 hr_image = hr_image.rotate(r, expand=True)
#                 hr_images.append(hr_image)
#         for hr in hr_images:
#             hr_width = (hr.width // scale) * scale
#             hr_height = (hr.height // scale) * scale
#             hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
#             lr = hr.resize((hr.width // scale, hr_height // scale), resample=Image.BICUBIC)
#             hr =  color.rgb2ycbcr(hr)
#             lr = color.rgb2ycbcr(lr)
#             hr = np.array(hr)[:, :, 0:1].astype(np.float32)
#             lr = np.array(lr)[:, :, 0:1].astype(np.float32)
#             for k in range(0, lr.shape[0] - patch_size + 1, scale):
#                 for q in range(0, lr.shape[1] - patch_size + 1, scale):
#                     lr_patches.append(lr[k:k+patch_size, q:q+patch_size])
#                     hr_patches.append(hr[k*scale:(k+patch_size)*scale, q*scale:(q+patch_size)*scale])
#     #train_set['lr'] = lr_patches
#     #train_set['hr'] = hr_patches
#     #with open(r'C:\Users\33602\Desktop\598_mini_project\train_upscale_' + str(scale) + '.pkl', 'wb') as file:
#     #    pickle.dump(train_set, file)
#     #np.save(r'C:\Users\33602\Desktop\598_mini_project\train_lr_upscale_' + str(scale) + '.npy', lr_patches)
#     #np.save(r'C:\Users\33602\Desktop\598_mini_project\train_hr_upscale_' + str(scale) + '.npy', hr_patches)
#     lr_patches = np.array(lr_patches)
#     hr_patches = np.array(hr_patches)
#     h5_file.create_dataset('lr', data=lr_patches)
#     h5_file.create_dataset('hr', data=hr_patches)
#     h5_file.close()

# def val_preprocess(test_dir, scale):
#     image_list  = os.listdir(test_dir)
#     #hr_set = []
#     #lr_set = []
#     test_set = []
#     h5_file = h5py.File(r'C:\Users\33602\Desktop\598_mini_project\test_upscale_2.h5', 'w')
#     for i in tqdm(range(len(image_list))):
#         if image_list[i].split('.')[0][-2] == 'H':
#             hr = Image.open(os.path.join(test_dir, image_list[i]))
#             hr_width = (hr.width // scale) * scale
#             hr_height = (hr.height // scale) * scale
#             hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
#         else:
#             continue
#         lr_name = image_list[i].replace('H', 'L')
#         lr = Image.open(os.path.join(test_dir, lr_name))
#         hr =  color.rgb2ycbcr(hr)
#         lr = color.rgb2ycbcr(lr)
#         hr = hr.astype(np.float32)
#         lr = lr.astype(np.float32)
#         #lr_set.append(lr)
#         #hr_set.append(hr)
#         test_set.append((hr, lr))
#     #lr_set = np.stack(lr_set, axis=0)
#     #hr_set = np.stack(hr_set, axis=0)
#     #h5_file.create_dataset('lr', data=lr_set)
#     #h5_file.create_dataset('hr', data=hr_set)
#     #h5_file.close()
#     with open(r'C:\Users\33602\Desktop\598_mini_project\test_upscale_' + str(scale) + '.pkl', 'wb') as file:
#         pickle.dump(test_set, file)

def val_preprocess(val_dir, scale, image_list):
    #image_list  = os.listdir(val_dir)
    #hr_set = []
    #lr_set = []
    h5_file = h5py.File(r'C:\Users\33602\Desktop\598_mini_project\val_upscale_' + str(args.upscale_factor)+ '.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    for i in tqdm(range(len(image_list))):
        hr = Image.open(os.path.join(val_dir, image_list[i])).convert('RGB')
        hr_width = (hr.width // scale) * scale
        hr_height = (hr.height // scale) * scale
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
        hr = np.array(hr)
        lr = np.array(lr)
        hr = color.rgb2ycbcr(hr)
        lr = color.rgb2ycbcr(lr)
        hr = hr[:, :, 0:1].astype(np.float32)
        lr = lr[:, :, 0:1].astype(np.float32)
        # hr = hr.astype(np.float32)
        # lr = lr.astype(np.float32)
        #lr_set.append(lr)
        #hr_set.append(hr)
        #test_set.append((hr, lr))
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)
    
def test_preprocess(test_dir, scale):
    image_list  = os.listdir(test_dir)
    #hr_set = []
    #lr_set = []
    test_set = []
    h5_file = h5py.File(r'C:\Users\33602\Desktop\598_mini_project\test_upscale_' + str(args.upscale_factor)+ '_' + args.test_set_name + '.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    g = -1
    for i in tqdm(range(len(image_list))):
        if image_list[i].split('.')[0][-2] == 'H':
            hr = Image.open(os.path.join(test_dir, image_list[i])).convert('RGB')
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
            g = g+1
        else:
            continue
        lr_name = image_list[i].replace('H', 'L')
        lr = Image.open(os.path.join(test_dir, lr_name)).convert('RGB')
        hr =  color.rgb2ycbcr(hr)
        lr = color.rgb2ycbcr(lr)
        hr = np.array(hr)[:, :, 0:1].astype(np.float32)
        lr = np.array(lr)[:, :, 0:1].astype(np.float32)
        # hr = hr.astype(np.float32)
        # lr = lr.astype(np.float32)
        #lr_set.append(lr)
        #hr_set.append(hr)
        #test_set.append((hr, lr))
        lr_group.create_dataset(str(g), data=lr)
        hr_group.create_dataset(str(g), data=hr)
    #lr_set = np.stack(lr_set, axis=0)
    #hr_set = np.stack(hr_set, axis=0)
    #h5_file.create_dataset('lr', data=lr_set)
    #h5_file.create_dataset('hr', data=hr_set)
    #h5_file.close()
if __name__ == '__main__':
    train_dir = r'C:\Users\33602\Desktop\598_mini_project\T91'
    image_list  = os.listdir(train_dir)
    train_index = np.random.choice(len(image_list), int(np.ceil(len(image_list)*args.train_val_ratio)), replace=False).tolist()
    train_list = []
    val_list = []
    for i in range(len(image_list)):
        if i in train_index:
            train_list.append(image_list[i])
        else:
            val_list.append(image_list[i])
    #train_preprocess(args.train_path, args.upscale_factor, args.patch_size, train_list)
    #al_preprocess(args.val_path, args.upscale_factor, val_list)
    test_preprocess(args.test_path, args.upscale_factor)

