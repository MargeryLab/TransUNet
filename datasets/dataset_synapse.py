import os
import random

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk


def compute_mean_std():
    data_train_root = '/home/dell/WinDisk2/DLDemo/MSBC-Net-Simple-Class/datasets/rectalTumor/rectal_tumor_train'
    m_list, s_list = [], []

    for img in os.listdir(data_train_root):
        img_arr = cv2.imread(os.path.join(data_train_root, img))
        m, s = cv2.meanStdDev(img_arr)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_arr = np.array(m_list)
    s_arr = np.array(s_list)
    m = m_arr.mean(axis=0, keepdims=True)
    s = s_arr.mean(axis=0, keepdims=True)
    return m[0][::-1], s[0][::-1]

pixel_mean, pixel_std = [180.5142679 , 150.84969904, 139.45428343],[23.89445224, 28.3921791 , 32.05544164]


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


def save_npy():
    root_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/NEW12Month/RectalCancerDataFull'
    raw_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/NEW12Month/RectalCancerDataFull/dcmData'
    labels_path = '/media/margery/4ABB9B07DF30B9DB/DARA-SIRRUNRUN/NEW12Month/RectalCancerDataFull/labelData'
    train_npy_path = './data/Synapse/train_npz'
    test_npy_path = './data/Synapse/test_npz'
    train_ls = open(os.path.join(root_path, 'train.txt')).readlines()

    train_slice_ls = []
    test_slice_ls = []

    for ID in os.listdir(labels_path):
        data_path = os.path.join(raw_path, ID[:-7], ID[:-7], 'HRT2')
        reader = sitk.ImageSeriesReader()
        dcm_path = reader.GetGDCMSeriesFileNames(data_path)
        reader.SetFileNames(dcm_path)
        imgs = reader.Execute()
        imgs_arr = sitk.GetArrayFromImage(imgs)
        imgs_arr = (imgs_arr-pixel_mean)/pixel_std

        label_path = os.path.join(labels_path, ID)
        label = sitk.ReadImage(label_path)
        label_arr = sitk.GetArrayFromImage(label)
        label_arr[label_arr == 4] = 1  # 黄色交界处定义为肿瘤
        label_arr[label_arr > 2] = 0 #包含012

        # for i in range(imgs_arr.shape[0]):
        #     m, s = cv2.meanStdDev(imgs_arr[i])
        #     m_list.append(m)
        #     s_list.append(s)
        # m = np.mean(m_list)
        # s = np.mean(s_list)
        # print(m)
        # print(s)
        #
        # m = np.mean(m_list)
        # s = np.mean(s_list)

        if ID[:-7]+'\n' in train_ls:
            for i in range(1, label_arr.shape[0]+1):
                if i < 10:
                    np.savez(os.path.join(train_npy_path, ID[:-7] +'-0{}'.format(i)+'.npz'), image=imgs_arr[i-1], label=label_arr[i-1])  #按病例保存npy
                    train_slice_ls.append(ID[:-7] +'-0{}'.format(i))
                else:
                    np.savez(os.path.join(train_npy_path, ID[:-7] + '-{}'.format(i) + '.npz'), image=imgs_arr[i-1], label=label_arr[i-1])
                    train_slice_ls.append(ID[:-7] + '-{}'.format(i))
        else:
            test_slice_ls.append(ID[:-7])
            h5f = h5py.File(os.path.join(test_npy_path, ID[:-7]+'.npy.h5'), 'w')
            h5f.create_dataset('image', data=imgs_arr)
            h5f.create_dataset('label', data=label_arr)
            h5f.close()
            # for i in range(1, label_arr.shape[0] + 1):
            #     if i < 10:
            #         np.savez(os.path.join(test_npy_path, ID[:-7] + '-0{}'.format(i) + '.npz'), image=imgs_arr[i - 1],
            #                  label=label_arr[i - 1])  # 按病例保存npy
            #         test_slice_ls.append(ID[:-7] + '-0{}'.format(i))
            #     else:
            #         np.savez(os.path.join(test_npy_path, ID[:-7] + '-{}'.format(i) + '.npz'), image=imgs_arr[i - 1],
            #                  label=label_arr[i - 1])
            #         test_slice_ls.append(ID[:-7] + '-{}'.format(i))

    with open('data/Synapse/train.txt', mode='w+') as file:
        file.writelines([line+'\n' for line in train_slice_ls])

    with open('data/Synapse/test_vol.txt', mode='w+') as file:
        file.writelines([line + '\n' for line in test_slice_ls])


def png_save_npy():
    data_train_root = '/home/dell/WinDisk2/DLDemo/MSBC-Net-Simple-Class/datasets/rectalTumor/rectal_tumor_train'
    data_test_root = '/home/dell/WinDisk2/DLDemo/MSBC-Net-Simple-Class/datasets/rectalTumor/rectal_tumor_val'
    train_label_path = '/home/dell/WinDisk/Datasets/ISIC-2017_Training_Part1_GroundTruth'
    test_label_path = '/home/dell/WinDisk/Datasets/ISIC-2017_Test_v2_Part1_GroundTruth'
    des_train_path = 'data/Synapse/train_npz'
    des_test_path = 'data/Synapse/test_npz'

    trainID_list, testID_list = [], []

    for img in os.listdir(data_test_root):
        testID_list.append(img[:-4])
        img_arr = cv2.imread(os.path.join(data_test_root, img))
        img_arr = (img_arr - pixel_mean) / pixel_std
        label_arr = cv2.imread(os.path.join(test_label_path, img[:-4]+'_segmentation.png'))/255

        h5f = h5py.File(os.path.join(des_test_path, img[:-4] + '.npy.h5'), 'w')
        h5f.create_dataset('image', data=img_arr)
        h5f.create_dataset('label', data=label_arr)
        h5f.close()

    for img in os.listdir(data_train_root):
        trainID_list.append(img[:-4])
        img_arr = cv2.imread(os.path.join(data_train_root, img))
        img_arr = (img_arr - pixel_mean) / pixel_std
        label_arr = cv2.imread(os.path.join(train_label_path, img[:-4]+'_segmentation.png'))/255

        np.savez(os.path.join(des_train_path, img[:-4]+'.npz'), image=img_arr, label=label_arr)

    with open('data/Synapse/train.txt', mode='w+') as file:
        file.writelines([line+'\n' for line in trainID_list])

    with open('data/Synapse/test_vol.txt', mode='w+') as file:
        file.writelines([line + '\n' for line in testID_list])


if __name__ == '__main__':
    # print(compute_mean_std())
    # save_npy()#for dcm
    png_save_npy() #for png data