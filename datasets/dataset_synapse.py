import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk


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
            image, label = data['image'][:], data['label'][:].astype(np.int16)

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

        if ID[:-7]+'\n' in train_ls:
            pass
            # for i in range(1, label_arr.shape[0]+1):
            #     if i < 10:
            #         np.savez(os.path.join(train_npy_path, ID[:-7] +'-0{}'.format(i)+'.npz'), image=imgs_arr[i-1], label=label_arr[i-1])  #按病例保存npy
            #         train_slice_ls.append(ID[:-7] +'-0{}'.format(i))
            #     else:
            #         np.savez(os.path.join(train_npy_path, ID[:-7] + '-{}'.format(i) + '.npz'), image=imgs_arr[i-1], label=label_arr[i-1])
            #         train_slice_ls.append(ID[:-7] + '-{}'.format(i))
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

    # with open('data/Synapse/train.txt', mode='w+') as file:
    #     file.writelines([line+'\n' for line in train_slice_ls])

    with open('data/Synapse/test_vol.txt', mode='w+') as file:
        file.writelines([line + '\n' for line in test_slice_ls])


if __name__ == '__main__':
    # for i in range(imgs_arr.shape[0]):
    #     m,s = cv2.meanStdDev(imgs_arr[i])
    #     m_list.append(m)
    #     s_list.append(s)
    # m = np.mean(m_list)
    # s = np.mean(s_list)
    # print(m)
    # print(s)

    # m = np.mean(m_list)
    # s = np.mean(s_list)
    save_npy()