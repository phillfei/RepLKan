import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
from medpy import metric
import h5py
from sklearn.metrics import confusion_matrix

def compute_iou(pred, target):

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        iou = np.nan  # 忽略没有样本的类别
    else:
        iou = intersection / union

    return iou  # 计算所有类别的平均 IoU (mIoU


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        output = net(input)
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1,classes):
        metric_list.append((compute_iou(prediction == i, label == i),calculate_metric_percase(prediction == i, label == i)[0],calculate_metric_percase(prediction == i, label == i)[1]))
    return metric_list
def Synapse_test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    metric_list = []
    # print(output.shape)
    for i in range(1, classes):
        metric_list.append((compute_iou(prediction == i, label == i),calculate_metric_percase(prediction == i, label == i)[0],calculate_metric_percase(prediction == i, label == i)[1]))
    print(metric_list)
    return metric_list

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
def load_image(filename,scale,flag=True):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    if ext == '.npz':
        # print(np.load(filename))
        img = np.load(filename)['img']
        mask = np.load(filename)['label']
        if flag == True:
            return img,mask
        else:
            return mask
    elif ext == '.h5':
        data = h5py.File(filename)
        image, label = data['image'][:], data['label'][:]
        return image, label
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        image = Image.open(filename)
        return image
def unique_mask_values1(idx, mask_dir, mask_suffix,scale):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]

    mask = np.asarray(load_image(mask_file,scale,flag=0))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def unique_mask_values(idx, mask_dir, mask_suffix,scale = 224):
    # print(mask_dir.glob(idx + mask_suffix + '.*'))
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # mask_file = list(mask_dir.glob(idx.replace('volume','segmentation') + mask_suffix + '.*'))[0]
    # mask_file = list(mask_dir.glob(idx+ '_segmentation' + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file,scale))
    if mask.ndim == 2:
        # print(np.unique(mask))
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
import random
import torchvision.transforms as transforms


import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from scipy.ndimage import zoom
import h5py
import os
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str = None, scale: int = 224, mask_suffix: str = '', augment: bool = True, split: str = 'train',classes: int = 8):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.split = split
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        self.mask_values = [i for i in range(classes)]
        logging.info(f'Unique mask values: {self.mask_values}')
        if self.augment:
            self.transform = A.Compose([
                
                A.Resize(224, 224),
                A.GaussianBlur(blur_limit=7, always_apply=False, p=0.5),
                A.Rotate(limit=90, p=0.5),

            
            ]# Only image normalization
            )  # Add 'mask' to specify no normalization for masks
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
            ])

    def __len__(self):
        return len(self.ids)
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = scale,scale
        if newW != w:
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name+ '_segmentation' + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0],self.scale)
        img = load_image(img_file[0],self.scale)
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # Convert images and masks to numpy arrays
        img = np.array(img)
        mask = np.array(mask)
        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = self.preprocess(self.mask_values, Image.fromarray(img), self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, Image.fromarray(mask), self.scale, is_mask=True)

            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()
        else:
            
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            img = self.preprocess(self.mask_values, Image.fromarray(img), self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, Image.fromarray(mask), self.scale, is_mask=True)

            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()
        return img,mask



class ACDCDataset(Dataset):
    def __init__(self, images_dir: str = None, mask_dir: str = None, scale: int = 224, mask_suffix: str = '', augment: bool = True,split: str = 'train',classes : int = 9):
        if images_dir is not None:
            self.images_dir = Path(images_dir)
        else :
            self.images_dir = None
        if mask_dir is not None:
            self.mask_dir = Path(mask_dir)
        else:
            self.mask_dir = None
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, file)) and not file.startswith('.')]
        self.split = split
        self.mask_values = [i for i in range(classes)]
        if self.augment:
            self.transform = A.Compose([
                A.Rotate(limit=90, p=0.5),
                A.GaussianBlur(blur_limit=7, always_apply=False, p=0.5),
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = scale, scale
        if newW != w:
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))

        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img,mask = load_image(mask_file[0],self.scale)
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.augment:
            if self.split == 'train':
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            img = self.preprocess(self.mask_values, Image.fromarray(img), self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, Image.fromarray(mask), self.scale, is_mask=True)
            # print(np.unique(mask))
            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()
        else:
            return img,mask

        return img,mask


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
