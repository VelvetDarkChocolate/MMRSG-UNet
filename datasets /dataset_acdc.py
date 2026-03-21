import os
import random
import numpy as np
import torch
import h5py
from scipy.ndimage import zoom, rotate, map_coordinates, gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms


class StrongRandomGenerator(object):
    def __init__(self, output_size=[224, 224]):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = self.elastic_transform(image, label, alpha=15, sigma=3)

        if random.random() > 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            label = np.rot90(label, k)

        if random.random() > 0.5:
            axis = random.choice([0, 1])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            image = rotate(image, angle, order=3, reshape=False, mode='nearest')
            label = rotate(label, angle, order=0, reshape=False, mode='nearest')

        if random.random() > 0.5:
            gamma = random.uniform(0.7, 1.5)
            min_val = image.min()
            rng = image.max() - min_val
            image = ((image - min_val) / (rng + 1e-7)) ** gamma * rng + min_val

        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(-0.1, 0.1)
            image = image * contrast + brightness

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            zoom_x = self.output_size[0] / x
            zoom_y = self.output_size[1] / y
            image = zoom(image, (zoom_x, zoom_y), order=3)
            label = zoom(label, (zoom_x, zoom_y), order=0)

        if random.random() > 0.7:
            h, w = image.shape
            mask_size = random.randint(10, h // 4)
            y1 = np.random.randint(0, h - mask_size)
            x1 = np.random.randint(0, w - mask_size)
            image[y1:y1 + mask_size, x1:x1 + mask_size] = image.min()

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        return {'image': image, 'label': label.long()}

    def elastic_transform(self, image, label, alpha, sigma):
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        distorted_image = map_coordinates(image, indices, order=3, mode='reflect').reshape(shape)
        distorted_label = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)
        return distorted_image, distorted_label


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.slice_dir = os.path.join(self.data_dir, "slices")
        txt_path = os.path.join(list_dir, self.split + '.txt')
        list_path = os.path.join(list_dir, self.split + '.list')
        if os.path.exists(txt_path):
            self.sample_list = open(txt_path).readlines()
        elif os.path.exists(list_path):
            self.sample_list = open(list_path).readlines()
        else:
            raise FileNotFoundError(f"Neither {txt_path} nor {list_path} exists for split={split}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip()
        if "slice" in name:
            h5_path = os.path.join(self.slice_dir, name + ".h5")
        else:
            h5_path = os.path.join(self.data_dir, name + ".h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)
        with h5py.File(h5_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = name
        return sample


class ACDCVolume(Dataset):
    def __init__(self, base_dir, list_file, transform=None):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.transform = transform
        with open(list_file, "r") as f:
            self.case_list = [ln.strip() for ln in f.readlines() if ln.strip()]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_name = self.case_list[idx]
        vol_path = os.path.join(self.data_dir, case_name + ".h5")
        if not os.path.exists(vol_path):
            raise FileNotFoundError(vol_path)
        with h5py.File(vol_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = case_name
        return sample
