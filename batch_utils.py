import numpy as np
import random
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
import torch


def is_target_file(filename):
    return filename.endswith(".npy")


def load_img(filepath):
    y = np.load(filepath).astype(np.float32)
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(
        self,
        image_dir,
        image_size,
        is_training=True,
        input_transform=None,
        target_transform=None,
    ):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_target_file(x)
        ]
        self.image_size = image_size
        self.is_training = is_training
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        s = np.floor(self.image_size * 1.5).astype("int16")
        cropEdge = 30
        got_image = False

        while got_image == False:
            path = self.image_filenames[index]

            target = load_img(path).astype("float32")
            input_img = load_img(path.replace("target", "input")).astype("float32")
            input_img = self.normalize_channels(input_img)

            if input_img.ndim == 2:
                input_img = np.expand_dims(input_img, axis=0)

            if target.ndim == 2:
                target = np.expand_dims(target, axis=0)

            for _ in range(20):
                xx = random.randint(cropEdge, input_img.shape[1] - s - cropEdge)
                yy = random.randint(cropEdge, input_img.shape[2] - s - cropEdge)
                if np.mean(target[:, xx : xx + s, yy : yy + s]) < 245:
                    got_image = True
                    break

            index = random.randint(0, len(self.image_filenames) - 1)

        img = input_img[:, xx : xx + s, yy : yy + s]
        lab = target[:, xx : xx + s, yy : yy + s]

        if self.is_training:
            if random.randint(0, 1):
                img = img[:, :, ::-1]
                lab = lab[:, :, ::-1]
            if random.randint(0, 1):
                img = img[:, ::-1, :]
                lab = lab[:, ::-1, :]
            rot = random.randint(0, 3)
            img = np.rot90(img, k=rot, axes=(1, 2)).copy()
            lab = np.rot90(lab, k=rot, axes=(1, 2)).copy()

        img = torch.tensor(img)
        lab = torch.tensor(lab)
        if self.is_training:
            angle, translations, scale, shear = transforms.RandomAffine.get_params(
                degrees=(-20, 20),
                translate=None,
                scale_ranges=(1, 1),
                shears=(-5, 5, -5, 5),
                img_size=(s, s),
            )
            img = transforms.functional.affine(
                img, angle=angle, translate=translations, scale=scale, shear=shear
            )
            lab = transforms.functional.affine(
                lab, angle=angle, translate=translations, scale=scale, shear=shear
            )

        img = transforms.functional.center_crop(img, [self.image_size, self.image_size])
        lab = transforms.functional.center_crop(lab, [self.image_size, self.image_size])

        img = img[[0, 1, 2, 3], :, :]
        lab = lab / 128 - 1

        return img, lab

    def normalize_channels(self, im):
        norm_factor = [
            1 / 1500,
            1 / 1500,
            1 / 1500,
            1 / 1000,
        ]

        for ch_idx in range(len(norm_factor)):
            im[ch_idx, :, :] = im[ch_idx, :, :] * norm_factor[ch_idx]

        return im

    def __len__(self):
        return len(self.image_filenames)


def get_data_set(data_dir, crop_size, is_training=True):
    return DatasetFromFolder(
        data_dir,
        crop_size,
        is_training=is_training,
        input_transform=None,
        target_transform=None,
    )
