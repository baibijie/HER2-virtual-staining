from __future__ import print_function
from configobj import ConfigObj
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader

import os
from os import listdir
from os.path import join
import torch.utils.data as data
from PIL import Image
import Attention_GAN


def init_parameters(
    is_training=True,
    batch_size=12,
    image_size=256,
    n_channels=64,
    n_threads=1,
    n_blocks=5,
    input_ch_num=4,
    output_ch_num=3,
    checkpoint=3,
    mse_weight=10,
    ssim_weight=0.05,
    dis_weight=0.2,
    lr_G=1e-4,
    lr_D=3e-5,
):
    tc = ConfigObj()
    tc.is_training = is_training
    tc.batch_size = batch_size
    tc.image_size = image_size
    tc.n_channels = n_channels
    tc.n_threads = n_threads
    tc.n_blocks = n_blocks
    tc.input_ch_num = input_ch_num
    tc.output_ch_num = output_ch_num
    tc.checkpoint = checkpoint
    tc.lamda = mse_weight  # mse
    tc.nu = ssim_weight  # ssim
    tc.alpha = dis_weight  # dis
    tc.lr_G = lr_G
    tc.lr_D = lr_D
    return tc


def is_target_file(filename):
    return filename.endswith(".npy")


def load_img(filepath):
    y = np.load(filepath).astype(np.float32)
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(
        self, image_dir, image_size, input_transform=None, target_transform=None
    ):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_target_file(x)
        ]
        self.image_size = image_size
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fpath = self.image_filenames[index]
        input = load_img(fpath)
        input = self.normalize_channels(input)
        input = input[[0, 1, 2, 3], :, :]
        target = load_img(fpath.replace("input", "input"))

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target, os.path.basename(fpath)

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


def get_data_set(data_dir, crop_size):
    return DatasetFromFolder(
        data_dir, crop_size, input_transform=None, target_transform=None
    )


def norm_img(img):
    img.clone()
    img.clamp_(min=-1, max=1)
    img.add_(1).div_(2)
    ndarr = img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    return im


if __name__ == "__main__":
    cont_config = init_parameters()

    input_path = "test_images/input"
    model_path = "test_model_path/"
    save_path = "test_images/output"
    test_epoch = 600

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Attention_GAN.Generator(
        cont_config,
        in_channels=cont_config.input_ch_num,
        out_channels=cont_config.output_ch_num,
        padding=1,
        batch_norm=False,
        pooling_mode="maxpool",
    )

    if torch.cuda.device_count() > 1:
        netG = torch.nn.DataParallel(netG)
    netG_dict = torch.load(model_path + "netG_epoch_{}.pth".format(test_epoch))
    netG.load_state_dict(netG_dict)
    netG.to(device)

    data_set = get_data_set(input_path, 256)
    data_loader = DataLoader(
        dataset=data_set,
        num_workers=6,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    netG.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            fnames = batch[2]
            input = batch[0].to(device)
            output = netG(input)

            for i, fname in enumerate(fnames):
                img = norm_img(output[i])
                img.save(save_path + "/" + fname[:-4] + ".tif")
