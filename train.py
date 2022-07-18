from __future__ import print_function
from configobj import ConfigObj
import os
from time import time
from datetime import datetime
from batch_utils import get_data_set
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import pytorch_ssim
import Attention_GAN


def init_parameters(is_training=True):
    tc = ConfigObj()

    # network and training
    tc.batch_size = 28
    tc.image_size = 256
    tc.n_channels = 64
    tc.n_blocks = 5
    tc.input_ch_num = 4
    tc.output_ch_num = 3

    tc.checkpoint = 3
    tc.lamda = 10  # L1 weight
    tc.nu = 0.2  # ssim weight
    tc.alpha = 0.5  # dis weight
    tc.lr_G = 1e-4
    tc.lr_D = 1e-5

    # data loading
    tc.n_threads = 10
    tc.crop_size = 256
    tc.max_epochs = 1000

    tc.epoch_len = 500
    tc.max_epochs = 1000
    tc.data_queue_len = 10000
    tc.patch_per_tile = 10
    tc.color_space = "RGB"

    tc.train_images_dir = r"train_data/train/target"
    tc.valid_images_dir = r"train_data/valid/target"

    return tc


if __name__ == "__main__":

    train_config = init_parameters()
    valid_config = init_parameters(is_training=False)

    cont_flag = False
    retr_epoch_id = 0

    total_iters = 1000
    Generator_iters = 4  # every n epoches G is trained, D will be trained once

    im_save_path = "Validation_images"
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = (
        "model/"
        + datetime_str
        + im_save_path
        + "-lr_G={:.0e}-lr_D={:.0e}".format(train_config.lr_G, train_config.lr_D)
    )
    summary_path = (
        "log/"
        + datetime_str
        + im_save_path
        + "-lr_G={:.0e}-lr_D={:.0e}".format(train_config.lr_G, train_config.lr_D)
    )

    if not os.path.exists(im_save_path):
        os.makedirs(im_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    print("===> Building model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # build generator and discriminator
    netG = Attention_GAN.Generator(
        train_config,
        in_channels=train_config.input_ch_num,
        out_channels=train_config.output_ch_num,
        padding=1,
        batch_norm=False,
        pooling_mode="maxpool",
    )

    netD = Attention_GAN.Discriminator(
        train_config, in_channels=train_config.output_ch_num, batch_norm=False
    )

    if torch.cuda.device_count() > 1:
        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)

    netG.to(device)
    netD.to(device)

    if cont_flag:
        netG_dict = torch.load(
            model_save_path + "/netG_epoch_{}.pth".format(retr_epoch_id)
        )
        netG.load_state_dict(netG_dict)

        netD_dict = torch.load(
            model_save_path + "/netD_epoch_{}.pth".format(retr_epoch_id)
        )
        netD.load_state_dict(netD_dict)

    # define training and validation sets
    train_set = get_data_set(
        train_config.train_images_dir, train_config.image_size, is_training=True
    )
    valid_set = get_data_set(
        valid_config.valid_images_dir, valid_config.image_size, is_training=False
    )

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=train_config.n_threads,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    valid_data_loader = DataLoader(
        dataset=valid_set,
        num_workers=valid_config.n_threads,
        batch_size=valid_config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # print(len(train_set), len(valid_set))

    writer = SummaryWriter(summary_path)

    # loss functions
    bce_loss_fun = nn.BCEWithLogitsLoss(size_average=True).to(device)
    l1_loss_fun = nn.SmoothL1Loss(size_average=True).to(device)
    ssim_loss_fun = pytorch_ssim.SSIM(size_average=True).to(device)

    real_label = 1
    fake_label = 0
    label_real = torch.full(
        (train_config.batch_size, 1),
        real_label,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )
    label_fake = torch.full(
        (train_config.batch_size, 1),
        fake_label,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )

    input = (
        torch.Tensor(
            train_config.batch_size,
            train_config.input_ch_num,
            train_config.image_size,
            train_config.image_size,
        )
        .requires_grad_(requires_grad=False)
        .to(device)
    )
    target = (
        torch.Tensor(
            train_config.batch_size,
            train_config.output_ch_num,
            train_config.image_size,
            train_config.image_size,
        )
        .requires_grad_(requires_grad=False)
        .to(device)
    )
    val_input = (
        torch.Tensor(
            train_config.batch_size,
            train_config.input_ch_num,
            train_config.image_size,
            train_config.image_size,
        )
        .requires_grad_(requires_grad=False)
        .to(device)
    )
    val_target = (
        torch.Tensor(
            train_config.batch_size,
            train_config.output_ch_num,
            train_config.image_size,
            train_config.image_size,
        )
        .requires_grad_(requires_grad=False)
        .to(device)
    )

    # setup optimizer
    optimizerG = optim.AdamW(
        netG.parameters(), lr=train_config.lr_G, betas=(0.9, 0.999)
    )
    optimizerD = optim.AdamW(
        netD.parameters(), lr=train_config.lr_D, betas=(0.9, 0.999)
    )

    print("===> Training Start")
    valid_log = open("valid_log.txt", "w")

    if cont_flag:
        startEpoch = retr_epoch_id
    else:
        startEpoch = 0

    niter = total_iters
    for epoch in range(startEpoch, niter, 1):
        start_time = time()
        # train
        run_loss_G = 0
        run_loss_G_l1 = 0
        run_loss_G_ssim = 0
        run_loss_D_real = 0
        run_loss_D_fake = 0

        counter = 0
        netG.train()
        netD.train()
        for i, batch in enumerate((train_data_loader), 1):

            input.copy_(batch[0])
            target.copy_(batch[1])

            ############################
            # (1) Update G network: maximize log(D(G(z)))
            ###########################
            fake = netG(input)
            netG.zero_grad()
            G_dis_loss = bce_loss_fun(netD(fake), label_real)
            G_l1_loss = l1_loss_fun(fake, target)
            G_ssim_loss = -torch.log(
                (1 + ssim_loss_fun((fake + 1.0) / 2.0, (target + 1.0) / 2.0)) / 2
            )
            errG = (
                train_config.alpha * G_dis_loss
                + train_config.lamda * G_l1_loss
                + train_config.nu * G_ssim_loss
            )

            clip_grad_norm_(netG.parameters(), 0.5)
            errG.backward()
            optimizerG.step()

            ############################
            # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            indie_G = Generator_iters
            if i % indie_G == 0:
                netD.zero_grad()
                D_fake_loss = bce_loss_fun(netD(fake.detach()), label_fake)
                D_real_loss = bce_loss_fun(netD(target), label_real)
                errD = (D_fake_loss + D_real_loss) * 0.5
                clip_grad_norm_(netD.parameters(), 0.5)
                errD.backward()
                optimizerD.step()

            if i % indie_G == 0:
                counter = counter + 1
                run_loss_G = run_loss_G + errG.item()
                run_loss_G_l1 = run_loss_G_l1 + train_config.lamda * G_l1_loss.item()
                run_loss_G_ssim = run_loss_G_ssim + train_config.nu * G_ssim_loss.item()

                run_loss_D_real = run_loss_D_real + D_real_loss.item()
                run_loss_D_fake = run_loss_D_fake + D_fake_loss.item()

        run_loss_G = run_loss_G / counter
        run_loss_G_l1 = run_loss_G_l1 / counter
        run_loss_G_ssim = run_loss_G_ssim / counter

        run_loss_D_real = run_loss_D_real / counter
        run_loss_D_fake = run_loss_D_fake / counter
        run_loss_D = (run_loss_D_real + run_loss_D_fake) * 0.5

        # valid
        val_loss_G = 0
        val_loss_G_l1 = 0
        val_loss_G_ssim = 0
        val_loss_D_real = 0
        val_loss_D_fake = 0
        counter = 0
        netG.eval()
        netD.eval()
        with torch.no_grad():
            for i, batch in enumerate((valid_data_loader), 1):

                val_input.copy_(batch[0])
                val_target.copy_(batch[1])
                val_fake = netG(val_input)
                netG.zero_grad()
                G_dis_loss = bce_loss_fun(netD(val_fake), label_real)
                G_l1_loss = l1_loss_fun(val_fake, val_target)
                G_ssim_loss = ssim_loss_fun(val_fake / 255.0, val_target / 255.0)
                errG = (
                    train_config.alpha * G_dis_loss
                    + train_config.lamda * G_l1_loss
                    + train_config.nu * G_ssim_loss
                )

                D_fake_loss = bce_loss_fun(netD(fake.detach()), label_fake)
                D_real_loss = bce_loss_fun(netD(target), label_real)
                errD = (D_fake_loss + D_real_loss) * 0.5

                counter = counter + 1
                val_loss_G = val_loss_G + errG.item()
                val_loss_G_l1 = val_loss_G_l1 + valid_config.lamda * G_l1_loss.item()
                val_loss_G_ssim = val_loss_G_ssim + valid_config.nu * G_ssim_loss.item()

                val_loss_D_real = val_loss_D_real + D_real_loss.item()
                val_loss_D_fake = val_loss_D_fake + D_fake_loss.item()

        val_loss_G = val_loss_G / counter
        val_loss_G_l1 = val_loss_G_l1 / counter
        val_loss_G_ssim = val_loss_G_ssim / counter
        val_loss_D_real = val_loss_D_real / counter
        val_loss_D_fake = val_loss_D_fake / counter
        valid_loss_D = (val_loss_D_real + val_loss_D_fake) * 0.5

        text = (
            "[%d/%d] Loss_G: %.4f Loss_G_l1: %.4f Loss_G_ssim: %.4f | Loss_D: %.4f D(x): %.4f D(G(z)): %.4f Time: %d s"
            % (
                epoch,
                niter,
                run_loss_G,
                run_loss_G_l1,
                run_loss_G_ssim,
                run_loss_D,
                run_loss_D_real,
                run_loss_D_fake,
                int(time() - start_time),
            )
        )
        print(text)

        writer.add_scalars("loss_G", {"train": run_loss_G, "valid": val_loss_G}, epoch)
        writer.add_scalars(
            "loss_G_l1", {"train": run_loss_G_l1, "valid": val_loss_G_l1}, epoch
        )
        writer.add_scalars(
            "loss_G_l1", {"train": run_loss_G_ssim, "valid": val_loss_G_ssim}, epoch
        )
        writer.add_scalars(
            "loss_D_real", {"train": run_loss_D_real, "valid": val_loss_D_real}, epoch
        )
        writer.add_scalars(
            "loss_D_fake", {"train": run_loss_D_fake, "valid": val_loss_D_fake}, epoch
        )
        writer.add_scalars(
            "loss_D", {"train": run_loss_D, "valid": valid_loss_D}, epoch
        )

        # do checkpointing
        if epoch % train_config.checkpoint == 0:
            vutils.save_image(
                val_input[:, [0], :, :],
                "%s/inputAF_epoch_%03d.png" % (im_save_path, epoch),
                normalize=True,
                range=(0, 2.55),
            )
            vutils.save_image(
                val_target,
                "%s/target_epoch_%03d.png" % (im_save_path, epoch),
                normalize=True,
                range=(-1, 1),
            )
            vutils.save_image(
                val_fake.detach(),
                "%s/network_epoch_%03d.png" % (im_save_path, epoch),
                normalize=True,
                range=(-1, 1),
            )

            torch.save(
                netG.state_dict(), "%s/netG_epoch_%d.pth" % (model_save_path, epoch)
            )
            torch.save(
                netD.state_dict(), "%s/netD_epoch_%d.pth" % (model_save_path, epoch)
            )
