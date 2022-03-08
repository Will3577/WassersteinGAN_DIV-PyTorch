# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import math
import os

import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

import wgandiv_pytorch.models as models
from wgandiv_pytorch import calculate_gradient_penalty
from wgandiv_pytorch import discriminator
from wgandiv_pytorch import init_torch_seeds
from wgandiv_pytorch import select_device
from wgandiv_pytorch import weights_init
from wgandiv_pytorch.utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)



import io
import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional

import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
# from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, Sampler

# from utils.utils import id_, map_, class2one_hot
# from utils.utils import simplex, sset, one_hot, depth, augment
# from utils.my_transforms import PNG_Transform, GT_Transform, DUMMY_Transfrom

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


# resizing_fn = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
class PNG_Transform():
    def __init__(self):
        
        pass

    def __call__(self, img):
        # img = img.convert('L')
        img_array = np.array(img, dtype='float')#[np.newaxis, ...]
        img_array = np.transpose(img_array,(2,0,1))
        img_array /= 255
        return torch.tensor(img_array, dtype=torch.float32)

class SliceDataset(Dataset):
    def __init__(self, data,) -> None:
        self.filenames = os.listdir(data+'/train_256/')
        self.data_path = data+'/train_256/'
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> List[Any]:
        filename: str = self.filenames[index]
        # path_name: Path = Path(filename)
        # images: List[D]

        images = Image.open(self.data_path+self.filenames[index])
        # if path_name.suffix == ".png":
        #     images = [Image.open(files[index]) for files in self.files]
        # elif path_name.suffix == ".npy":
        #     images = [np.load(files[index]) for files in self.files]
        # else:
        #     raise ValueError(filename)

        # if self.augment:
            # augment = partial(augment,)
            # images = augment(*images)
        # print(np.array(images[0]).shape)
        # Final transforms and assertions
        # assert len(images) == len(self.folders) == len(self.transforms)
        # print(len(images),len(self.folders),len(self.transforms))
        # t_tensors: List[Tensor] = [tr(e) for (tr, e) in zip([PNG_Transform], [images])]
        t_tensors: List[Tensor] = [PNG_Transform(images)]

        # print("tensors: ",t_tensors[0].shape)

        # main image is between 0 and 1
        # if not self.ignore_norm:
        #     assert 0 <= t_tensors[0].min() and t_tensors[0].max() <= 1, (t_tensors[0].min(), t_tensors[0].max())
        # print(t_tensors[0].shape,t_tensors[1].shape)
        # _, w, h = t_tensors[0].shape
        # for ttensor in t_tensors[1:]:  # Things should be one-hot or at least have the shape
        #     # print(ttensor.shape,(self.C, w, h), (ttensor.shape, self.C, w, h))
        #     # print(len(t_tensors))
        #     assert ttensor.shape == (self.C, w, h), (ttensor.shape, self.C, w, h,t_tensors[1].shape,t_tensors[2].shape)

        # for ttensor, is_hot in zip(t_tensors, self.are_hots):  # All masks (ground truths) are class encoded
        #     if is_hot:
        #         assert one_hot(ttensor, axis=0), torch.einsum("cwh->wh", ttensor)

        # img, gt = t_tensors[:2]
        # print(img.shape, gt.shape, img.type(), gt.type())

        return t_tensors







import torch
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import numbers
import collections
from skimage import morphology
import SimpleITK as sitk
import time
import copy
from skimage import io
import albumentations as albu
import warnings
warnings.filterwarnings("ignore")
class RandomResize(object):
    """Randomly Resize the input PIL Image using a scale of lb~ub.
    Args:
        lb (float): lower bound of the scale
        ub (float): upper bound of the scale
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, lb=0.8, ub=1.3, interpolation=Image.BILINEAR):
        self.lb = lb
        self.ub = ub
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Images): Images to be scaled.
        Returns:
            PIL Images: Rescaled images.
        """

        for img in imgs:
            if not isinstance(img, Image.Image):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        scale = random.uniform(self.lb, self.ub)
        # print scale

        w, h = imgs[0].size
        ow = int(w * scale)
        oh = int(h * scale)


        do_albu = 1
        if(do_albu == 1):
            transf = albu.Resize(always_apply=False, p=1.0, height=oh, width=ow, interpolation=0)

            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2]) #np.expand_dims(imgs[2], axis=2)
            if (len(label.shape) == 2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            if(len(image.shape)==2):
                image = image.reshape(image.shape[0], image.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            concat_map_transf = transf(image=np.array(concat_map))['image']
            image_channel = image.shape[-1]
            image_transf = concat_map_transf[:, :, :image_channel]
            image_transf = np.squeeze(image_transf)
            weightmap_transf = concat_map_transf[:, :, image_channel]
            if (label.shape[2] == 1):
                #label = label.reshape(label.shape[0], label.shape[1], 1)
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics = []
            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            if scale < 1:
                padding_l = (w - ow)//2
                padding_t = (h - oh)//2
                padding_r = w - ow - padding_l
                padding_b = h - oh - padding_t
                padding = (padding_l, padding_t, padding_r, padding_b)

            pics = []
            for i in range(len(imgs)):
                img = imgs[i]
                img = img.resize((ow, oh), self.interpolation)
                if scale < 1:
                    img = ImageOps.expand(img, border=padding, fill=0)
                pics.append(img)



        return tuple(pics)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)
        # print(args.dataset,args.data)
        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        if args.dataset in ["imagenet", "folder", "lfw"]:
            # folder dataset
            dataset = SliceDataset(args.data)
            # dataset = torchvision.datasets.ImageFolder(root=args.data,
            #                                            transform=transforms.Compose([
            #                                                 RandomResize(),
            #                                         #     #    transforms.Resize((args.image_size, args.image_size)),
            #                                         #     #    transforms.CenterCrop(args.image_size),
            #                                                 transforms.RandomHorizontalFlip(p=0.5),
            #                                                 transforms.RandomVerticalFlip(p=0.5),

            #                                                 transforms.ToTensor(),
            #                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #                                            ])
            #                                            )
        elif args.dataset == "lsun":
            classes = [c + "_train" for c in args.classes.split(",")]
            dataset = torchvision.datasets.LSUN(root=args.data, classes=classes,
                                                transform=transforms.Compose([
                                                    transforms.Resize((args.image_size, args.image_size)),
                                                    transforms.CenterCrop(args.image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))
        else:
            # print(args.dataset)
            classes = [c + "_train" for c in args.classes.split(",")]
            dataset = torchvision.datasets.LSUN(root=args.data, classes=classes,
                                                transform=transforms.Compose([
                                                    transforms.Resize((args.image_size, args.image_size)),
                                                    transforms.CenterCrop(args.image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ]))
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      shuffle=True,
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.data}`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=1)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch]().to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator().to(self.device)

        # self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

        # Parameters of pre training model.
        self.start_epoch = math.floor(args.start_iter / len(self.dataloader))
        self.epochs = math.ceil(args.iters / len(self.dataloader))
        # self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        logger.info(f"Model training parameters:\n"
                    f"\tIters is {int(args.iters)}\n"
                    f"\tEpoch is {int(self.epochs)}\n"
                    f"\tOptimizer Adam\n"
                    f"\tBetas is (0.5, 0.999)\n"
                    f"\tLearning rate {args.lr}")

    def run(self):
        args = self.args

        # Load pre training model.
        if args.netD != "":
            self.discriminator.load_state_dict(torch.load(args.netD))
        if args.netG != "":
            self.generator.load_state_dict(torch.load(args.netG))

        self.discriminator.train()
        self.generator.train()

        # Start train PSNR model.
        logger.info(f"Training for {self.epochs} epochs")

        # fixed_noise = torch.randn(args.batch_size, 100, 1, 1, device=self.device)
        fixed_noise = torch.randn(args.batch_size, 512, 32, 32, device=self.device)

        for epoch in range(self.start_epoch, self.epochs):
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for i, data in progress_bar:
                print(len(data),data[0].shape,data[1].shape)
                real_images = torch.autograd.Variable(data[0].type(torch.Tensor), requires_grad=True)
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Sample noise as generator input
                # noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                noise = torch.randn(batch_size, 512, 32, 32, device=self.device)

                ##############################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ##############################################
                # Set discriminator gradients to zero.
                self.discriminator.zero_grad()

                # Train with real
                real_output = self.discriminator(real_images)
                errD_real = torch.mean(real_output)
                D_x = real_output.mean().item()

                # Generate fake image batch with G
                fake_images = self.generator(noise)

                # Train with fake
                # print(fake_images.shape)
                fake_output = self.discriminator(fake_images)
                errD_fake = -torch.mean(fake_output)
                D_G_z1 = fake_output.mean().item()

                # Calculate W-div gradient penalty
                gradient_penalty = calculate_gradient_penalty(real_images, fake_images,
                                                              real_output, fake_output, 2, 6,
                                                              self.device)

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake + gradient_penalty
                errD.backward()
                # Update D
                self.optimizer_d.step()

                # Train the generator every n_critic iterations
                if (i + 1) % args.n_critic == 0:
                    ##############################################
                    # (2) Update G network: maximize log(D(G(z)))
                    ##############################################
                    # Set generator gradients to zero
                    self.generator.zero_grad()
                    # Generate fake image batch with G
                    fake_images = self.generator(noise)
                    fake_output = self.discriminator(fake_images)
                    errG = torch.mean(fake_output)
                    D_G_z2 = fake_output.mean().item()
                    errG.backward()
                    self.optimizer_g.step()

                    progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.dataloader)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                iters = i + epoch * len(self.dataloader) + 1
                # The image is saved every 1000 epoch.
                if iters % 1000 == 0:
                    print("saving")
                    vutils.save_image(real_images,
                                      os.path.join("output", "real_samples.png"),
                                      normalize=True)
                    fake = self.generator(fixed_noise)
                    vutils.save_image(fake.detach(),
                                      os.path.join("output", f"fake_samples_{iters}.png"),
                                      normalize=True)

                    # do checkpointing
                    torch.save(self.generator.state_dict(), f"weights/{args.arch}_G_iter_{iters}.pth")
                    torch.save(self.discriminator.state_dict(), f"weights/{args.arch}_D_iter_{iters}.pth")
                    # print("end saving")
                if iters == int(args.iters):  # If the iteration is reached, exit.
                    break
