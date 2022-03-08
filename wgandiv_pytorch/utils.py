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
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

__all__ = [
    "create_folder", "calculate_gradient_penalty", "init_torch_seeds", "select_device", "weights_init"
]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(f"Create `{os.path.join(os.getcwd(), folder)}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def calculate_gradient_penalty(real_data, fake_data, real_outputs, fake_outputs, k=2, p=6, device=torch.device("cpu")):
    real_grad_outputs = torch.full((real_data.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad_outputs = torch.full((fake_data.size(0),), 1, dtype=torch.float32, requires_grad=False, device=device)
    # print("real_grad_outputs: ",real_grad_outputs.shape, real_data.shape, real_outputs.shape)
    real_gradient = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_data,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=fake_outputs,
        inputs=fake_data,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

    gradient_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    return gradient_penalty


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    logger.info("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str = "", batch_size: int = 1) -> torch.device:
    r""" Choose the right equipment.

    Args:
        device (optional, str): Use CPU or CUDA. (Default: ````)
        batch_size (optional, int): Data batch size, cannot be less than the number of devices. (Default: 1).

    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3".
    only_cpu = device.lower() == "cpu"
    if device and not only_cpu:  # if device requested other than "cpu".
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable.
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if only_cpu else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB.
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count.
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA "
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            logger.info(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, total_memory={int(x[i].total_memory / c)}MB)")
    else:
        logger.info("Using CPU.")

    return torch.device("cuda:0" if cuda else "cpu")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

from torch import Tensor
from PIL import Image, ImageOps
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from transforms import *

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))

def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False, crop=(256,256),
            norm_rate=(1,0),hard_rate=(0,0.5)) -> List[Image.Image]:
    assert norm_rate[0]+hard_rate[0]==1.0
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    # i, j, h, w = T.RandomCrop.get_params(imgs[0], (64,64))
    # for idx in range(len(imgs)):
    #     imgs[idx] = F.crop(imgs[idx], i, j, h, w)

    # rand_num = random()
    # img = np.array(imgs[0])
    # mask = np.array(imgs[1])
    # foreground = remove_by_label(img,mask,255)
    # background = remove_by_label(img,mask,0)
    # if rand_num<easy_rate[0]:

    #     imgs[0] = Image.fromarray(cv2.addWeighted(foreground,1,background,easy_rate[1],0))
    # if rand_num<hard_rate[0]:
    #     fore_rgb = get_rgb(foreground)
    #     background = changeColor(background,fore_rgb,np.ones(3)*0.5)
    #     imgs[0] = Image.fromarray((foreground+background).astype(np.uint8))
    
    # if flip and random() > 0.5:
    #     imgs = map_(ImageOps.flip, imgs)
    
    # trans = RandomChooseAug()
    # imgs = trans(imgs)
    trans = RandomColor()
    imgs = trans(imgs)
    trans = RandomHorizontalFlip()
    imgs = trans(imgs)
    trans = RandomAffine()
    imgs = trans(imgs)
    # trans = RandomRotation(90)
    # imgs = trans(imgs)
    # trans = RandomResize()
    # imgs = trans(imgs)

    # trans = RandomElasticDeform()
    # imgs = trans(imgs)
    # trans = RandomCrop(256)
    # imgs = trans(imgs)
    # trans =

    # trans =


    return imgs