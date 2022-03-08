import os
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
from wgandiv_pytorch.utils import augment

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
    def __init__(self, data) -> None:
        self.filenames = os.listdir(data+'/train_256/')
        self.data_path = data+'/train_256/'
        self.augment = True
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> List[Any]:
        # filename: str = self.filenames[index]
        # path_name: Path = Path(filename)
        # images: List[D]
        png_transform = PNG_Transform()
        images = Image.open(self.data_path+self.filenames[index])
        # if path_name.suffix == ".png":
        #     images = [Image.open(files[index]) for files in self.files]
        # elif path_name.suffix == ".npy":
        #     images = [np.load(files[index]) for files in self.files]
        # else:
        #     raise ValueError(filename)

        if self.augment:
            # augment = partial(augment,)
            images = augment(*[images])
        # print(np.array(images[0]).shape)
        # Final transforms and assertions
        # assert len(images) == len(self.folders) == len(self.transforms)
        # print(len(images),len(self.folders),len(self.transforms))
        # t_tensors: List[Tensor] = [tr(e) for (tr, e) in zip([PNG_Transform], [images])]
        t_tensors: List[Tensor] = [png_transform(images)]


        return t_tensors


