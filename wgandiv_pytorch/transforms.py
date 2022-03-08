
""" 
Link: https://github.com/honglianghe/CDNet/blob/f436555539e140ff8bafa3c9f54cbc2550b7cebd/my_transforms.py
Author: Hongliang He 
"""

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
class Compose(object):
    """ Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        # self.selectorNameList = selectorNameList
    def __call__(self, imgs):
        # number = 0
        for t in self.transforms:
            #selectorName = str(self.selectorNameList[number])
            #start_time = time.time()
            imgs = t(imgs)

            # number = number + 1
        return imgs



class Scale(object):
    """Rescale the input PIL images to the given size. """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        pics = []
        for img in imgs:
            if isinstance(self.size, int):
                w, h = img.size
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    pics.append(img)
                    continue
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    pics.append(img.resize((ow, oh), self.interpolation))
                    continue
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    pics.append(img.resize((ow, oh), self.interpolation))
            else:
                pics.append(img.resize(self.size, self.interpolation))
        return tuple(pics)


import cv2
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


        do_albu = 0 # TODO                                                
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
                    # img = np.array(img)
                    img = cv2.copyMakeBorder(np.array(img),padding_t,padding_b,padding_l,padding_r,cv2.BORDER_REFLECT)
                    print(img.shape)
                    Image.fromarray(img)
                    print(img.size)
                    # img = ImageOps.expand(img, border=padding , fill=0)
                pics.append(img)


        # print(pics[0].size)
        return tuple(pics)


class RandomColor(object):

    def __init__(self, randomMin = 1, randomMax = 2):

        self.randomMin = randomMin
        self.randomMax = randomMax


    def __call__(self, imgs):

        out_imgs = list(imgs)
        img = imgs[0]
        random_factor = 1 + (np.random.rand()-0.5)

        color_image = ImageEnhance.Color(img).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        img_output = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        out_imgs[0] = img_output

        return tuple(out_imgs)



class RandomAffine(object):
    """ Transform the input PIL Image using a random affine transformation
        The parameters of an affine transformation [a, b, c=0
                                                    d, e, f=0]
        are generated randomly according to the bound, and there is no translation
        (c=f=0)
    Args:
        bound: the largest possible deviation of random parameters
    """

    def __init__(self, bound):
        if bound < 0 or bound > 0.5:
            raise ValueError("Bound is invalid, should be in range [0, 0.5)")

        self.bound = bound

    def __call__(self, imgs):
        img = imgs[0]
        x, y = img.size

        a = 1 + 2 * self.bound * (random.random() - 0.5)
        b = 2 * self.bound * (random.random() - 0.5)
        d = 2 * self.bound * (random.random() - 0.5)
        e = 1 + 2 * self.bound * (random.random() - 0.5)

        # correct the transformation center to image center
        c = -a * x / 2 - b * y / 2 + x / 2
        f = -d * x / 2 - e * y / 2 + y / 2

        trans_matrix = [a, b, c, d, e, f]

        pics = []
        for img in imgs:
            pics.append(img.transform((x, y), Image.AFFINE, trans_matrix))

        return tuple(pics)



class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """

        pics = []
        if random.random() < 0.5:
            for img in imgs:#imgs
                pics.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return tuple(pics)
        else:
            return imgs


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        pics = []
        if random.random() < 0.5:
            for img in imgs:
                pics.append(img.transpose(Image.FLIP_TOP_BOTTOM))

            return tuple(pics)
        else:
            return imgs


class RandomElasticDeform(object):
    """ Elastic deformation of the input PIL Image using random displacement vectors
        drawm from a gaussian distribution
    Args:
        sigma: the largest possible deviation of random parameters
    """
    def __init__(self, num_pts=4, sigma=20):
        self.num_pts = num_pts
        self.sigma = sigma

    def __call__(self, imgs):
        pics = []

        do_albu = 1
        if (do_albu == 1):
            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
            if(len(label.shape)==2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            if(len(image.shape)==2):
                image = image.reshape(image.shape[0], image.shape[1], 1)
            
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            transf = albu.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50, alpha_affine=50,
                                           interpolation=0, border_mode=0,
                                           value=(0, 0, 0),
                                           mask_value=None, approximate=False)  # border_mode 用于指定插值算法

            concat_map_transf = transf(image=concat_map)['image']
            image_channel = image.shape[-1]
            image_transf = concat_map_transf[:, :, :image_channel]
            image_transf = np.squeeze(image_transf)
            weightmap_transf = concat_map_transf[:, :, image_channel]
            if (label.shape[2] == 1):
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            
            
            
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            img = np.array(imgs[0])
            if len(img.shape) == 3:
                img = img[:,:,0]

            sitkImage = sitk.GetImageFromArray(img, isVector=False)
            mesh_size = [self.num_pts]*sitkImage.GetDimension()
            tx = sitk.BSplineTransformInitializer(sitkImage, mesh_size)

            params = tx.GetParameters()
            paramsNp = np.asarray(params, dtype=float)
            paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * self.sigma

            paramsNp[0:int(len(params)/3)] = 0  # remove z deformations! The resolution in z is too bad

            params = tuple(paramsNp)
            tx.SetParameters(params)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitkImage)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(tx)
            resampler.SetDefaultPixelValue(0)

            for img in imgs:
                is_expand = False
                if not isinstance(img, np.ndarray):
                    img = np.array(img)

                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                    is_expand = True

                img_deformed = np.zeros(img.shape, dtype=img.dtype)

                for i in range(img.shape[2]):
                    sitkImage = sitk.GetImageFromArray(img[:,:,i], isVector=False)
                    outimgsitk = resampler.Execute(sitkImage)
                    img_deformed[:,:,i] = sitk.GetArrayFromImage(outimgsitk)

                if is_expand:
                    img_deformed = img_deformed[:,:,0]
                # print img_deformed.dtype
                pics.append(Image.fromarray(img_deformed))


        return tuple(pics)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=Image.BILINEAR, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imgs):
        """
            imgs (PIL Image): Images to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        pics = []

        do_albu = 1
        if (do_albu == 1):
            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
            if (len(label.shape) == 2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            if(len(image.shape)==2):
                image = image.reshape(image.shape[0], image.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            transf = albu.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0,
                             value=(0, 0, 0), mask_value=None)  # border_mode 用于指定插值算法

            concat_map_transf = transf(image=concat_map)['image']
            image_channel = image.shape[-1]
            image_transf = concat_map_transf[:, :, :image_channel]
            image_transf = np.squeeze(image_transf)
            weightmap_transf = concat_map_transf[:, :, image_channel]
            if (label.shape[2] == 1):
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            for img in imgs:
                pics.append(img.rotate(angle, self.resample, self.expand, self.center))

        return tuple(pics)




class RandomChooseAug(object):

    def __call__(self, imgs):

        pics = []

        p_value = random.random()

        if p_value < 0.25:
            pics.append(imgs[0].filter(ImageFilter.BLUR))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.5:
            pics.append(imgs[0].filter(ImageFilter.GaussianBlur))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.75:
            pics.append(imgs[0].filter(ImageFilter.MedianFilter))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        else:
            return imgs



class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0, fill_val=(0,)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill_val = fill_val

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        pics = []

        w, h = imgs[0].size
        th, tw = self.size
        if(th > h or tw > w):
            ow = tw
            oh = th

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
                
                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if(image_h!=weightmap_h or image_h != label_h or image_w!=weightmap_w or image_w != label_w or weightmap_h!=label_h or weightmap_w!=label_w):

                    image_transf = np.resize(image,(th, tw, 3))
                    weightmap_transf = np.resize(weightmap,(th, tw))
                    label_transf = np.resize(label,(th, tw, 3))
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))
                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
                else:
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

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
        else:

            do_albu = 1
            if (do_albu == 1):
                min_max_height = (int(th * 0.6), th)
                transf = albu.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=min_max_height, height=th, width=tw,
                                              w2h_ratio=1.0, interpolation=0)

                image = np.array(imgs[0])
                weightmap = np.expand_dims(imgs[1], axis=2)
                label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
                if (len(label.shape) == 2):
                    label = label.reshape(label.shape[0], label.shape[1], 1)
                if(len(image.shape)==2):
                    image = image.reshape(image.shape[0], image.shape[1], 1)
                
                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if(image_h!=weightmap_h or image_h != label_h or image_w!=weightmap_w or image_w != label_w or weightmap_h!=label_h or weightmap_w!=label_w):

                    image_transf = np.resize(image,(th, tw, 3))
                    weightmap_transf = np.resize(weightmap,(th, tw))
                    label_transf = np.resize(label,(th, tw, 3))
                    
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
                    
                else:
                    concat_map = np.concatenate((image, weightmap, label), axis=2)

                    concat_map_transf = transf(image=concat_map)['image']
                    image_channel = image.shape[-1]
                    image_transf = concat_map_transf[:, :, :image_channel]
                    image_transf = np.squeeze(image_transf)
                    weightmap_transf = concat_map_transf[:, :, image_channel]
                    if (label.shape[2] == 1):
                        label_transf = concat_map_transf[:, :, -1:]
                        label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
                    else:
                        label_transf = concat_map_transf[:, :, -3:]
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)



            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                for k in range(len(imgs)):
                    img = imgs[k]
                    if self.padding > 0:
                        img = ImageOps.expand(img, border=self.padding, fill=self.fill_val[k])

                    if w == tw and h == th:
                        pics.append(img)
                        continue

                    pics.append(img.crop((x1, y1, x1 + tw, y1 + th)))


        return tuple(pics)
