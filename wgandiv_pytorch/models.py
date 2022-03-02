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
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "Discriminator", "Generator", "discriminator",
    "lsun"
]

model_urls = {
    "lsun": "https://github.com/Lornatang/WassersteinGAN_DIV-PyTorch/releases/download/0.1.0/Wasserstein_DIV_lsun-700f9016.pth"
}


class Discriminator(nn.Module):
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.

        Returns:
            A four-dimensional vector (NCHW).
        """
        out = self.main(input)
        out = torch.flatten(out)
        return out


class Generator(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """

    def __init__(self):
        super(Generator, self).__init__()
        dim = 64
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(100, 8*dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(8*dim, 4*dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(4*dim, 2*dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(2*dim, 1*dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1*dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(1*dim, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.

        Returns:
            A four-dimensional vector (NCHW).
        """
        out = self.main(input)
        return out


def _gan(arch, pretrained, progress):
    # model = Generator()
    model = UGen_Net(100,3,1e-4)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def discriminator() -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """
    model = Discriminator()
    return model


def lsun(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1704.00028v3>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("lsun", pretrained, progress)



import math
from typing import Iterable, Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import Tensor
# from networks.layers import *
# from networks.networks import weights_init


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


def downSampleConv(nin, nout, kernel_size=3, stride=2, padding=1, bias=False):
    return nn.Sequential(
        convBatch(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
    )


class interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, cin):
        return F.interpolate(cin, mode=self.mode, scale_factor=self.scale_factor)


def upSampleConv(nin, nout, kernel_size=3, upscale=2, padding=1, bias=False):
    return nn.Sequential(
        # nn.Upsample(scale_factor=upscale),
        interpolate(mode='nearest', scale_factor=upscale),
        convBatch(nin, nout, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
        convBatch(nout, nout, kernel_size=3, stride=1, padding=1, bias=bias),
    )



class UGen_Net(nn.Module):
    def __init__(self, nin, nout, l_rate, nG=64, has_dropout=False):
        super().__init__()
        self.encoder = SharedEncoder(nin, nout, has_dropout=has_dropout).cuda()
        self.rec_decoder = ReconstructionDecoderWoSkip(nin, nout).cuda()

        self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)

        self.encoder.apply(weights_init)
        self.rec_decoder.apply(weights_init)

    def forward(self, input):
        feature, x0, x1, x2 = self.encoder(input)
        rec_probs = self.rec_decoder(feature)

        return rec_probs

class residualConv(nn.Module):
    def __init__(self, nin, nout):
        super(residualConv, self).__init__()
        self.convs = nn.Sequential(
            convBatch(nin, nout),
            nn.Conv2d(nout, nout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nout)
        )
        self.res = nn.Sequential()
        if nin != nout:
            self.res = nn.Sequential(
                nn.Conv2d(nin, nout, kernel_size=1, bias=False),
                nn.BatchNorm2d(nout)
            )

    def forward(self, input):
        out = self.convs(input)
        return F.leaky_relu(out + self.res(input), 0.2)

class SharedEncoder(nn.Module):
    def __init__(self, nin, nout, nG=64, has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout
        self.conv0 = nn.Sequential(convBatch(nin, nG),
                                   convBatch(nG, nG))
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2),
                                   convBatch(nG * 2, nG * 2))
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2),
                                   convBatch(nG * 4, nG * 4))

        self.bridge = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2),
                                    residualConv(nG * 8, nG * 8),
                                    convBatch(nG * 8, nG * 8))
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, input):
        input = input.float()
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        result = self.bridge(x2)
        if self.has_dropout:
            result = self.dropout(result)
        return result, x0, x1, x2


class SegmentationDecoder(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input, feature_scale0, feature_scale1, feature_scale2):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(torch.cat((task1_y0, feature_scale2), dim=1)))
        task1_y2 = self.deconv3(self.conv6(torch.cat((task1_y1, feature_scale1), dim=1)))
        task1_y3 = self.conv7(torch.cat((task1_y2, feature_scale0), dim=1))
        task1_result = self.unetfinal(task1_y3)
        return task1_result


class ReconstructionDecoder(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input, feature_scale0, feature_scale1, feature_scale2):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(torch.cat((task1_y0, feature_scale2), dim=1)))
        task1_y2 = self.deconv3(self.conv6(torch.cat((task1_y1, feature_scale1), dim=1)))
        task1_y3 = self.conv7(torch.cat((task1_y2, feature_scale0), dim=1))
        task1_result = self.unetfinal(task1_y3)
        # return torch.sigmoid(task1_result)
        return torch.tanh(task1_result)


class ReconstructionDecoderWoSkip(nn.Module):
    def __init__(self, nin, nout, nG=64):
        super().__init__()

        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 8, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 4, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 2, nG * 1),
                                   convBatch(nG * 1, nG * 1))
        self.unetfinal = nn.Conv2d(nG, 3, kernel_size=1)

    def forward(self, input):
        task1_y0 = self.deconv1(input)
        task1_y1 = self.deconv2(self.conv5(task1_y0))
        task1_y2 = self.deconv3(self.conv6(task1_y1))
        task1_y3 = self.conv7(task1_y2)
        task1_result = self.unetfinal(task1_y3)
        print("task1: ",task1_result.shape)
        return torch.sigmoid(task1_result)

        # return torch.tanh(task1_result)
