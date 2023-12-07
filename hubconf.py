# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torchvision.models.resnet import resnet50

import vision_transformer as vits

dependencies = ["torch", "torchvision"]


def dino_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_small"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb16(pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    Achieves 76.1% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    Achieves 77.4% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_base"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with DINO.
    Achieves 75.3% top-1 accuracy on ImageNet linear evaluation benchmark (requires to train `fc`).
    """
    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_xcit_small_12_p16(pretrained=True, **kwargs):
    """
    XCiT-Small-12/16 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', "xcit_small_12_p16", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_small_12_p8(pretrained=True, **kwargs):
    """
    XCiT-Small-12/8 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', "xcit_small_12_p8", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p16(pretrained=True, **kwargs):
    """
    XCiT-Medium-24/16 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', "xcit_medium_24_p16", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p8(pretrained=True, **kwargs):
    """
    XCiT-Medium-24/8 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', "xcit_medium_24_p8", num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model

# This code defines several functions for loading pre-trained vision models that have been trained with the DINO 
# (Data-IN-Opponent) self-supervised learning method. These functions provide convenient ways to load these models along 
# with their pre-trained weights. Let's break down each function and its purpose:

# 1. `dino_vits16`:
#    - This function loads a ViT-Small/16x16 model pre-trained with DINO.
#    - The model architecture is created using the `vits` module, specifying a patch size of 16x16.
#    - If `pretrained` is set to `True`, it loads the pre-trained weights for this model from a URL.
#    - The model is returned.

# 2. `dino_vits8`:
#    - Similar to `dino_vits16`, this function loads a ViT-Small/8x8 model pre-trained with DINO.
#    - The model architecture is created with a patch size of 8x8.
#    - Pre-trained weights are loaded if `pretrained` is `True`.
#    - The model is returned.

# 3. `dino_vitb16`:
#    - This function loads a ViT-Base/16x16 model pre-trained with DINO.
#    - The model architecture is created with a patch size of 16x16.
#    - Pre-trained weights are loaded if `pretrained` is `True`.
#    - The model is returned.

# 4. `dino_vitb8`:
#    - Similar to `dino_vitb16`, this function loads a ViT-Base/8x8 model pre-trained with DINO.
#    - The model architecture is created with a patch size of 8x8.
#    - Pre-trained weights are loaded if `pretrained` is `True`.
#    - The model is returned.

# 5. `dino_resnet50`:
#    - This function loads a ResNet-50 model pre-trained with DINO.
#    - The model is based on torchvision's ResNet-50 architecture.
#    - The fully connected (fc) layer of the model is replaced with an identity layer.
#    - Pre-trained weights are loaded if `pretrained` is `True`.
#    - The model is returned.

# 6. `dino_xcit_small_12_p16`, `dino_xcit_small_12_p8`, `dino_xcit_medium_24_p16`, `dino_xcit_medium_24_p8`:
#    - These functions load models from the XCiT family pre-trained with DINO.
#    - The model architectures are created using the Facebook Research XCiT repository.
#    - Different configurations, such as model size and patch size, are available.
#    - Pre-trained weights are loaded if `pretrained` is `True`.
#    - The respective model is returned based on the chosen function.

# In summary, these functions provide a convenient way to load various vision models that have been pre-trained with the DINO 
# self-supervised learning method. The models can be used for a variety of computer vision tasks, and their pre-trained weights 
# can be easily obtained from the provided URLs when needed.