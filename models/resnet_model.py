"""
Created on Jul 11, 2019

Network model for feature extraction

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import (Bottleneck, BasicBlock, model_urls)

__all__ = ['FeaturesModule', 'ResNetModule', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class Flatten(nn.Module):
    """Flatten input tensor to vector"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet(nn.Module):
    """RTesNet module"""

    def __init__(self, block, layers, channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class FeaturesModule(ResNet):
    """ResNet extension model"""

    def __init__(self, block: nn.Module, layers: list, channels: int = 3):
        super(FeaturesModule, self).__init__(block, layers, channels=channels)
        self.flatten = Flatten()

    def forward(self, input_tensor) -> torch.Tensor:
        x = super(FeaturesModule, self).forward(input_tensor)
        x = self.avgpool(x)
        features_tensor = self.flatten(x)

        return features_tensor


class ResNetModule(FeaturesModule):
    """ResNet extension model"""

    def __init__(self, block: nn.Module, layers: list, channels: int = 3, num_classes: int = 1000):
        super(ResNetModule, self).__init__(block, layers, channels=channels)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, input_tensor) -> torch.Tensor:
        x = super(ResNetModule, self).forward(input_tensor)
        output_tensor = self.fc(x)

        return output_tensor


def _init_layers(layers: list) -> list:
    """
    Initializes layer with default value if not defined
    Args:
        layers(list): layers for ResNet module

    Returns:
        default value if layers are not defined
    """
    return [2, 2, 2, 2] if layers is None else layers


def _init_model(core_type: nn.Module = ResNetModule, block: nn.Module = BasicBlock, layers: list = None,
                model_key: str = 'resnet18', pretrained: bool = False, strict: bool = False, **kwargs) -> nn.Module:
    """
    Initializes appropriated model
    Args:
        core_type(nn.Module): type for model core initialization
        block(nn.Module): block for layers initialization
        layers(list): model layers
        model_key(str): key for model URL dictionary
        pretrained(bool): flag for trained weights
        strict(bool): flag for weight loading
        **kwargs:  additional arguments

    Returns:
        model: network model with pre-trained weights
    """
    model = core_type(block, _init_layers(layers), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_key]), strict=strict)

    return model


def _init_module(core_type: nn.Module = ResNetModule, block: nn.Module = BasicBlock, layers: list = None,
                 model_key: str = 'resnet18', pretrained: bool = False, strict: bool = False, **kwargs) -> nn.Module:
    """
    Initializes appropriated model
    Args:
        block(nn.Module): block for layers initialization
        layers(list): model layers
        model_key(str): flag for trained weights
        pretrained(bool): flag for trained weights
        strict(bool): flag for weight loading
        **kwargs: additional arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_model(core_type=core_type, block=block, layers=layers,
                       model_key=model_key, pretrained=pretrained, strict=strict, **kwargs)


def resnet18(core_type=ResNetModule, pretrained: bool = False, strict: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-18 model
    Args:
        core_type(nn.Module): core netwok architecture
        pretrained(bool): returns a model pre-trained on ImageNet data if True else initialized weights
        strict(bool): flag for weight loading
        **kwargs: additional named arguments

    Returns:
        network model width pre-trained weights

    """
    return _init_module(core_type=core_type, block=BasicBlock, layers=[2, 2, 2, 2], model_key=resnet18.__name__,
                        pretrained=pretrained, strict=strict, **kwargs)


def resnet34(core_type: nn.Module = ResNetModule, pretrained: bool = False, strict: bool = False,
             **kwargs) -> nn.Module:
    """
    Constructs a ResNet-34 model
    Args:
        core_type(nn.Module): core netwok architecture
        pretrained(bool): If True, returns a model pre-trained on ImageNet data
        strict(bool):flag for weight loading
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(core_type=core_type, block=BasicBlock, layers=[3, 4, 6, 3], model_key=resnet34.__name__,
                        pretrained=pretrained, strict=strict, **kwargs)


def resnet50(core_type: nn.Module = ResNetModule, pretrained: bool = False, strict: bool = False,
             **kwargs) -> nn.Module:
    """
    Constructs a ResNet-50 model
    Args:
        core_type(nn.Module): core netwok architecture
        pretrained(bool): If True, returns a model pre-trained on ImageNet data
        strict(bool):flag for weight loading
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(core_type=core_type, block=Bottleneck, layers=[3, 4, 6, 3], model_key=resnet50.__name__,
                        pretrained=pretrained, strict=strict, **kwargs)


def resnet101(core_type: nn.Module = ResNetModule, pretrained: bool = False, strict: bool = False,
              **kwargs) -> nn.Module:
    """
    Constructs a ResNet-101 model
    Args:
        core_type(nn.Module): core netwok architecture
        pretrained(bool): If True, returns a model pre-trained on ImageNet data
        strict(bool):flag for weight loading
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(core_type=core_type, block=Bottleneck, layers=[3, 4, 23, 3], model_key=resnet101.__name__,
                        pretrained=pretrained, strict=strict, **kwargs)


def resnet152(core_type: nn.Module = ResNetModule, pretrained: bool = False, strict: bool = False,
              **kwargs) -> nn.Module:
    """
    Constructs a ResNet-152 model
    Args:
        core_type(nn.Module): core netwok architecture
        pretrained(bool): If True, returns a model pre-trained on ImageNet data
        strict(bool):flag for weight loading
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(core_type=core_type, block=Bottleneck, layers=[3, 8, 36, 3], model_key=resnet152.__name__,
                        pretrained=pretrained, strict=strict, **kwargs)
