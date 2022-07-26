import torch.nn as nn
import torchvision
from utils.func import *


class VGGGAP(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGGGAP, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential((nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 4), nn.Sigmoid()))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGG16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        temp_classifier = torchvision.models.vgg16(pretrained=pretrained).classifier
        removed = list(temp_classifier.children())
        removed = removed[:-1]
        temp_layer = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 4), nn.Sigmoid())
        removed.append(temp_layer)
        self.classifier = nn.Sequential(*removed)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def choose_locmodel(model_name, pretrained=False, ckpt_path='resnet50loc.pth.tar'):
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load(ckpt_path))
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load(ckpt_path))
    elif model_name == 'vgggap':
        model = VGGGAP(pretrained=True, num_classes=1000)
        if pretrained:
            model = copy_parameters(model, torch.load(ckpt_path))
    elif model_name == 'vgg16':
        model = VGG16(pretrained=True, num_classes=1000)
        if pretrained:
            model = copy_parameters(model, torch.load(ckpt_path))
    elif model_name == 'inceptionv3':
        model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        # model.AuxLogits = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 4),
        #     nn.Sigmoid()
        # )
        if pretrained:
            model = copy_parameters(model, torch.load(ckpt_path))

    else:
        raise ValueError('Do not have this model currently!')
    return model


def choose_clsmodel(model_name, num_classes=1000):
    if model_name == 'vgg16':
        cls_model = torchvision.models.vgg16(pretrained=True, num_classes=num_classes)
    elif model_name == 'inceptionv3':
        cls_model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
    elif model_name == 'resnet50':
        cls_model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet161':
        cls_model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'dpn131':
        cls_model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True, test_time_pool=True)
    elif model_name == 'efficientnetb7':
        from efficientnet_pytorch import EfficientNet
        cls_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    return cls_model
