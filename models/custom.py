import torch.nn as nn
import torchvision.models
from .base import BaseModel

MODEL_LIST = 'AlexNet, VGG, ResNet, SqueezeNet, \
              DenseNet, ShuffleNet v2, MobileNetV2, \
              MobileNetV3, ResNeXt, Wide ResNet, MNASNet'

class ImageClassifier(BaseModel):
    def __init__(self,conf):
        super(ImageClassifier,self).__init__()
        model_func = getattr(torchvision.models, conf['model']['name'])
        self.features = model_func(pretrained=conf['model']['pretrained'])
        self.num_classes = len(conf['data']['class_names'])
        ### change the classifier (reset the weights and change the linear layer)
        try:
            if hasattr(self.features, 'classifier'):
                if isinstance(self.features.classifier,nn.Sequential):
                    if 'squeeze' in conf['model']['name']:
                        self.features.classifier[1] = nn.Conv2d(in_channels=self.features.classifier[1].in_channels,
                                                        out_channels=self.num_classes,
                                                        kernel_size=self.features.classifier[1].kernel_size,
                                                        stride=self.features.classifier[1].stride)
                    else:
                        self.features.classifier[-1] = nn.Linear(self.features.classifier[-1].in_features,self.num_classes)
                else:
                    self.features.classifier = nn.Linear(self.features.classifier.in_features,self.num_classes)
            else:
                self.features.fc = nn.Linear(self.features.fc.in_features,self.num_classes)
        except ValueError:
            print('Use the following classifiers: ' + MODEL_LIST)
    def forward(self,xb):
        return self.features(xb)
