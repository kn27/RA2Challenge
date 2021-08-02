from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
#from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from models import detection
from models.detection.faster_rcnn import FastRCNNPredictor

def fasterrcnn_resnet50_fpn2(num_classes, saved_model=None,pretrained=True,device='cpu'):
    if saved_model is not None:
        pretrained = False
    model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
    return model


def fasterrcnn_resnet50_fpn(num_classes, saved_model=None,pretrained=True,device='cpu'):
    if saved_model is not None:
        pretrained = False
    model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
    return model
 
def basic_classification(num_classes, saved_model=None,device='cpu'):
    model = Base_Classification_Net(num_classes)
    model.to(device)
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
    return model

class Base_Classification_Net(nn.Module):
    def __init__(self, num_classes):
        super(Base_Classification_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 3, padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 256, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 1024, 3, stride = 3),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 5 * 5, 256),
            nn.ReLU(True),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            #nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
