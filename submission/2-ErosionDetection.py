#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


# In[2]:


# from matplotlib import pyplot as plt 
# from matplotlib import figure, colors
# from matplotlib.patches import Rectangle
# from data import visualize, visualize_multiple
# import PIL.ExifTags
# from PIL import ImageOps
# import xmltodict
# from torchvision.transforms import functional as F
# from engine import train_one_epoch

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import random

from data import ID_TO_NAME_MAP_EROSION as ID_TO_NAME_MAP, NAME_TO_ID_MAP_EROSION as NAME_TO_ID_MAP, RADataSet
from model import fasterrcnn_resnet50_fpn
import utils.transforms, utils.datasets, utils.optimizer


# In[3]:


#handle = '/output'
handle = './output'


# In[4]:


import logging
logger = logging.getLogger('0')
hdlr = logging.FileHandler(handle + '/logs/2-erosiondetection.log')
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]   %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


# In[5]:


INPUT = handle + '/test'
JOINTOUTPUT = handle + '/test/erosion_all'
PRETRAIN_MODEL = './pretrained_models/saved_model_30_epochs_erosion_from_synthetic_to_full.txt'
logger.info (f'INPUT:{INPUT}, OUTPUT:{JOINTOUTPUT}, PRETRAIN_MODEL:{PRETRAIN_MODEL}')


# In[6]:


dataset = RADataSet(INPUT,transforms=utils.transforms.get_transform(train=True), score='Erosion')


# In[7]:


num_classes = len(ID_TO_NAME_MAP)
logger.info(f'Size of dataset: {len(dataset)}')
logger.info(f'Number of joints: {num_classes}')


# ### Model 

# In[9]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = 'cpu'
model = fasterrcnn_resnet50_fpn(num_classes=num_classes, saved_model = PRETRAIN_MODEL)
model.to(device);
logger.info(f'Device: {device}')


# In[10]:


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# In[11]:


#model.load_state_dict(torch.load(PRETRAIN_MODEL))


# ### Inference

# In[12]:


def filter_prediction(prediction, threshold = 0.2, filter_labels = None):
    if filter_labels is None:
        filter_labels = np.array(prediction['labels'].cpu())
    labels = np.array(prediction['labels'].cpu())
    scores = np.array(prediction['scores'].cpu())
    boxes = np.array(prediction['boxes'].cpu())
    filtered_prediction = {'labels':[], 'scores':[], 'boxes':[]}
    for label in set(filter_labels):
        loc = np.where(labels == label)[0]
        if len(loc) > 0: 
            label_scores = scores[loc]
            if max(label_scores) > threshold:
                box = boxes[loc[np.where(label_scores == max(label_scores))]]
                filtered_prediction['labels'].append(label)
                filtered_prediction['scores'].append(max(label_scores))
                filtered_prediction['boxes'].append(box)
    return filtered_prediction


# In[13]:


predictions = []
for i,image in enumerate(dataset.imgs):
    if i %100 == 0:
        logger.info(f'Inference on image :{i+1}')
    img, label = dataset[i]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    prediction = prediction[0]
    predictions.append((i,prediction))


# In[14]:


predictions


# In[14]:


# # Randomly checking 9 images at a time
# import random
# idx = list(range(len(predictions)))
# random.shuffle(idx)
# fig,axes = plt.subplots(figsize = (120,210), dpi = 20, nrows = 3, ncols = 3)
# axes = axes.flatten()
# for i in range(9):
#     j, prediction = predictions[idx[i]]
#     print(f'Local ID: {idx[i]}, Global ID: {j}')
#     img, label = dataset[j]
#     visualize(img, prediction, True, ax = axes[i],erosion = False)


# In[15]:


threshold = 0
filtered_predictions = []
underlabel = {}
limbs = ['LH', 'RH', 'LF', 'RF']
logger.info(f'Apply filter at threshold = {threshold}')
for i,prediction in enumerate(predictions):
    limb = [limb for limb in limbs if limb in dataset.imgs[i]][0]
    LABELS = [NAME_TO_ID_MAP[x] for x in NAME_TO_ID_MAP.keys() if limb in x]
    filtered_prediction = filter_prediction(prediction[1], 0, LABELS)
    filtered_predictions.append((prediction[0], filtered_prediction))
    if len(filtered_prediction['labels']) < len(LABELS):
        underlabel[i] = set(LABELS) - set(filtered_prediction['labels'])
logger.warning(f'Number of underlabel images: {len(underlabel)}')
logger.warning(f'Number of underlabel joints: {len([y for x in underlabel.values() for y in x])}')
logger.warning(f'Underlabels: {underlabel}')


# In[16]:


#underlabel


# In[17]:


# # Randomly checking 9 images at a time
# import random
# #idx = list(range(len(filtered_predictions)))
# idx = list(underlabel.keys())
# random.shuffle(idx)
# fig,axes = plt.subplots(figsize = (120,210), dpi = 20, nrows = 3, ncols = 3)
# axes = axes.flatten()
# for i in range(9):
#     j, prediction = filtered_predictions[idx[i]]
#     print(f'Local ID: {idx[i]}, Global ID: {j}')
#     img, label = dataset[j]
#     visualize(img, prediction, True, ax = axes[i],erosion = False)


# ### Write out joint images

# In[18]:


joint_count = 0
for i in range(len(dataset)):
    image_id = dataset.imgs[i].strip('.jpg')
    img,_ = dataset[i]
    img = img.mul(255).permute(1,2,0).byte().numpy()
    for j,box in enumerate(filtered_predictions[i][1]['boxes']):
        xmin, ymin, xmax, ymax = box[0]
        label = int(filtered_predictions[i][1]['labels'][j])
        joint_img = Image.fromarray(img[int(np.ceil(ymin)):int(np.floor(ymax)),int(np.ceil(xmin)):int(np.floor(xmax)), :], 'RGB')
        joint_img.save(os.path.join(JOINTOUTPUT, f'{image_id}-{label}.jpg'))
        joint_count += 1
    if i%100 == 1:
        logger.info(f'Wrote joint images detected from {i} images')
    logger.info(f'Wrote {len(filtered_predictions[i][1]["labels"])} joint images for {image_id}')
logger.info(f' Wrote {joint_count} joint images for {len(dataset)}')
logger.info('Complete!')


# ### Copy the feet joints over but be careful about the label

# In[21]:


import shutil
from data import ID_TO_NAME_MAP_NARROWING
count = 0
for img in os.listdir(handle +'/test/narrowing_all'):
    if ('LF' in img) or ('RF' in img):
        name = img.strip('.jpg').split('-')
        name[2] = str(NAME_TO_ID_MAP[ID_TO_NAME_MAP_NARROWING[int(name[2])]])
        count += 1
        shutil.copy(os.path.join(handle +'/test/narrowing_all', img), os.path.join(handle +'/test/erosion_all', '-'.join(name)+'.jpg'))
logger.info(f'Copied {count} feet joints from narrowing to erosion') 


# In[22]:


torch.cuda.empty_cache()

