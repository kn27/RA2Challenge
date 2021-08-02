#!/usr/bin/env python
# coding: utf-8

# In[8]:


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


# In[9]:


# from matplotlib import pyplot as plt 
# from matplotlib import figure, colors
# from matplotlib.patches import Rectangle
#from data import visualize, visualize_multiple
#import PIL.ExifTags
#from PIL import ImageOpsimport xmltodict
#from torchvision.transforms import functional as F
#from engine import train_one_epoch

# import os
# import numpy as np
# import torch
# import torch.utils.data
# from PIL import Image
# import random
# from model import fasterrcnn_resnet50_fpn
# import utils.transforms, utils.datasets, utils.optimizer


# In[10]:


handle = '/output'
#handle = './output'


# In[11]:


from data import ID_TO_NAME_MAP_EROSION as ID_TO_NAME_MAP, NAME_TO_ID_MAP_EROSION as NAME_TO_ID_MAP
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd


# In[12]:


import logging
logger = logging.getLogger('0')
hdlr = logging.FileHandler(handle  +'/logs/4-erosionscore.log')
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]   %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


# In[13]:


TEST = '/test/'
PATH = './training.csv'
JOINTINPUT = handle  +'/test/erosion_all'
PRETRAIN_MODEL = './pretrained_models/narrow_score_75.pth'#'./pretrained_model/narrow_score_80_with_batchnormfull_scale_50.pth'
SCOREOUTPUT = handle  +'/erosion_score.csv'
logger.info (f'INPUT:{JOINTINPUT}, OUTPUT:{SCOREOUTPUT}, PRETRAIN_MODEL:{PRETRAIN_MODEL}')


# In[14]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# ### Data

# In[15]:


class scale(object):
    def __call__(self, image):
        image = image/image.mean()*0.5
        return image
    
class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.transforms = transform
        self.img_folder = data_path 
        self.imgs = list(sorted(os.listdir(self.img_folder)))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder,self.imgs[idx])).convert("RGB")
        target = {}
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imgs)


# In[16]:


transform = transforms.Compose(   
    [transforms.Resize((128,128)),
    transforms.ToTensor(),
    scale(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ScoreDataset(JOINTINPUT, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=2, shuffle = False)
logger.info(f'Length of dataset: {len(dataloader)}')


# In[17]:


class Base_Classification_Net2(nn.Module):
    def __init__(self, num_classes):
        super(Base_Classification_Net2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 1024, 3, stride = 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 5 * 5, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[18]:


net = Base_Classification_Net2(num_classes = 5)
net.to(device)
net.load_state_dict(torch.load(PRETRAIN_MODEL))


# In[19]:


def validate(model,validloader, device):
    correct = 0
    total = 0
    all_predictions = []
    with torch.no_grad():
        for i,images in enumerate(validloader):
            logger.info(f'Inference on batch {i}')
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += predicted.size(0)
            all_predictions.append(predicted)
    all_predictions = np.array([float(x) for x in torch.cat(all_predictions,0).cpu()])
    return all_predictions


# In[20]:


predictions = validate(net, dataloader, device)


# In[21]:


assert len(predictions) == len(dataset)


# In[26]:


output = []
for i,x in enumerate(dataset.imgs):
    x = x.strip('.jpg').split('-')
    name = ID_TO_NAME_MAP[float(x[2])].split('_')
    name.insert(2, 'E_')
    x[2] = '_'.join(name)
    if x[1] in x[2]:
        x.pop(1)
        x.append(predictions[i])
        output.append(x)
    else:
        logger.error(f'There is a mislabel {x[2]} for image {dataset.imgs[i]}')
output = pd.DataFrame(output, columns = ['Patient_ID', 'Joint', 'Score'])
pivoted = output.pivot(index = 'Patient_ID', columns = 'Joint', values = 'Score')
logger.warning(f'Number of missing per joint: {dict(pivoted.isna().sum()[pivoted.isna().sum()>0])}')
pivoted = pivoted.reset_index().sort_values(by = 'Patient_ID',ascending = True)


# In[34]:


output


# In[28]:


pivoted.T


# In[22]:


Patient_ID = set([x.split('-')[0] for x in os.listdir(TEST) if x.endswith('.jpg')])
missing = Patient_ID - set(pivoted.Patient_ID)
for item in missing:
    pivoted.loc[pivoted.shape[0], 'Patient_ID'] = item

columns = pd.read_csv(PATH).columns
columns = [x for x in columns if 'E' in x]
pivoted = pivoted.loc[:,['Patient_ID'] + columns]
pivoted = pivoted.fillna(2)


# In[25]:


pivoted.RF_mtp_E__ip


# In[15]:


pivoted.to_csv(SCOREOUTPUT,index = False)
logger.info(f'Wrote erosion output to :{SCOREOUTPUT}!')

