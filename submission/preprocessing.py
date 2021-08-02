from matplotlib import pyplot as plt 
from matplotlib import figure, colors
from matplotlib.patches import Rectangle
import xmltodict
import os
#%matplotlib inline
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

def get_label(label_file):
    with open(label_file) as fd:
        doc = xmltodict.parse(fd.read())
        return doc
        #return [x for x in doc['annotation']['object']]        

def get_img_and_label(img_folder, label_folder, idx = 0):
    imgs = list(sorted(os.listdir(img_folder)))
    labels = list(sorted(os.listdir(label_folder)))
    img = Image.open(os.path.join(img_folder,imgs[idx])).convert("RGB")
    label = get_label(os.path.join(label_folder, labels[idx]))
    return img, label

def visualize_raw(img = None,label = None,img_path = None, label_path = None):
    if img_path is not None:
        img = Image.open(img_path)
    if label_path is not None:
        with open(label_path, 'r') as f:
            label = xmltodict.parse(f.read())
    fig,ax = plt.subplots(figsize = (20,10))
    ax.imshow(img)
    for i,box in enumerate(label['annotation']['object']):
        vertices = box['bndbox']
        name = box['name']
        xmin, ymin, xmax, ymax =[float(x) for x in vertices.values()]
        norm = colors.Normalize(vmin=-5, vmax=5)
        cmap = plt.cm.rainbow
        ax.add_patch(Rectangle((xmin,ymin), xmax- xmin,ymax-ymin, alpha = 0.2))
        plt.text((xmin + xmax)/2, (ymin + ymax)/2,name, color = 'white')

def manual_map_img(label_path, new_label_path):
    # This will update the labels in the repo to match the challenge
    with open(label_path) as fd:
        doc = xmltodict.parse(fd.read())
    label = doc['annotation']['object']  
    label = [bb for bb in label if bb['name'] != 'DIP']
    Wrist = [float(bb['bndbox']['xmin'])/2 + float(bb['bndbox']['xmax'])/2 for bb in label if bb['name'] == 'Wrist'].pop()
    Ulna = [float(bb['bndbox']['xmin'])/2 + float(bb['bndbox']['xmax'])/2 for bb in label if bb['name'] == 'Ulna'].pop()
    left = Wrist > Ulna
    if not left:
        print(label_path)
    pips = {i:(float(bb['bndbox']['xmin'])/2 + float(bb['bndbox']['xmax'])/2) for i,bb in enumerate(label) if bb['name'] == 'PIP'}
    pips = sorted(pips, key = lambda i: pips[i], reverse = left)
    for i,pip in enumerate(pips):
        label[pip]['name'] = ('LH' if left else 'RH') + '_pip_' +str(i+1) 
        
    mcps = {i:(float(bb['bndbox']['xmin'])/2 + float(bb['bndbox']['xmax'])/2) for i,bb in enumerate(label) if bb['name'] == 'MCP'}
    mcps = sorted(mcps, key = lambda i: mcps[i], reverse = left)
    for i,mcp in enumerate(mcps):
        label[mcp]['name'] = ('LH' if left else 'RH') + '_mcp_' +str(i+1)
    label = [bb for bb in label if not(bb['name'].endswith('pip_1'))]
    doc['annotation']['object']  = label   
    with open(new_label_path, 'w') as result_file:
        result_file.write(xmltodict.unparse(doc))
    
def manual_map(label_folder, new_label_folder):
    if not os.path.exists(new_label_folder):
        os.mkdir(new_label_folder)
    label_paths = os.listdir(label_folder)
    for idx in range(len(label_paths)):
        manual_map_img(os.path.join(label_folder, label_paths[idx]),os.path.join(new_label_folder, label_paths[idx]))
