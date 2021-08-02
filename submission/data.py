import torch
from preprocessing import *
from torchvision.transforms import functional

NAME_TO_ID_MAP_NARROWING = {
 'Empty':0,
 'LF_mtp_1': 1,
 'LF_mtp_2': 2,
 'LF_mtp_3': 3,
 'LF_mtp_4': 4,
 'LF_mtp_5': 5,
 'LF_mtp_ip': 6,
 'LH_mcp_1': 7,
 'LH_mcp_2': 8,
 'LH_mcp_3': 9,
 'LH_mcp_4': 10,
 'LH_mcp_5': 11,
 'LH_pip_2': 12,
 'LH_pip_3': 13,
 'LH_pip_4': 14,
 'LH_pip_5': 15,
 'LH_wrist_capnlun': 16,
 'LH_wrist_cmc3': 17,
 'LH_wrist_cmc4': 18,
 'LH_wrist_cmc5': 19,
 'LH_wrist_mna': 20,
 'LH_wrist_radcar': 21,
 'RF_mtp_1': 22,
 'RF_mtp_2': 23,
 'RF_mtp_3': 24,
 'RF_mtp_4': 25,
 'RF_mtp_5': 26,
 'RF_mtp_ip': 27,
 'RH_mcp_1': 28,
 'RH_mcp_2': 29,
 'RH_mcp_3': 30,
 'RH_mcp_4': 31,
 'RH_mcp_5': 32,
 'RH_pip_2': 33,
 'RH_pip_3': 34,
 'RH_pip_4': 35,
 'RH_pip_5': 36,
 'RH_wrist_capnlun': 37,
 'RH_wrist_cmc3': 38,
 'RH_wrist_cmc4': 39,
 'RH_wrist_cmc5': 40,
 'RH_wrist_mna': 41,
 'RH_wrist_radcar': 42}

NAME_TO_ID_MAP_EROSION = {
 'Empty':0, 
 'LF_mtp_1': 1,
 'LF_mtp_2': 2,
 'LF_mtp_3': 3,
 'LF_mtp_4': 4,
 'LF_mtp_5': 5,
 'LF_mtp_ip': 6,
 'LH_mcp_1': 7,
 'LH_mcp_2': 8,
 'LH_mcp_3': 9,
 'LH_mcp_4': 10,
 'LH_mcp_5': 11,
 'LH_mcp_ip': 12,
 'LH_pip_2': 13,
 'LH_pip_3': 14,
 'LH_pip_4': 15,
 'LH_pip_5': 16,
 'LH_wrist_lunate': 17,
 'LH_wrist_mc1': 18,
 'LH_wrist_mul': 19,
 'LH_wrist_nav': 20,
 'LH_wrist_radius': 21,
 'LH_wrist_ulna': 22,
 'RF_mtp_1': 23,
 'RF_mtp_2': 24,
 'RF_mtp_3': 25,
 'RF_mtp_4': 26,
 'RF_mtp_5': 27,
 'RF_mtp_ip': 28,
 'RH_mcp_1': 29,
 'RH_mcp_2': 30,
 'RH_mcp_3': 31,
 'RH_mcp_4': 32,
 'RH_mcp_5': 33,
 'RH_mcp_ip': 34,
 'RH_pip_2': 35,
 'RH_pip_3': 36,
 'RH_pip_4': 37,
 'RH_pip_5': 38,
 'RH_wrist_lunate': 39,
 'RH_wrist_mc1': 40,
 'RH_wrist_mul': 41,
 'RH_wrist_nav': 42,
 'RH_wrist_radius': 43,
 'RH_wrist_ulna': 44}
 
ID_TO_NAME_MAP_NARROWING = {NAME_TO_ID_MAP_NARROWING[key]:key for key in NAME_TO_ID_MAP_NARROWING}
ID_TO_NAME_MAP_EROSION = {NAME_TO_ID_MAP_EROSION[key]:key for key in NAME_TO_ID_MAP_EROSION}

class RADataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, label_folder = None, transforms=None, score='Erosion'):
        assert score in ('Erosion', 'Narrowing')
        print(f'Set score = {score}')
        self.score = score
        self.transforms = transforms
        self.img_folder = data_path + '/images'
        if self.score == 'Erosion':
            self.label_folder = data_path + '/erosion_labels'
        else:
            self.label_folder = data_path + '/narrowing_labels'
        if os.path.exists(self.label_folder):
            self.labels = list(sorted(os.listdir(self.label_folder)))
            self.imgs = list(sorted([label.replace('.xml', '.jpg') for label in self.labels]))
            assert len(set(self.imgs) - set(os.listdir(self.img_folder))) == 0, print(set(self.imgs) - set(os.listdir(self.img_folder)))
        else:
            if score == 'Erosion':
                self.imgs = list(sorted([img for img in os.listdir(self.img_folder) if ('LH' in img) or ('RH' in img)]))
            else:
                self.imgs = list(sorted(os.listdir(self.img_folder)))
    def __getitem__(self, idx):
        # load images ad masks
        img = Image.open(os.path.join(self.img_folder,self.imgs[idx])).convert("RGB")
        target = {}
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        if hasattr(self, "labels"):
            label = get_label(os.path.join(self.label_folder, self.labels[idx]))  
            boxes = []
            ids = []                            
            def parse_bb(bb):
                if self.score == 'Erosion':
                    if bb['name'] in NAME_TO_ID_MAP_EROSION:
                        xmin,ymin,xmax,ymax = bb['bndbox'].values()
                        boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                        ids.append(NAME_TO_ID_MAP_EROSION[bb['name']])
                elif self.score == 'Narrowing':
                    if bb['name'] in NAME_TO_ID_MAP_NARROWING:
                        xmin,ymin,xmax,ymax = bb['bndbox'].values()
                        boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                        ids.append(NAME_TO_ID_MAP_NARROWING[bb['name']])
            if 'object' not in label['annotation']: #NOTE: This is to handle the quirkiness of xmltodict.unparse
                print(f'ERROR: There is no object in the annotation for image {image_id[0]}')
            else:
                if type(label['annotation']['object']) == list:
                    for bb in label['annotation']['object']:
                        parse_bb(bb)
                else:
                    parse_bb(label['annotation']['object']) #NOTE: This is to handle the quirkiness of xmltodict.unparse

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            ids = torch.as_tensor(ids, dtype=torch.int64)
            target["boxes"] = boxes
            target["labels"] = ids
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)

def visualize(img,labels,result = False, ax = False, filter_labels = None,erosion = False):
    """
    Visualize a single image with label
    """
    if result:
        img = img.mul(255).permute(1,2,0).byte().numpy()
    else:
        img = np.rollaxis(np.array(img),0,3)
    if not ax:
        fig,ax = plt.subplots(figsize = (20,10))
    ax.imshow(img)
    if type(labels) == dict:    
        for i,box in enumerate(labels['boxes']):
            if not type(labels['labels']) == list:
                label = int(labels['labels'].cpu()[i])
            else:
                label = int(labels['labels'][i])
            if filter_labels is not None:
                if label not in filter_labels:
                    continue
            if erosion:
                label_ = ID_TO_NAME_MAP_EROSION[label]            
            else:            
                label_ = ID_TO_NAME_MAP_NARROWING[label]
            xmin, ymin, xmax, ymax = box.flatten()
            norm = colors.Normalize(vmin=-5, vmax=5)
            cmap = plt.cm.rainbow
            ax.add_patch(Rectangle((xmin,ymin), xmax- xmin,ymax-ymin, alpha = 0.2, edgecolor='yellow',linewidth=2.0))
            ax.text((xmin + xmax)/2, (ymin + ymax)/2, f'{label}:{label_}', color = 'white', size = 10)

def visualize_multiple(dataset,idx,result = False, filter_labels = None, erosion = False):
    """
    Visualize 9 images from the dataset based on a list of indices
    """
    ncols = 3
    fig,axes = plt.subplots(figsize = (80,80), dpi = 20, ncols = ncols, nrows = round(len(idx)/ncols))
    axes = axes.flatten()
    for j,i in enumerate(idx):
        img,labels = dataset[i]
        if result:
            img = img.mul(255).permute(1,2,0).byte().numpy()
        else:
            img = np.rollaxis(np.array(img),0,3)
        plt.tight_layout()
        axes[j].set_title(f'Sample # {i}: {dataset.imgs[i]}', size = 60)
        axes[j].axis('off')
        axes[j].imshow(img)
        for t,box in enumerate(labels['boxes']):
            if not type(labels['labels']) == list:
                label = int(labels['labels'].cpu()[t])
            else:
                label = int(labels['labels'][t])
            if filter_labels is not None:
                if label not in filter_labels:
                    continue
            if erosion:
                label_ = ID_TO_NAME_MAP_EROSION[label]            
            else:            
                label_ = ID_TO_NAME_MAP_NARROWING[label]
            xmin, ymin, xmax, ymax = box.flatten()
            norm = colors.Normalize(vmin=-5, vmax=5)
            cmap = plt.cm.rainbow
            axes[j].add_patch(Rectangle((xmin,ymin), xmax- xmin,ymax-ymin, alpha = 0.2))
            axes[j].text((xmin + xmax)/2, (ymin + ymax)/2,f'{label}:{label_}', color = 'white', size = 40)        
