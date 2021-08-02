#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%load_ext autoreload
#%autoreload 2
#%matplotlib inline


# In[1]:


#handle = '/output'
handle = './output'


# In[2]:


from PIL import ImageOps, Image
import PIL.ImageOps
import piexif
import os
import shutil
import logging


# In[3]:


print(os.getcwd())
print(os.listdir())


# In[4]:


os.mkdir(handle + '/logs')
os.mkdir(handle + '/test')
os.mkdir(handle + '/test/narrowing_all')
os.mkdir(handle + '/test/erosion_all')


# In[5]:


logger = logging.getLogger('0')
hdlr = logging.FileHandler(handle + '/logs/0-preprocessing.log')
formatter = logging.Formatter('[%(asctime)s][%(levelname)s]   %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


# In[6]:


logger.info(f'{os.listdir(handle)}')


# In[7]:


PATH = '/test' if handle == '/output' else './test'
NEWPATH = handle + '/test/images'


# In[8]:


if os.path.exists(NEWPATH):
    shutil.rmtree(NEWPATH)
shutil.copytree(PATH,NEWPATH)
logger.info(f'Copy from {PATH} to {NEWPATH}')


# In[9]:


temp = []
for f in os.listdir(NEWPATH):
    if not f.endswith('.jpg'):
        os.remove(os.path.join(NEWPATH,f))
        logger.info(f'Remove {f}')
    else:
        img = Image.open(os.path.join(NEWPATH, f))
        exif_dict = piexif.load(os.path.join(NEWPATH, f))
        if 274 not in exif_dict['0th']:
            logger.info(f'No image metadata: {f}')
        elif exif_dict['0th'][274] == 3:
            exif_dict['0th'][274] = 1
            exif_bytes = piexif.dump(exif_dict)
            PIL.ImageOps.mirror(PIL.ImageOps.flip(img)).save(os.path.join(NEWPATH, f), 'jpeg')
            piexif.insert(exif_bytes, os.path.join(NEWPATH, f))
            logger.warning(f'Flipped: {f}')
        elif exif_dict['0th'][274] != 1:
            logger.warning(f'Check orientation: {f}')
logger.info('Done with preprocessing')


# In[ ]:





# In[ ]:





# In[ ]:




