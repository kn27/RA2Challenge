#!/usr/bin/env python
# coding: utf-8

import os 
import shutil

if os.path.exists('./models'):
    shutil.rmtree('./models')
if os.path.exists('./utils'):
    shutil.rmtree('./utils')
if os.path.exists('./data.py'):
    os.remove('./data.py')
if os.path.exists('./preprocessing.py'):
    os.remove('./preprocessing.py')
if os.path.exists('./model.py'):
    os.remove('./model.py')

shutil.copytree('../models', './models')
shutil.copytree('../utils', './utils')
shutil.copyfile('../data.py', './data.py')
shutil.copyfile('../preprocessing.py', './preprocessing.py')
shutil.copyfile('../model.py', './model.py')

#shutil.rmtree('./test')
#shutil.rmtree('./output')

