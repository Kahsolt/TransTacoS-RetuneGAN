#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/04/16 

# NOTE: proxy by TransTacoS for finetune

import os
import sys
from importlib import import_module
BASH_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASH_PATH)
print('TRANSTACOS_PATH:', os.path.join(BASH_PATH, 'transtacos'))
AP = import_module('transtacos.audio')
