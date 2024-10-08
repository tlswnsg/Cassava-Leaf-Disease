import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

import glob
import torchvision.transforms as T
from PIL import Image
import PIL

import os
import math

learning_rate = 1e-3
epoch = 50
