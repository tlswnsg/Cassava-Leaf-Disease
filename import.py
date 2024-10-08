import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
from torch.utils.data import random_split, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

import glob
import torchvision.transforms as T
from PIL import Image
import PIL

import os
import math

learning_rate = 1e-3
epoch = 50
