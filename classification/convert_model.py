import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import random
import os

import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm
from scipy.ndimage import label, center_of_mass

from models import *

if __name__ == "__main__":

    out_name = ["classification_st1_left.pth", "classification_st1_right.pth", "classification_st2.pth"] #, "classification_ax_left.pth", "classification_ax_right.pth"]

    for out in tqdm.tqdm(out_name, desc="Converting"):
        backbones_768 = ['focalnet_small_lrf.ms_in1k', 'cs3darknet_m.c2ns_in1k', 'convnextv2_tiny.fcmae_ft_in1k', 'twins_svt_base.in1k']
        model = REM_Script(
            n_classes=3,
            n_fold_classifier=3,
            backbones=backbones_768,
        )

        model.load_state_dict(torch.load(out))
        scripted_model = torch.jit.script(model)
        scripted_model.save(out.replace(".pth", ".ts"))
