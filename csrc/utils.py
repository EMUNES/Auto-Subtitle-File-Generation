"""
Util functions for deep learning.
"""

from pathlib import Path
import random

import numpy as np
import torch
import os

from config import ROOT_PATH_ABS

def seed_all(s:int=42) -> None: 
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def seed_dataset(s:int=42) -> None:
    random.seed(s)
    

class TrainingDirs(object):
    """
    Initiate the working directory systems for training.
    """
    
    def __init__(self, dsname: str, pre_test: bool) -> None:
        super().__init__()
        ROOT = Path(ROOT_PATH_ABS)
        INPUT_ROOT = ROOT / 'data'
        TARGET_AUDIO_DIR = INPUT_ROOT / dsname
        print(f"Working with dataset under {TARGET_AUDIO_DIR}.")
        assert os.path.exists(TARGET_AUDIO_DIR), "Input dataset folder does not exist."
        
        self.dataset_folder = TARGET_AUDIO_DIR
        self.train_folder = TARGET_AUDIO_DIR / 'train' if pre_test else TARGET_AUDIO_DIR
        self.test_folder = TARGET_AUDIO_DIR / 'test' if pre_test else None
    