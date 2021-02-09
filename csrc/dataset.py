"""
Dataset for training.
"""

import numpy as np
import  soundfile as sf
import  torch.utils.data as data

from csrc.configurations import DatasetConfig as DC

PERIOD = 5 

# The only usuable dataset set for now.
class PANNsDataset(data.Dataset):
    def __init__(
            self,
            file_list,
            training_folder,
            test_folder,
            waveform_transforms=None,
            test: bool=False):
        self.file_list = file_list 
        self.waveform_transforms = waveform_transforms
        self.test = test
        self.working_dir = test_folder if test else training_folder

    def __getitem__(self, idx: int):
        wav_path = self.working_dir / self.file_list[idx]

        # Deal with the audio file
        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y, sample_rate=DC.dataset_sample_rate)
            
        len_y = len(y)
        effective_length = sr * PERIOD
        # In case the clips are not precisely cropped
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)
        y = np.nan_to_num(y)
            
        # Get the right label from the file name
        labels = np.zeros(2, dtype=np.float32)
        label = int(str(wav_path).split(".")[0][-1])
        labels[label] = 1
        #labels = torch.from_numpy(labels)

        return {"waveform": y, "targets": labels}

    def __len__(self):
        return len(self.file_list)


