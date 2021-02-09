"""
Audio augmentations.

Those augmentation methods should not change the length of the audio file.
"""

from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, PitchShift, AddBackgroundNoise

BaseAug = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    AddGaussianSNR(p=0.3),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
])
