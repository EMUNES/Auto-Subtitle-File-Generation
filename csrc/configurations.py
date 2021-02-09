"""
Those are the configuration settings.
"""

class ModelConfig(object):
    # Configurations for SED model.
    sed_model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": 527,
    }
    
class DatasetConfig(object):
    """
    Configuration for building your own dataset from sources.
    
    Attributes:
        dataset_clip_time(int): Clip length for dataset. Default 5s.
        dataset_sample_rate(int): Clip length for dataset. Default 32000.
        dataset_audio_format(str): Clip format for dataset. Default using "wav"
        sub_encoding(str): Use "utf-8" encoding for Aegisub subtitle support.
    """
    
    dataset_clip_time = 3 # seconds
    
    dataset_sample_rate = 32000
    
    dataset_audio_format = "wav" 
    
    sub_encoding = "utf-8" # recommanded

class InferenceConfig(object):
    """
    Configurations for Inference.
    
    Attributes: 
        period(int): Clip length for inference loop.
        coding_map(dict): Tag encoding. DO NOT change.
        threshold(float): Threshold for picking speech (0~1).
    """
    
    coding_map = {
        0: "non-speech",
        1: "speech",
    }