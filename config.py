"""
Configurations for dataset and inference.

Do ont include training configurations.
"""
# Use which model params to inference and get output in the program.
INFERENCE_PARAMS_PATH = "./model/train.6_full.pth"

class SSourceConfig(object):
    """
    Subtitle source configurations.
    
    The source is a default set of configurations that ensure the subtitle
    can work fine but do not give higher functionalities.
    
    Attributes:
        headers: Standard script info headers.
        v4plus_pairs: Default v4plus settings.
        v4_pairs: Default v4 settings.
        events_pairs: Default events format. None stands for API to offer explicit value.
    """
    
    headers = {
        "Title": None,
        "Original Script": "ASG",
        "PlayResX": None,
        "PlayResY": None,
        "Timer": 100.0000,
    }
    
    v4plus_pairs = {
        "Name": "chs",
        "Fontname": "simhei",
        "Fontsize": 20,
        "PrimaryColour": "&H00ffffff",
        "SecondaryColour": "&H0000ffff",
        "OutlineColour": "&H00000000",
        "BackColour": "&H80000000",
        "Bold": 1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 90,
        "ScaleY": 90,
        "Spacing": 0,
        "Angle": 0.00,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 2,
        "Alignment": 2,
        "MarginL": 20,
        "MarginR": 20,
        "MarginV": 15,
        "Encoding": 1,
    }
    
    v4_pairs = {
        "Name": "eng",
        "Fontname": "Arial Narrow",
        "Fontsize": 12,
        "PrimaryColour":"&H00ffeedd",
        "SecondaryColour": "&H00ffc286",
        "TertiaryColour": "&H00000000",
        "BackColour": "&H80000000",
        "Bold": -1,
        "Italic": 0,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 0,
        "Alignment": 2,
        "MarginL": 20,
        "MarginR": 20,
        "MarginV": 2,
        "AlphaLevel": 0,
        "Encoding": 1,
    }
    
    # Unchangeable attributes.
    events_pairs = {
        "Marked": 0,
        "Start": None,
        "End": None,
        "Style": None,
        "Name": "",
        "MarginL": "0000",
        "MarginR": "0000",
        "MarginV": "0000",
        "Effect": "",
        "Text": "xxx",
    }
    
    content = "xxx"
    

class PostProcessConfig(object):
    
    standard_dialogure_break = 0.1
    
    loose_dialogue_threshold = 2 * standard_dialogure_break
    loose_dialogue_delay = loose_dialogue_threshold / 5
    
    max_sigle_speech_length = 1
    
class InferenceConfig(object):
    
    best_around_period = 1
    
    threshold = 0.8
    
    coding_map = {
        0: "non-speech",
        1: "speech",
    }