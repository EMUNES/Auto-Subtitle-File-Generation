"""Configurations for dataset and inference.

Do ont include training configurations.
"""

import os


### THOSE PACHAGES are not used but needed as backend.
import ffmpeg

# Use which model params to inference and get output in the program.

# Abosulte path for this project root folder.
# Using relative path system causing many troubles.
ROOT_PATH_ABS = os.path.split(os.path.realpath(__file__))[0]
INF_OUTPUT_FOLDER_ABS = ROOT_PATH_ABS + "/inf/output"
RESULT_FOLDER_ABS = ROOT_PATH_ABS + "/results"
TEMP_FOLDER_ABS = ROOT_PATH_ABS + "/temp"

INFERENCE_PARAMS_PATH = ROOT_PATH_ABS + "/models/model_best.pth"


class DecoderConfig(object):
    """Configrations for decoder.
    
    Those trimming bias should be set between 0.2 ~ 0.3.
    Too large you will lose words. 
    To Small there will be more false speech label.
    """
    
    trimming_end = 0.33
    trimming_start = 0.25


class PostProcessConfig(object):
    """
    Basic params for postprocess.
    """
    
    # The least break between dialogues.
    standard_dialogue_break = 0.1
    
    # If there is a large gap betweeen dialogues, we extend the subtitle,
    # giving a slightly more time.
    loose_dialogue_threshold = 2 * standard_dialogue_break
    loose_dialogue_delay = loose_dialogue_threshold / 4
    
    # A bias cause for the detection somehow.
    global_bias = 0.25

    # TODO: Adust this parameter and use it in postprocess with hooks.
    # How long does one speech takes to make the program cut it.
    max_sigle_speech_length = 1
    
    
class InferenceConfig(object):
    """
    Important params for predicting the output.
    """
    
    # How long should we make the inference once a clip that can give the best result.
    ### Default 1 is a good choice for this task.
    best_around_period = 1
    
    # Giving how much we take the period as a dialogue.
    ### The higher the threshold, the lesser duration for each speech.
    ### And more breaks for dialogues.
    threshold = 0.6
    
    # The coding map for deep learning model.
    ### Change this will only cause bugs. I shouldn't write those code here.
    coding_map = {
        0: "non-speech",
        1: "speech",
    }
    
    
class STTInferenceConfig(object):
    """Configurations for Speech To Text task.
    
    properties:
        language_choices: Available languages supported in STT.
    """
    
    language_choices = {
        "eng": "English",
        "chs": "Chinese",
    }
    
    def __init__(self) -> None:
        super().__init__()
    
    def iter_avl_langs(self, verbose=False):
        for lang in self.language_choices.keys():
            if verbose:
                print(f"{lang} - {self.language_choices[k]}")
            language_name = self.language_choices[lang]
            yield lang, language_name


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
        "Original Script": "ASFG",
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
    
