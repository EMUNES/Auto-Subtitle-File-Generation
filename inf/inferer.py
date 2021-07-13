import numpy as np
from numpy.lib.utils import source
import pandas as pd
import torch
import librosa
import soundfile

from utils import check_type, extract_audio, mono_load
from config import InferenceConfig  as IC
from csrc.configurations import DatasetConfig as DC
from csrc.configurations import ModelConfig as MC
from csrc.dataset import PANNsDataset
from csrc.models import PANNsCNN14Att, AttBlock
from inf.post import SpeechSeries
from stt.vosk_api import get_by_ffmpeg
from config import TEMP_FOLDER_ABS

# Those configurations are from csrc configurations and should not be altered here.
### Those parameters are used for standard clip inference not for breakpoint timestamp.
PERIOD = IC.best_around_period
THRESHOLD = IC.threshold
CODING = IC.coding_map 
SR = DC.dataset_sample_rate


class Pannscnn14attInferer():
    
    def __init__(self, clip_y, model_path, period=0, device=None, ds=None):
        self.clip_y = clip_y
        self.model_path = model_path
        
        self.period = period
        self.device = device
        self.ds = ds if ds else PANNsDataset
        
        self.model = PANNsCNN14Att(**MC.sed_model_config)
        self.model.att_block = AttBlock(2048, 2, activation='sigmoid')
        self.model.att_block.init_weights()
        self.model.load_state_dict(torch.load(self.model_path)['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def make_inference_result(self):
        audios = []
        start = 0
        end = PERIOD * SR

        # Split audio into clips.
        while True:
            y_batch = self.clip_y[start:end].astype(np.float32)
            if len(y_batch) != PERIOD * SR:
                y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
                y_pad[:len(y_batch)] = y_batch
                audios.append(y_pad)
                break
            start = end
            end += PERIOD * SR
            audios.append(y_batch)

        # Get tensors
        arrays = np.asarray(audios)
        tensors = torch.from_numpy(arrays)

        estimated_event_list = []
        global_time = 0.0
        for image in tensors:
            image = image.view(1, image.size(0))
            image = image.to(self.device)

            with torch.no_grad():
                prediction = self.model(image)
                framewise_outputs = prediction["framewise_output"].detach(
                    ).cpu().numpy()[0]
                    
            thresholded = framewise_outputs >= THRESHOLD

            for target_idx in range(thresholded.shape[1]):
                if thresholded[:, target_idx].mean() == 0:
                    pass
                else:
                    detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
                    head_idx = 0
                    tail_idx = 0
                    while True:
                        if (tail_idx + 1 == len(detected)) or (
                                detected[tail_idx + 1] - 
                                detected[tail_idx] != 1):
                            onset = 0.01 * detected[
                                head_idx] + global_time
                            offset = 0.01 * detected[
                                tail_idx] + global_time
                            onset_idx = detected[head_idx]
                            offset_idx = detected[tail_idx]
                            max_confidence = framewise_outputs[
                                onset_idx:offset_idx, target_idx].max()
                            mean_confidence = framewise_outputs[
                                onset_idx:offset_idx, target_idx].mean()
                            estimated_event = {
                                "speech_recognition": CODING[target_idx],
                                "start": onset,
                                "end": offset,
                                "max_confidence": max_confidence,
                                "mean_confidence": mean_confidence,
                            }
                            estimated_event_list.append(estimated_event)
                            head_idx = tail_idx + 1
                            tail_idx = tail_idx + 1
                            if head_idx >= len(detected):
                                break
                        else:
                            tail_idx += 1
            global_time += PERIOD
            
        return estimated_event_list
    
    def get_breakpoint(self):
        pass
    

class SttInferer():
    """Speech To Text inference using vosk api.
    
    This inferer will receive the DataFrame from SED inferer and calling
    vosk api to run recognization for each detected clip.
    """
    
    def __init__(self, sed_df: pd.DataFrame, targ_path, source_lang="eng") -> None:
        self.df = sed_df
        self.target_file_path = targ_path
        self.lang = source_lang
        
    def _voice_split(self):
        pass
        
    def _voice_recognize(self, onset: float, offset: float, callback=False):
        """Recognize text in a clip.
        
        Args:
            onset: The begging of this clip (seconds).
            offset: The ending point of this clip (seconds).
            lang: The language spoken in this clip.
            callback: Whether to use callback function to split the clip again.
        
        Returns: 
            event_text: The text of this clip.
        """
        event_onset = onset
        event_duration = offset - onset
        
        print(f"Running voice recogniztion on {event_onset} to {offset}...")    
        
        # Cut the clip and write it in temp folder.
        y, sr = librosa.load(self.target_file_path, sr=None, offset=event_onset, duration=event_duration)
        soundfile.write(file=TEMP_FOLDER_ABS+"/"+"stt_temp.wav", data=y, samplerate=sr, format="wav")

        # Run recognization to the file in the temp folder.
        event_text = get_by_ffmpeg.ffmpeg_sst("stt_temp.wav", lang=self.lang)
        
        return event_text
        # Cut the clip out
        
    def make_inference_result(self):
        """Take SED df and generate text for each row.
        """
        text_all = []
        
        for (onset, offset) in zip(self.df.start, self.df.end):
            current_text = self._voice_recognize(onset=onset, offset=offset)
            text_all.append(current_text)
            
        df_with_text = self.df.copy(deep=True)
        df_with_text["recognized_text"] = text_all

        return df_with_text
            
        
def get_inference(targ_file_path, params_path, fname, lang, post_process=True, output_folder="inf/output", short_clip=0, device=None, inferer=None):
    """
    Get the inference result for SED and STT tasks.
    
    Args:
        targ_file_path: Target file path to get the inference result.
        params_path: SED model path.
        fname: File name fot the output.
        post_process: Weather to use postprocess for SED task.
        output_folder: Where to hold the output. Default works fine.
        short_clip: Deprecated. Use vosk instead.
        device: Device used for inference.
        inferer: Declare other inference model for you task.
        
    Returns:
        output_df: The output in pd.DataFrame format. The output_df will be written in *inf/output* folder.
    """
    
    
    output = None
    
    if torch.cuda.is_available():
        device = device if device else torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = inferer if inferer else Pannscnn14attInferer
    print(f"Inferencing using model: {model.__name__}...\n")      
    
    fname = fname if fname else "current"
    out_file = f"{output_folder}/{fname}.csv"
    out_src_file = f"{output_folder}/{fname}-all.csv"
    
    is_video = check_type(targ_file_path)
    targ_file_path = extract_audio(targ_file_path, format=DC.dataset_audio_format) if is_video else targ_file_path
    y, _ = mono_load(targ_file_path)
    
    print("Using model to generate output...\n")
    if short_clip:
        output = model(y, params_path, period=short_clip, device=device).get_breakpoint()
        print(f"Output breakpoint for short clip: {output}\n")
    else:
        output = model(y, params_path, device=device).make_inference_result()
        print(f"Output: {len(output)} breaks.\n")
        prediction_df = pd.DataFrame(output)
        output_df = prediction_df[prediction_df.speech_recognition=="speech"]

        # Whether to use post process for SED task. This is recommanded.
        if post_process:
            output_df = SpeechSeries(output_df).series
            prediction_df = SpeechSeries(prediction_df).series
            print("Post process applied.\n")
        
        # Running Speech TO Text based on SED result.
        if lang: # See run.py. If lang is an empty string (no user input) then stt won't run. It works poorly anyway~
            output_df = SttInferer(output_df, targ_path=targ_file_path, source_lang=lang).make_inference_result()
        
        # Write to local as logs.
        output_df.to_csv(out_file, index=False)
        prediction_df.to_csv(out_src_file, index=False)
        
        print(f"Inference output file generated (This is not the final output), see: {output_folder}.\n")
    
    return output_df
