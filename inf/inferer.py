import numpy as np
import pandas as pd
import torch

from utils import check_type, extract_audio, mono_load
from config import InferenceConfig  as IC
from csrc.configurations import DatasetConfig as DC
from csrc.configurations import ModelConfig as MC
from csrc.dataset import PANNsDataset
from csrc.models import PANNsCNN14Att, AttBlock
from inf.post import SpeechSeries

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
        
def get_inference(targ_file_path, params_path, fname, post_process=True, output_folder="inf", short_clip=0, device=None, inferer=None):
    output = None
    
    if torch.cuda.is_available():
        device = device if device else torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = inferer if inferer else Pannscnn14attInferer
    print(f"Inferencing using model: {model.__name__}...\n")      
    
    fname = fname if fname else "most-recent-output"
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

        if post_process:
            output_df = SpeechSeries(output_df).series
            print("Post process applied.\n")
            
        output_df.to_csv(out_file, index=False)
        prediction_df.to_csv(out_src_file, index=False)
        print(f"Inference output file generated (This is not the final output), see: {output_folder}.\n")
    
    return output
