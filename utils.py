"""
Preprocess some input:

- If it"s video, extract audio from it.
- If it"s audio, do nothing.

To run the audio extraction process normally, you should name the video right
in the same standard as the audio and make sure it supports the format the 
Python Module Moviepy supports.

Check "https://pypi.org/project/moviepy/" for formats Moviepy supports.
"""

from pathlib import Path
import os
import time
import mimetypes
import os

import librosa
import moviepy.editor as mp

def mono_load(path, sr=16000, mono=True):
    """
    A custonmized librosa loading process emphasizing mono channel.
    
    Args:
        path: Audio file path.
        sr: Sample rate for librosa loading.
        mono: Indicate mono channel for the output.
        
    Returns: 
        y: Librosa loading output.
        c: The number of channels for the original audio file. 
    """
    
    start = time.time()
    
    print(f"Loading file: {path}")
    y, c = librosa.load(path, sr=sr, mono=mono)
    
    end = time.time()
    
    print(f"Loading completed. Cost {(end-start):.2f}s\n")
    
    return y, c 
    
def vb(pre, any, v):
    """
    Verbose print.
    """
    if v:
        print(f"{pre} {repr(any)}")
        
def check_type(file_path):
    """
    Check whether it"s a audio or video straitforwardly.
    
    Args:
        file_path: The path of the file to check type.
        
    Returns:
        is_video: Whether the file is a video or an audio.
    """
    
    assert (Path(file_path).exists()) and (Path(file_path).is_file()), "Your input file path is not valid or the file doesn't exist."
    
    is_video = False
    
    mimetypes.init()
    
    mimestart = mimetypes.guess_type(str(file_path))[0]
    
    if mimestart: # If the metadata can't be pared, it's mostly because the file is an audio file.
        try:
            mimestart = mimestart.split("/")[0]
        except RuntimeError as e:
            print(e)
            print("Unrecognizable file type. Is the file format valid? (Using mimetypes)\n")    
        
        assert mimestart=="video" or mimestart=="audio", "Input file format unrecognizable as video or audio (using mimetypes).\n"
        
        if mimestart == "video": is_video = True 

    return is_video

def extract_audio(file_path, format: str="wav"):
    """
    Extract audio from video.
    
    Args:
        file_path: File path for the video clip.    
        
    Returns:
        mv_audio_file: Audio file path extracted.
    """
    
    print(f"Extracting audio from {file_path}")
    
    mv = mp.VideoFileClip(file_path)
    assert mv!=None, "Unable to extract any information from the video clip."
    
    mv_name = str(file_path).split("/")[-1].split(".")[0]
    mv_audio_file = Path(file_path).parent / f"{mv_name}.{format}"
    
    # A potential error for moviepy to resolve system Path could trigger AttributeError.
    # We catch the error and then use string for movie py to resolve the file path.
    # This error occurs on Windows.
    try:
        mv.audio.write_audiofile(mv_audio_file)
    except AttributeError:
        print("\nNote: Moviepy failed to resolve your video path. Currently Use Path as string for moviepy to work.\n")
        mv.audio.write_audiofile(str(mv_audio_file))
     
    print(f"Extraction Successful! Writing {Path(mv_audio_file).stat().st_size} in {mv_audio_file}.")
    
    return mv_audio_file

def count_class(path):
    """
    Count how many instances of 0 and 1 under the folder path.
    
    Args:
        file_path: The path of the folder containing the class instances.
        
    Returns:
        zeros: The number of instances in class 0 (No human speaking).
        ones: The numger of instances in class 0 (Human speaking).
        total: total instances.
    """
    
    zeros = 0
    ones = 0
    total = 0
    
    for file in os.listdir(path):
        c = str(file).split(".")[0][-1]
        ones += int(c)
        total += 1
    zeros = total - ones
    
    print(f"\nLabel 1 instances: {ones}")
    print(f"Label 0 instances: {zeros}")
    print(f"Total clips: {total}\n")
    
    return zeros, ones, total

def get_duration(audio_file_path, y=None, sr=None):
    """
    Check the consistency between audio header metadata and audio waveform.
    
    Args:
        audio_file_path: The audio file path.
        y: Audio time series.
        sr: Audio sample rate.
        
    Returns (one of):
        waveform_duration: Audio duration according to the audio waveform.
        header_duration: Audio duration accrording to audio metadata.
    """
    
    header_duration = librosa.get_duration(filename=audio_file_path)
    
    if sr:
        wavform_duration = librosa.get_duration(y=y, sr=sr)
        
        if header_duration == wavform_duration:
            print("Audio file consistency ensured.")
        else:
            print("There is inconsistency between the audio waveform and header metadata." \
                "This could be ignored.")
            
        return wavform_duration
    
    return header_duration
    