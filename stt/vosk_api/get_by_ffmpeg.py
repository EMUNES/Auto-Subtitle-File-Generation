"""Get voice recognize result using ffmpeg.
"""

from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import subprocess
import json

from config import TEMP_FOLDER_ABS

def ffmpeg_sst(fname:str, lang:str="eng", model_spec:str="model"):
    
    SetLogLevel(-1)
    
    # This file must be running under current folder to get model settings!
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    
    # Opne with absolute path to avoid system error.
    fpath = TEMP_FOLDER_ABS + '/' + fname
        
    if not os.path.exists("model-eng"):
        print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    
    sample_rate=16000
    # Choose the model based on given language or specified model name.
    model = Model(f"model-{lang}") if model_spec=="model" else Model(model_spec)
    rec = KaldiRecognizer(model, sample_rate)

    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                fpath,
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)

    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
        # else:
        #     print(rec.PartialResult())

    # print(rec.FinalResult())
    # return rec.FinalResult
    res = json.loads(rec.FinalResult())
    print (f"All text: {res['text']}\n")
    
    return res['text']
