"""Get voice recognize result using ffmpeg.
"""

from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import subprocess

SetLogLevel(0)

def ffmpeg_sst(fname):
    sample_rate=16000
    model = Model("model")
    rec = KaldiRecognizer(model, sample_rate)

    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                fname,
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)

    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print(rec.Result())
        else:
            print(rec.PartialResult())

    print(rec.FinalResult())
    return rec.FinalResult
