# Guide

## This folder contains all the row materials for generating your dataset

- A supported audio/video file and its corresponding subtitle file should be in the same sub folder.
- Rename your audio/video file to *VideoName-LanguageType* for better clearity in your dataset. For example: "ab-eng.ac3" can stand for the audio file of *American Beauty* in English.

## Notice

- video files are supported by *moviepy*. So "supported" means any format supported by Python Module Moviepy. Also, it may takes more time to extract audio than some professional tools.
- Audio files are supported by *librosa*. So "supported" means any audio format supported by Python Module Librosa. Also, for large video files that last hours could take a long for librosa to load and create clips for training.
- The subtitle file must be encoded in utf-8. If not, just open it with an editor and save it as utf-8. This will save you a lot of time and work.
