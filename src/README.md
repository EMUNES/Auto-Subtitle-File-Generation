# Guide

## This folder contains all the row materials for generating your dataset

- A supported audio/video file and its corresponding subtitle file should be in the same folder.
- Rename your audio/video file to *VideoName-LanguageType* for ASFG to build dataset for you. For example: "ab-eng.ac3" can stand for the audio file of *American Beauty* in English. You don't need to name your subtitle files with language type.

## Notice

- video files are supported by *moviepy*. So "supported" means any format supported by Python Module Moviepy.
- Audio files are supported by *librosa*. So "supported" means any audio format supported by Python Module Librosa.
- The subtitle file must be encoded in utf-8. If not, open it with an editor and save it as utf-8.
