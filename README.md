# ASG - auto subtitle generation

This project focuses on how to automatically generate subtitle files from an input audio/video file, serving an easy end-to-end service.

## Tasks

### SED recognition

- use **Sound Event Detection** based on deep learning model (PANNs) to recognize **speech** from **streaming** audio files. Each detected speech clip will ouput like (onset, endset, speech class)

### aggregation

Aggregate final speech from the ouput. Generate corresponding subtitle file type for direct usuage

## Datasets

Make the dataset automatically.

## Utils

win10 + python environment + pytorch & pytorch ecosystem + vscode

## Challenges

This project contains many challenges.

### Speech recognition

Speech recognition is the core for the project. But this is not so much a bigger challege if not for perfect accuracy.

### Speech separation

Speech separation is extremely challenging in this project. As dialogues can happen real fast and just look like a consistent speech all together istead of different people talking. Solve this and the project will be a complete success.

### Data corruption

- As the datasets for training are automatically generated. It's possible that in subtitle files, sometimes clips that contains no speech will be tagged by the author.
- Differenct people tag the subtitles differently.

## Thanks

Thanks to following (models&pretrained weights&open source code):

- PANNs.
- Kaggle competitioner.

All credits to them! I'll be unable to come up with this project without their great idea and wonderful work.
