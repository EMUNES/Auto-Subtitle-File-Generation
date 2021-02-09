# Project development diary

## Reference

[1] [PANNS](https://arxiv.org/abs/1912.10211)
[2] [kaggle - Introduction to sound event detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)
[3] [some code source](https://github.com/koukyo1994/kaggle-birdcall-6th-place) from kaggle competitioner *hidehisaarai1213's* solution

## Known

- Trainig clips length should be set between 1-5 seconds. Around 3 sec is works the best.
- Decrease the *inference clip length* will make better recognitions with more **accuracy**. Trying: 0.5-3secs. For using 5s long clip, the dialogue can be all recognized but the break and accuracy still need improvement. The improvement is notible but not enough. **Dialogue break** improves a lot! setting inference clips around 1s!!!
- Add basic augmentaion reduce the overfitting **hugely** but increase a hell of long time for training.

## Dev

### 2.3

- Add data basic augmentaion. Training time increases a lot.
- Using fp16 for training. Hope to reduce overfitting and speed up the training process. install apex using: pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+<https://github.com/NVIDIA/apex> --- FAILED

### 2.4

- Using improved panns loss.

### 2.8

- Inference with 2s, accuracy improves, especially for determing when the dialogue will be end.
- Inference with 1s, dialogue break finally works much better than before ! Aroud 1s is great for dialogue recognition. However, recognizing animal voice as human speech still exists. However, as the inference length is decreased to a very small length. The chance of misrecognize certain clips occurs more.
- Test threshold for 0.6 and 0.9. The former changes little while the last make the recogniztion worth for speech detection accuracy.
