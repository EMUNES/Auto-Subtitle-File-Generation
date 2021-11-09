# ASFG - auto subtitle file generation

The Chinese version doc [here](https://blog.csdn.net/weixin_44145222/article/details/113820454).

This project focuses on how to automatically generate subtitle **files** from an input audio/video file containing all the timelines for dialogues and human speeches.

The project is in a primitive phase :sweat_smile: Further development will come when I find more time after The Lunar New Year.

how it seems :point_down:

![sample](.github/sample.gif)

(In case you don't know why I put the GIF here, the small 'xxx' overlapping are from the output subtitle file generated using this project which shows it konws when people talk. Yes, I've choose the good part :smirk:. Be patient to see when it's fully loaded :smile:)

There is still much to improve in real time cases. See *TODO.md* in the project folder for details.

## UPDATE

- Upload source scripts (jupyter notebook files) for reference/training/post_process and training/test data generation (*build-dataset.ipynb*). To run those scripts, move them from *jns* folder to the project root folder or just execute scripts currently under the root folder (they are the same). Now it's much easier to adjust codes and models, and it's so simple to just run *build-dataset.ipynb* to build your own dataset from source files, which include videos and subtitle files in *.ass* and *.srt* format. - 2021.11
- After updating the model with a much bigger dataset, the model inference result under most of the circumstances still can only parse sentences on second-wise, which lacks precision for practical usage. I need to review the network and I may choose another network for ms-precision. Those will take a long time so I may say this project is not abandoned but stagnated techniquely. 在使用更大的数据集完成模型的更新后，预测结果在大多数情况下仍然智能在秒单位基础上完成句子的分割，这样一来这个项目就缺乏实际运用所需要的精度。为此我需要重新检查更新神经网络或者使用一套崭新的神经网络来构建模型从而实现毫秒级的预测，这会花费我大量的时间。因此我想说这个项目没有被放弃但是由于技术原因停滞了。I will try to update it soon once a complete methodology found for this task that can be called successful. 当我能发现一套能够成功实现任务预期目标的完整方法体系时，我会尝试尽快更新项目。 - 2021.7
- **Speech to text** transition based on vosk api is implemented and now you can get subtitle files with recognized texts in it. However, the offline model does not perform very well. For more knowledge about offline STT model, check [vosk models](https://alphacephei.com/vosk/models) for details and *stt/vosk_api/README.md* local file for how to integrate offline model to this project. --- 2021.4

## A handy guide first

Currently, you need to build the python environment to run and use this program :sweat:

1. (Install python on your machine. Python version >= 3.8.)
2. Download the project code on your machine.
3. Download and Extract model under *models* folder. See README.md under *models* folder.
4. (Open your terminal or cmd under the project root folder and type`python install -r requirements.txt`. Install pytorch following its official website if it fails to download torch.). If you see error while running the program, try to run `python install -r requirements_full.txt` (install a full bundle with specified version of packages) and the you should be fine with this project.
5. Open your terminal or cmd under the project root folder and type `python run.py`. This will evoke an old-style script to accept the absolute path of your audio/video file, and offer a choice to name the output file.
6. Get your subtitle file under *results* folder under the project root folder.

Most audio/video file types are supported (Check python module *librosa* and *moviepy* for details) and you can get subtitles in **ass** or **srt** format.

## A better introduction

ASFG is based on deep learning technique for Sound Event Detection, which is a task to recognize different kinds of sounds in audio and make predictions containing the timeline for that sound event, using only weak label data. ASFG uses pytorch for deep learning, and contains many other parts and modules to make the project work.

The deep learning procedure is **open**, which means anyone can train and get your own model - from building the dataset to fine-tuning the model, ASFG has prepared middlewares and algorithms all you need to make deep learning work immediately. Only three procedures you mainly need to pay attention to:

- Get data source (And ASFG can build dataset for you automatically).
- Choose models and parts for your training pipeline (under *csrc* folder).
- Tune your params as you like.

Check [ASFG-train](https://github.com/EMUNES/ASFG-train) for details.

## Big thanks!!!

Great thanks to kaggle user [Hidehise Arai](https://www.kaggle.com/hidehisaarai1213) from whom I learn to build my baseline for SED, which I use heavenly for deep learning in this project too.

The pretrained model I use for this project is from: [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211).

The STT (speech to task) task is implemented using [vosk-api](https://github.com/alphacep/vosk-api).

All credits to them :thumbsup:
