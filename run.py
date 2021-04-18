"""Take user input and run the program.

This should be the Only entry for the program. This project
uses absolute import system, so moving this script to other place
will cause System Error. Just go in the project root folder and run
this script
"""

from generator import generator
from config import STTInferenceConfig

if __name__ == '__main__':
    
        
    print("Welcome to ASFG! Contact author: hzb000113@qq.com\n\
欢迎尝试！联系作者 - hzb000113@qq.com\n")
    
    # Choose the target file path to generate subtitles.
    target_file_name = input("The absolute path for your audio/video:\n\
音视频文件的绝对路径:\n")
    
    # Choose your output file name.
    output_file_name = input("\nDeclare the file name and file type for output(default 'current.ass').\n\
Valid input such as: 'sample.ass', 'sample', 'sample.srt'. No quote mark needed:\n\
你希望的输出文件名称和格式（默认是current.ass）。\n\
合法的文件名如：'sample.ass', 'sample', 'sample.srt'，不需要引号:\n")

    # Choose the lanauge.    
    choices = {}  
    for (i, (lang, lang_name)) in enumerate(STTInferenceConfig().iter_avl_langs()):
        print(f"{i} - {lang_name}")
        choices[str(i)] = lang
    language_idx = input("Choose the lanauage index of your source:\n选择资源语言的对应序号:")
    source_language = choices[language_idx]
    
    
    print("\nPress 'Enter' if the cli gets stuck for a long time during the procedure.")
    print("如果控制台卡在某个地方长期不动，尝试按一下Enter键。\n")
    
    print("If you have nvidia GPU and CUDA on your machine, this will take less time to run. (Default device cuda0. Otherwise use CPU)")
    print("如果你的设备上拥有nvidia CPU和CUDA环境，花费的时间会更少。（默认使用cuda0，然后使用CPU）\n")
    
    generator(targ=target_file_name, fname=output_file_name, lang=source_language)
    
    print("The output subtitle file is under result folder in this project's root folder. Thanks for your time!\n\
输出最终字幕文件在项目根目录的result文件夹下，感谢您的使用。\n")