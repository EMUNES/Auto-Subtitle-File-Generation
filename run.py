"""
Simple script to take input and run on CLI.
"""

from generator import generator

if __name__ == '__main__':
    print("Welcome to ASG! Contact author: hzb000113@qq.com\n\
欢迎尝试！联系作者 - hzb000113@qq.com\n")
    
    target_file_name = input("The absolute path for your audio/video:\n\
音视频文件的绝对路径:\n")
    
    output_file_name = input("\nThe file name and type you want for the output file(default 'current.ass'): \n\
Valid input such as: 'sample.ass', 'sample', 'sample.srt'. No quote mark needed.\n\
你希望的输出文件名称和格式（默认是current.ass）\n\
合法的文件名如：'sample.ass', 'sample', 'sample.srt'，不需要引号:\n")
    
    print("\nIf the CLI stuck at Loading procudure for a long time. Try press 'Enter', and if no respond then wait longer.")
    print("如果控制台卡在读取文件阶段长时间不动，尝试按一下'Enter'键，若仍无反应则继续等待加载完成。")
    print("If you have nvidia GPU and CUDA on your machine, it will take less to run. (default device cuda0. Otherwise use CPU)")
    print("如果你的设备上拥有nvidia CPU和CUDA环境，花费的时间会更少。（默认使用cuda0，然后使用CPU）")
    generator(targ=target_file_name, fname=output_file_name)
    print("The output subtitle file is under result folder under this project root folder. Thanks for using!\n\
输出最终字幕文件将在项目根目录的result文件夹下，感谢您的使用。\n")