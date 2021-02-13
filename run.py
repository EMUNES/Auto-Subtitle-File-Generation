"""
Accept input file path from CLI and call methods for output.
"""

from generator import generator

if __name__ == '__main__':
    print("Welcome to ASG! Contact author: hzb000113@qq.com\n \
          欢迎尝试！联系作者 - hzb000113@qq.com\n\n")
    target_file_name = input("The absolute path for your audio/video:\n \
                              音视频文件的绝对路径:\n")
    output_file_name = input("The file name and type you want for the output file(default 'current.ass'): \n \
                              Valid input such as: 'sample.ass', 'sample', 'sample.srt'. Not quote mark needed.\n \
                              你希望的输出文件名称和格式（默认是current.ass）\
                              合法的文件名如：'sample.ass', 'sample', 'sample.srt'，不需要引号")
    
    print("The output subtitle file is under result folder under this project root folder.\
        输出最终字幕文件在项目根目录的result文件夹下，感谢您的使用。")
    
    generator(targ=target_file_name, fname=output_file_name)
    