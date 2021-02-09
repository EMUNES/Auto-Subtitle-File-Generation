"""
Parses the .csv file from inference output and generate subtitle file.
"""

import pandas as pd

from config import SSourceConfig as SSC


class Encoder(object):
    """
    Decode pd dataframe for subtitles.
    
    Args:
        df: DataFrame containing all events starts and endings.
        
    Attributes:
        start_series(List(float)): Start timestamps for events.  
        end_series(List(float)): End timestamps for events.
    """
    
    def __init__(self, df:pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.start_series = [f"{self._format_time(float(fl))}" for fl in self.df.start]
        self.end_series = [f"{self._format_time(float(fl))}" for fl in self.df.end]

    def _format_time(self, fl):
        """
        Float to string conversion function.
        """
        
        int_str_part, decimal_str_part = str(fl).split(".")
        int_part = int(int_str_part)
        decimal_str_part = decimal_str_part[:2]
        
        s = int_part % 60 # seconds
        m = (int_part // 60) % 60 # minutes
        h = int_part // 3600 # hours

        return f"{h}:{m}:{s}.{decimal_str_part}"


class ASSEncoder(Encoder):
    """
    Encode for .ass subtitle format.

    Args: 
        df(pd.DataFrame): The dataframe contain source for events.
        lang_style(str): The language style in ass standards.
        title(str): The title for script infor.
        x(str/int): X coordinate for subtitle.
        y(str/int): Y coordinate for subtitle.
        
    Properties:
        script_info: The script information for .ass subtitle file.
        v4_styles: V4 style setting for ssa subtitles.
        v4plus_styles: V4+ style setting for ass subtitles.
        events: Subtitle events. 
    """
    
    def __init__(self, df: pd.DataFrame, lang_style: str, title: str="ASG", x=384, y=288) -> None:
        super().__init__(df)
        self.lang_style = lang_style
        self.title = title
        self.x = x
        self.y = y
        
    def _iter2str(self, iter):
        """
        Transfer iterables to string.
        
        Args: 
            iter: Iterable object.
            tight(boolean): Whether to use blank space in delimiter.
            
        Returns:
            str_line: A single string line.
        """
        
        delimiter = ","
        list_source = [str(el) for el in iter]
        
        str_line = delimiter.join(list_source)
        
        return str_line
        
    def _dict2str(self, d: dict, parse=True):
        """
        Transfer Dict object to List object.
        
        Args: 
            dict: The Dictionary object.
            parse: Whether to parse dict series.
            
        Returns:
            str_lines: String lines.
        """
        
        str_lines = []
        
        for key in d.keys():
            str_value = self._iter2str(d[key]) if parse else str(d[key])
            str_line = f"{key}: {str_value}"
            str_lines.append(str_line)
            
        return str_lines
    
    def _format_time_presentation(self, str_time):
        """
        Perfect the format for fit ASS format.
        
        Args: 
            str_time(str): Represent time as x:x:x.xx
            
        Returns:
            formatted_str_time(str): Represent time as x:xx:xx.xx
        """
        
        i, f = str_time.split(".")
        
        h, m, s = i.split(":")
        
        m = ("0" + m) if len(m)<2 else m
        s = ("0" + s) if len(s)<2 else s
        f = ("0" + f) if len(f)<2 else f

        formatted_str_time = f"{h}:{m}:{s}.{f}"
        
        return formatted_str_time

    @property
    def script_info(self) -> list:
        info = SSC.headers
        
        if "Title" in info:
            info["Title"] = self.title
        if "PlayResX" in info:
            info["PlayResX"] = self.x
        if "PlayResY" in info:
            info["PlayResY"] = self.y

        return self._dict2str(info, parse=False)

    @property
    def v4_styles(self) -> list:
        
        return self._dict2str({
            "Format": SSC.v4_pairs.keys(),
            "Style": SSC.v4_pairs.values(),
        })
    
    @property 
    def v4plus_styles(self) -> list:
        
        return self._dict2str({
            "Format": SSC.v4plus_pairs.keys(),
            "Style": SSC.v4plus_pairs.values(),
        })
    
    @property
    def events(self) -> list:
        info = SSC.events_pairs

        if "Style" in info:
            info["Style"] = self.lang_style
            
        if "Start" in info and "End" in info:
            for (s, e) in zip(self.start_series, self.end_series):
                info["Start"] = self._format_time_presentation(s)
                info["End"] = self._format_time_presentation(e)
                
                yield self._dict2str({
                    "Format": info.keys(),
                    "Dialogue": info.values(),
                })
                
    def generate(self, file_name, target_dir="./results/", encoding="utf-8"):
        
        v4_styles = []
        try:
            v4_styles = self.v4_styles
        except:
            pass
        
        v4plus_styles = []
        try:
            v4plus_styles = self.v4plus_styles
        except:
            pass
        
        assert v4plus_styles or v4_styles, "No styles input for .ass files."
        
        path = target_dir + file_name 
        if (not ".ssa" in file_name) or (not ".ass" in file_name):
            path = path + (".ass" if v4plus_styles else ".ssa")
        
        with open(path, mode="w", encoding="utf-8") as f:
            for i in self.script_info:
                f.write(i)
                f.write("\n")
                
            f.write("\n")

            if v4plus_styles:
                f.write("[V4+ Styles]")
                f.write("\n")
                for i in v4plus_styles:
                    f.write(i)
                    f.write("\n")
                    
            f.write("\n")

            if v4_styles:
                f.write("[V4 Styles]")
                f.write("\n")
                for i in v4_styles:
                    f.write(i)
                    f.write("\n")
                    
            f.write("\n")
                    
            f.write("[Events]")
            f.write("\n")
            f.write(next(self.events)[0])
            for event in self.events:
                    f.write(event[1])
                    f.write("\n")
                    
            f.write("\n")
            

class SRTEncoder(Encoder):
    """
    Decode .srt subtitle files.
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        
    def _format_time_presentation(self, str_time):
        """
        Perfect the format of float numbers to fit SRT format.
        
        Args: 
            str_time(str): Represent time as x:x:x.xx
            
        Returns:
            formatted_str_time(str): Represent time as xx:xx:xx,xx0
        """
        
        i, f = str_time.split(".")
        
        h, m, s = i.split(":")
        
        h = ("0" + h) if len(h)<2 else h
        m = ("0" + m) if len(m)<2 else m
        s = ("0" + s) if len(s)<2 else s

        while len(f) < 3:
            f = f + "0"

        formatted_str_time = f"{h}:{m}:{s},{f}"
        
        return formatted_str_time

    @property
    def events(self):
        event_collections = []
        
        for (s, e) in zip(self.start_series, self.end_series):
            event_line = f"{self._format_time_presentation(s)} --> {self._format_time_presentation(e)}"
            event_collections.append(event_line)
        
        return event_collections
        
    def generate(self, file_name, target_dir="./results/", encoding="utf-8"):
        """
        API offering subtitle generation survice.
        
        Args:
            file_name(str): The file name for generating the final result.
            target_dir(str): Target folder holding the result.
            encoding(str): The encoding for the subtitle file.
        """
        
        path = target_dir + file_name            
        if not "srt" in file_name:
            path = path + ".srt"
            
        with open(path, mode="w", encoding=encoding) as f:
            for (idx, event) in enumerate(self.events):
                f.write(str(idx+1))
                f.write("\n")
                f.write(event)
                f.write("\n")
                f.write(SSC.content)
                f.write("\n")

                f.write("\n")
                