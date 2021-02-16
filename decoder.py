"""
Decode subtitle files for materials bulding the dataset.
"""

from pathlib import Path


class ASSDecoder(object):
    """
    Decode .ass(.ssa) subtitle files
    
    Args: 
        file_path: The file path of the subtitle file.
        encoding: The encoding of the subtitle file.
        
    Attributes:
        file_type: Subtitle file format.
        
    Properties:

    """
    
    file_type = "ass"

    def __init__(self, file_path, encoding="utf-8"):
        assert isinstance(file_path, str) or isinstance(file_path, Path), "Invalid file path, only 'str' and Pathlib.Path' supported."
        self.file_path = file_path
        self.encoding = encoding
        self.flag = 0
        self.tags =  self._tags()
        assert len(self.tags["events"])==1, "There should only be one [Events] tag in sub file."
        assert len(self.tags.keys()) == 3, "Your sub file should only and must contain following components: headers lines, [...Styles], [Events]."

    def _tags(self):

        # iterate the whole file and return tags of headers/styles/events
        with open(self.file_path, encoding=self.encoding) as f:
            tags = dict()
            tags["headers"] = 0
            tags["events"] = list()
            tags["styles"] = list()
            for i, line in enumerate(f.readlines()):
                if "events" in line.lower() and line.startswith("["):
                    tags["events"].append(i)
                if "styles" in line.lower() and line.startswith("["):
                    tags["styles"].append(i)

        return tags
    
    def _decode_time(self, str_time):
        """Decode time from src to float(.2f), which stands for seconds.
        
        Returns:
            float_time: Seconds of the corresponding time.
            
        Args:
            str_time: String format for time object.
            
        Properties:
            events: Parsed events output.
            time_series: series of begining and ending timestamps.
        """
        
        tail = float(str_time.split(".")[-1]) * 1e-2
        h, m, s = str_time.split(".")[0].split(":")
        float_time = int(h)*3600 + int(m)*60 + int(s) + tail
        
        return float_time

    @property
    def events(self):
        with open(self.file_path, encoding=self.encoding) as f:
            assert len(self.tags["events"])==1, "There should only be one [Events] tag in sub file."

            events = []
            events_info = {
                "tag": "",
                "header": "",
                "features": ""
            }

            # get well formatted sub events list
            events = f.readlines()[self.tags["events"][0]:]
            events = [event.lstrip().rstrip() for event in events if event != "\n"]

            # collect events information
            events_tag = events[0]
            events_header = events[1]
            features = events_header.split(":")[1].split(",")
            assert len(features) == 10, "Events feature number does not fit the standard, please check your sub file."

            events_info["tag"] = events_tag
            events_info["header"] = events_header
            events_info["features"] = features

        return events[2:], events_info
    
    @property
    def time_series(self):
        """Return two timestamp lists, in a list each element stands for the beginning or the end of a dialogue.

        Returns:
            on_ts: List of all event onsets.
            off_ts: List of all event offsets.
        """
        
        events, _ = self.events
        on_ts = list()
        off_ts = list()
        
        assert events is not None and events is not [], "Events empty, can not generate time series"
            
        for event in events:
            # remove duplicated timeseries, for multilanguage sub file we directly ingore them
            # we assume all languages" sub file are in the same time series
            
            onset = event.split(",")[1]
            offset = event.split(",")[2]
            
            if onset in on_ts or offset in off_ts:
                break

            on_ts.append(self._decode_time(onset))
            off_ts.append(self._decode_time(offset))
            
        assert len(on_ts) >= 1, "No valid time"
        assert len(on_ts) == len(off_ts), "Unable to match onset with offset for Dialogues, please check your sub file"
        
        return on_ts, off_ts
             
            
class SRTDecoder(object):
    """
    Decode .srt format subtitle files.
    
    Args:
        file_path(str/path): The subtitle file path for decoding.
        encoding(str): The encoding of the subtitle file.
    
    Attributes:
        file_type: The subtitle file type.
        
    Properties:
        time_series -> on_ts, off_ts: Timestamp collections of the beginning and ending of each event. 
    """
    
    file_type = "srt"
    
    def __init__(self, file_path, encoding="utf-8") -> None:
        self.file_path = file_path
        self.encoding = encoding
        
    def _decode_time(self, str_time):
        """Decode time from src to float(.2f), which stands for seconds.
        
        Returns:
            float_time: Seconds of the corresponding time.
            
        Args:
            str_time: String format for time object.
            
        Properties:
            events: Parsed events output.
            time_series: series of begining and ending timestamps.
        """
                
        tail = float(str_time.split(",")[-1]) * 1e-3
        h, m, s = str_time.split(",")[0].split(":")
        float_time = int(h)*3600 + int(m)*60 + int(s) + tail
        
        return float_time
        
    @property
    def time_series(self):
        on_ts = []
        off_ts = []
        
        with open(self.file_path, mode="r", encoding=self.encoding) as f:
            for line in f.readlines():
                if "-->" in line:
                    onset = line.split("-")[0].lstrip().rstrip()
                    offset = line.split(">")[-1].lstrip().rstrip()
                    onset = self._decode_time(onset)
                    offset = self._decode_time(offset)
                    if onset:
                        on_ts.append(onset)
                    if offset: 
                        off_ts.append(offset)
        
        assert len(on_ts)==len(off_ts), "Mismatch for timestamp series."

        return on_ts, off_ts
