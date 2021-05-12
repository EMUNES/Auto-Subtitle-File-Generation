"""Decode subtitle files for materials bulding the dataset.
"""

from pathlib import Path
from config import DecoderConfig as DC


class Decoder(object):
    """Decoder containing public arguments and methods for subtitle files.
    """
    
    def __init__(self, file_path, encoding, trim) -> None:
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.trim = trim
        
    def _trim_events(self, onsets, offsets):
        """Trim the events.

        Sometimes the subtitle will continue after cutting out a clip (2s) but that time 
        is very short (like 0.1s) that nobody (let alone machines) can recognize human 
        speech in it. To avoid such bias causing a lot of mislabeled, unrecognizable clips
        in the dataset. we should trim the event onsets and offsets to keep the obvious, 
        recognizable speech clips only.
        
        Args:
            onsets: [events onsets].
            offsets: [events offsets].
            
        Returns: 
            onsets: [trimmed events onsets].
            offsets: [trimmed events offsets].
        """
        
        def handle_offset(offset):
            if (offset % 2.) < DC.trimming_end:
                offset = float(int(offset))
            
            return offset
                
        def handle_onset(onset):
            if on_end := ((onset / 2. + 1) * 2 - onset) < DC.trimming_start:
                onset = float(on_end)
            
            return onset
            
        # Trim offsets.
        offsets = list(map(handle_offset, offsets))
        
        # Trim onsets.
        onsets = list(map(handle_onset, onsets))
        
        return onsets, offsets


class ASSDecoder(Decoder):
    """
    Decode .ass(.ssa) subtitle files
    
    Args: 
        file_path: The file path of the subtitle file.
        encoding: The encoding of the subtitle file.
        
    Attributes:
        file_type: Subtitle file format.
        
    Properties:
        time_series: Containing all events timestamps (s).
    """
    
    file_type = "ass"

    def __init__(self, file_path, encoding="utf-8", trim=True):
        assert isinstance(file_path, str) or isinstance(file_path, Path), "Invalid file path, only 'str' and Pathlib.Path' supported."
        super().__init__(file_path, encoding, trim)
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
            
        on_ts, off_ts = self._trim_events(on_ts, off_ts) if self.trim else (on_ts, off_ts)
            
        assert len(on_ts) == len(off_ts), "Unable to match onset with offset for Dialogues, please check your sub file"
        
        return on_ts, off_ts
             
            
class SRTDecoder(Decoder):
    """Decode .srt format subtitle files.
    
    Args:
        file_path(str/path): The subtitle file path for decoding.
        encoding(str): The encoding of the subtitle file.
    
    Attributes:
        file_type: The subtitle file type.
        
    Properties:
        time_series -> on_ts, off_ts: Timestamp collections of the beginning and ending of each event. 
    """
    
    file_type = "srt"
    
    def __init__(self, file_path, encoding="utf-8", trim=True) -> None:
        assert isinstance(file_path, str) or isinstance(file_path, Path), "Invalid file path, only 'str' and Pathlib.Path' supported."
        super().__init__(file_path, encoding, trim)

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
        """Return event timestamps.
        """
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
                        
        on_ts, off_ts = self._trim_events(on_ts, off_ts) if self.trim else (on_ts, off_ts)
        
        assert len(on_ts)==len(off_ts), "Mismatch for timestamp series."

        return on_ts, off_ts
