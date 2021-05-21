"""Process the SED inference result.

The inference results should contain everything we need to generate subtitle files.
But we still need some process to make the results suitable for subtitle encoder.
"""


import pandas as pd

from config import PostProcessConfig as PPC


class SpeechNode(object):
    def __init__(self, cat:str="speech", onset:float=0, offset:float=0, mac:float=0, mec:float=0,) -> None:
        self.cat = cat
        self.onset = onset
        self.offset = offset
        self.mac = mac
        self.mec = mec

        self.next = None
        
        
class SpeechSeries(object):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.columns = df.columns
        self._Node = SpeechNode
        self._head_node = None
        self._ori_len = len(df)
        assert len(self.df.columns)==5, "Numbers of features in your inference output csv file is not right. \
                                        There should be 5 columns for post process."
        self._build_linked_list()

    def _build_linked_list(self):
        pre_node = None

        for s in self.df.itertuples():
            node = self._Node(
                s.speech_recognition,
                s.start + PPC.global_bias, # global shift
                s.end + PPC.global_bias, # global shift
                s.max_confidence,
                s.mean_confidence,
            )

            if self._head_node == None:
                self._head_node = node
                pre_node = self._head_node
                continue
            
            pre_node.next = node

            pre_node = node

    def _merge_node(self, n1, n2):
        assert n2.onset >= n1.offset, "Node input order reversed."
        n1.offset = n2.offset
        n1.next = n2.next 
        return n1

    def _break_node(self, n, timestamp: float):
        new_offset = timestamp - PPC.break_period/2
        new_onset = timestamp + PPC.break_period/2

        new_node = self._Node(
            n.cat,
            new_onset,
            n.offset,
            n.max_confidence,
            n.mean_confidence
        )

        n.offset = new_offset

        new_node.next = n.next
        n.next = new_node

    def _post_process(self, n):
        """
        Post process based on linked list.
        """
        speech = n

        while speech.next != None:
            # Increase the offset properly.
            cur_time_gap = speech.offset - speech.onset
            bet_time_gap = speech.next.onset - speech.offset

            # If the gap between speeches is loose, add proper delay at the end.
            if bet_time_gap >= PPC.loose_dialogue_threshold:
                speech.offset += PPC.loose_dialogue_delay
            # If the gap between speeches is so tight, merge them together. 
            elif (bet_time_gap <= PPC.standard_dialogue_break/3) or bet_time_gap==0.0:
                speech = self._merge_node(speech, speech.next)
                continue
            # If the dialogue are not loose but also not tight enough to merge them together, make the break standard.
            elif (bet_time_gap > PPC.standard_dialogue_break/3) and (bet_time_gap < PPC.standard_dialogue_break):
                residue = PPC.standard_dialogue_break - bet_time_gap
                speech.offset -= residue / 2
                speech.next.onset += residue / 2

            # TODO: Need hooks to implement break point.
            if cur_time_gap > PPC.max_sigle_speech_length:
                pass
            
            speech = speech.next

        speech.offset += PPC.loose_dialogue_delay

    @property
    def series(self) -> pd.DataFrame:
        """Series of the sp
        """
        speech = self._head_node
        assert speech != None, "Nothing valid from inference output. Please check /inf/output/yourOutputFile."
        
        self._post_process(speech)
        
        columns = self.df.columns
        df = pd.DataFrame(index=None, columns=columns)
        
        while speech != None:
            s = [
                speech.cat,
                speech.onset,
                speech.offset,
                speech.mac,
                speech.mec
                ]
            d = dict(zip(self.df.columns, s))
            
            df = df.append(d, ignore_index=True)

            speech = speech.next

        return df
    