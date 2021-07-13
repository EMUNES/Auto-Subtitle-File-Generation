"""
Out of the box pipeline to generator the final result.
"""

import pandas as pd

from inf.inferer import get_inference
from encoder import ASSEncoder, SRTEncoder
from config import INFERENCE_PARAMS_PATH, INF_OUTPUT_FOLDER_ABS

def generator(targ, fname, sub_format: str="ass", lang: str="eng", post=True, output_folder=INF_OUTPUT_FOLDER_ABS):
    get_inference(targ_file_path=targ,
                  params_path=INFERENCE_PARAMS_PATH,
                  fname=fname,
                  lang=lang,
                  post_process=post,
                  output_folder=output_folder)

    df = pd.read_csv(f"{output_folder}/{fname}.csv")

    if ("ass" in fname) or (sub_format=="ass") or (sub_format==".ass"):
        encoder = ASSEncoder(df, "*eng")
    
    if ("srt" in fname) or (sub_format=="srt") or (sub_format==".srt"):
        encoder = SRTEncoder(df)
        
    print(f"Calling encoder to generate the final output...\n") 
    encoder.generate(fname)
    print(f"All procedures done! Subtitle file generated.\n")

