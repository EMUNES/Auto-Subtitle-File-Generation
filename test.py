from encoder import ASSEncoder, SRTEncoder
import pandas as pd

df = pd.read_csv("./inference/test.csv")

encoder = ASSEncoder(df, "*eng").generate("test")

