import pandas as pd

infile = "snid.txt"

df = pd.read_csv(infile, header=None, delimiter=r"\s+")
print(df.keys())
df_new = df[[0, 1]]
# print(df_new)
print(df_new)
df_new.to_csv("snidnew.txt", sep=" ", index=False)
