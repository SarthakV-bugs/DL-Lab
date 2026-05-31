import pandas as pd

lines = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
print(len(lines))
print(lines['source'].value_counts())
print(lines['source'].unique()) #['ted' 'indic2012' 'tides']
print(lines['english_sentence'].isna().any(axis=1))


