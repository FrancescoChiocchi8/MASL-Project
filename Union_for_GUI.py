import pandas as pd


df1 = pd.read_csv('output/MicrobiotaOutput_counts_12.csv')
df2 = pd.read_csv('output/LumeOutput_counts_15.csv')
df3 = pd.read_csv('output/NervousOutput_counts_15.csv')

merged_df = pd.merge(df1, df2, on='tick', suffixes=('_microbiota', '_lumen'))
merged_df = pd.merge(merged_df, df3, on='tick', suffixes=('', '_cnn'))

merged_df.to_csv('output/union_for_gui/union.csv', index=False)
