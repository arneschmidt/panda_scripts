import pandas as pd
import numpy as np


number_dummy_wsis = 20
train_df_path = './artifacts/train_active.csv'
out_dir = './artifacts/train_active_20.csv'

train_df = pd.read_csv(train_df_path).drop(columns='Unnamed: 0')

train_wsi_df_indices = np.where(train_df['Partition'] == 'train')[0]
random_indices = np.random.choice(train_wsi_df_indices, size=number_dummy_wsis, replace=False)

new_wsi_df = train_df.iloc[random_indices]
val_test_wsi_df = train_df.loc[np.logical_or(train_df['Partition']=='val', train_df['Partition']=='test')]
new_wsi_df = pd.concat([new_wsi_df, val_test_wsi_df]).reset_index().drop(columns='index')

new_wsi_df.to_csv(out_dir, index=False)







