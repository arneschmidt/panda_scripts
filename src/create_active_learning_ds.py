import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

seed = 0

wsi_df = pd.read_csv('./artifacts/train.csv', index_col=False)
wsi_df = wsi_df.loc[wsi_df['data_provider'] == 'radboud']

total_samples = len(wsi_df)

test_samples = 1000
val_samples = 30

train_val, test = train_test_split(wsi_df, test_size=test_samples, random_state=seed)
train, val = train_test_split(train_val, test_size=val_samples, random_state=seed)

train['Partition'] = 'train'
val['Partition'] = 'val'
test['Partition'] = 'test'

print(val['gleason_score'])

new_wsi_df = pd.concat([train, val, test])
new_wsi_df = new_wsi_df.reset_index(inplace=False)\
    .drop(columns='index')
new_wsi_df.to_csv('./artifacts/train_active.csv')