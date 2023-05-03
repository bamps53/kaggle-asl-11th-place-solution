import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import json


def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    save_dir = '../input/preprocessed_data'
    os.makedirs(save_dir, exist_ok=True)

    input_dir = '../input/asl-signs'

    df = pd.read_csv(f'{input_dir}/train.csv')
    df['fold'] = -1

    label_map = load_json(f'{input_dir}/sign_to_prediction_index_map.json')
    df['label'] = df['sign'].map(label_map)

    df['path'] = df['path'].map(lambda x: f"{input_dir}/{x}")

    kf = GroupKFold(n_splits=5)
    for fold, (_, val_index) in enumerate(kf.split(df, groups=df['participant_id'])):
        df.loc[val_index, 'fold'] = fold

    assert df.groupby('participant_id')['fold'].nunique().max() == 1

    df = df.drop(columns=['sequence_id'])

    save_path = f"{save_dir}/train_df_with_folds.csv"
    print('save dataframe to ', save_path)
    df.to_csv(save_path, index=False)
