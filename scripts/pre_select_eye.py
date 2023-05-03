import pickle
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

ROWS_PER_FRAME = 543  # number of landmarks per frame
LEFT_HAND_INDICES = [468, 469, 470, 471, 472, 473, 474, 475, 476,
                     477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488]
RIGHT_HAND_INDICES = [522, 523, 524, 525, 526, 527, 528, 529, 530,
                      531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
INNER_LIP_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
REYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE_INDICES = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
NOSE_INDICES = [1, 2, 98, 327]

SELECT_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES + \
    LEFT_HAND_INDICES + RIGHT_HAND_INDICES + REYE_INDICES + LEYE_INDICES


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def map_func(path):
    _features = load_relevant_data_subset(path)
    _features = _features[:, SELECT_INDICES]
    return _features


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'save to {path}')


def main():
    save_dir = '../input/preprocessed_data/'
    df = pd.read_csv(f"{save_dir}/train_df_with_folds.csv")

    num_features = len(SELECT_INDICES)
    print("num_features", num_features)

    pool = Pool(processes=cpu_count())
    features = list(tqdm(pool.imap(map_func, df['path'].values), total=len(df)))

    save_path = f'{save_dir}/pre_select_features_eye.pkl'
    save_pickle(features, save_path)


if __name__ == "__main__":
    main()
