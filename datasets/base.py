import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import gc
import pickle
import random


def normalize_coords(x, normalize_mode):
    if np.isnan(x).all():
        return x
    if normalize_mode == 'mean_std':
        x = x - x[~np.isnan(x)].mean(0, keepdims=True)  # noramlisation to common mean
        x = x / x[~np.isnan(x)].std(0, keepdims=True)
    elif normalize_mode == 'mean_std_frame0':
        x = x - x[0][~np.isnan(x[0])].mean(0, keepdims=True)  # noramlize by frame0 mean/std
        x = x / x[0][~np.isnan(x[0])].std(0, keepdims=True)
    elif normalize_mode == 'none':
        pass
    else:
        raise NotImplementedError()
    return x


def fillna(x, fillna_mode):
    if fillna_mode == 'zero':
        x[np.isnan(x)] = 0
    elif fillna_mode == 'mean':
        avg = np.nanmean(x, axis=0)[None, :, :]
        # nanを0次元目の平均で埋める
        x = np.where(np.isnan(x), avg, x)
        # 0次元目が全てnanだったときように
        x[x.isnan(x)] = 0
    elif fillna_mode == 'none':
        pass
    else:
        raise NotImplementedError()
    return x


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def resize_array(arr, length):
    old_length = arr.shape[0]
    new_shape = (length,) + arr.shape[1:]

    x = np.linspace(0, old_length-1, old_length)
    x_new = np.linspace(0, old_length-1, length)

    resized_arr = np.zeros(new_shape)
    for i in range(arr.shape[1]):
        for j in range(arr.shape[2]):
            y = arr[:, i, j]
            y_new = np.interp(x_new, x, y)
            resized_arr[:, i, j] = y_new

    return resized_arr


def random_rotation_matrix(angle):
    theta = np.radians(np.random.uniform(-angle, angle))
    phi = np.radians(np.random.uniform(-angle, angle))
    psi = np.radians(np.random.uniform(-angle, angle))

    # 回転行列を計算
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def random_3d_rotate(vertices, angle):
    R = random_rotation_matrix(angle)
    rotated_vertices = np.dot(vertices, R)
    return rotated_vertices


def shift_features(x, value):
    if value > 0:
        return np.pad(x, ((0, value), (0, 0), (0, 0)), mode='edge')[value:, :]
    else:
        return np.pad(x, ((-value, 0), (0, 0), (0, 0)), mode='edge')[:value, :]


def preprocess(features):
    features = normalize_coords(features, normalize_mode='mean_std')
    features = fillna(features, 'zero').astype(np.float32)
    return features


class TrainDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode

        df = pd.read_csv(cfg.df_path)
        if mode == "train":
            index = df.fold != fold
        elif mode == 'valid':  # 'valid
            index = df.fold == fold
        else:
            raise NotImplementedError()
        df = df.loc[index]
        self.indices = df.index
        self.df = df.reset_index(drop=True)
        self.max_len = cfg.max_len

        self.features = load_pickle(cfg.feature_path)
        print('num features before:', len(self.features))
        self.features = [self.features[self.indices[idx]] for idx in range(len(self.df))]
        print('num features after:', len(self.features))
        gc.collect()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = {}

        # _features = self.features[self.indices[idx]]
        _features = self.features[idx]

        _features = preprocess(_features)

        if self.mode == 'train' and random.uniform(0, 1) < self.cfg.resize_rate:
            length = len(_features)
            new_length = random.randint(int(length * (1.0-self.cfg.resize_range)),
                                        int(length * (1.0+self.cfg.resize_range)))
            if new_length > 0:
                _features = resize_array(_features, new_length)
        if len(_features) > self.cfg.max_len:
            _features = resize_array(_features, self.cfg.max_len)

        _is_null = _features[..., [0]] == 0

        if self.mode == 'train':
            if self.cfg.random_flip and random.uniform(0, 1) < 0.5:
                _features[..., 0] *= -1
                outer_lip = _features[:, :20][:, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11]]
                inner_lip = _features[:, 20:40][:, [10, 9, 8, 7, 6, 5,
                                                    4, 3, 2, 1, 0, 19, 18, 17, 16, 15, 14, 13, 12, 11]]
                left = _features[:, 40:61]
                right = _features[:, 61:82]
                reye = _features[:, 82:98]
                leye = _features[:, 98:114]

                left, right = right, left
                reye, leye = leye, reye
                _features = np.concatenate([
                    outer_lip,
                    inner_lip,
                    left,
                    right,
                    reye,
                    leye], axis=1)

            if self.cfg.noise_rate > 0.0 and random.uniform(0, 1) < 0.5:
                noise = 1 + np.random.normal(scale=self.cfg.noise_rate, size=_features.shape)
                _features *= noise

            if self.cfg.angle_range > 0.0 and random.uniform(0, 1) < 0.5:
                _features = random_3d_rotate(_features, self.cfg.angle_range)

            if self.cfg.scale_range > 0.0 and random.uniform(0, 1) < 0.5:
                scale = 1 + random.uniform(-self.cfg.scale_range, self.cfg.scale_range)
                _features *= scale

            if self.cfg.shift_range > 0.0 and random.uniform(0, 1) < 0.5:
                shift = np.abs(_features).max(0, keepdims=True).max(1, keepdims=True) * \
                    np.random.uniform(-self.cfg.shift_range, self.cfg.shift_range, size=(1, 1, 3))
                _features += shift

        if self.cfg.motion_features:
            diff_prev_features = _features - shift_features(_features, -1)
            diff_next_features = shift_features(_features, 1) - _features
            velocity_features = (diff_prev_features + diff_next_features) / 2
            motion_features = np.concatenate([diff_prev_features, diff_next_features, velocity_features], axis=2)
            motion_features = fillna(motion_features, 'zero').astype(np.float32)

            prev_is_null = shift_features(_is_null, -1)
            next_is_null = shift_features(_is_null, +1)
            _is_null = _is_null | prev_is_null | next_is_null

            motion_features[_is_null[..., 0], :] = 0
            inputs['motion_features'] = motion_features

        inputs['features'] = _features
        inputs['masks'] = np.ones((len(_features),), dtype=bool)

        if self.mode == 'train' and self.cfg.drop_frame_rate > 0.0:
            drop_frames = np.random.uniform(0, 1, size=len(_features)) < self.cfg.drop_frame_rate
            inputs['features'][drop_frames] = 0.0
            if self.cfg.motion_features:
                inputs['motion_features'][drop_frames] = 0.0
            inputs['masks'][drop_frames] = False
        inputs['labels'] = row['label'].astype(int)
        return inputs