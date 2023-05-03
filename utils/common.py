import gc
import time
from contextlib import contextmanager

import pandas as pd
import numpy as np
import random
import os
import torch
import wandb


@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"[{name}] {elapsed:.3f}sec")


def setup_df(df_path, fold, mode):
    df = pd.read_csv(df_path)
    if mode == "train":
        index = df.folds != fold
    elif mode == 'valid':  # 'valid
        index = df.folds == fold
    else:
        index = df.index
    df = df.loc[index]
    df = df.reset_index(drop=True)
    return df


def pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths)


def torch_pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    b = len(x)
    res = x.shape[1:]

    num_pad = max_len - b
    pad = torch.zeros((num_pad, *res)).to(x)
    return torch.cat([x, pad], dim=0)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None, score=None, model_ema=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if score is not None:
        checkpoint['score'] = score

    if model_ema is not None:
        checkpoint['model_ema'] = model_ema.module.state_dict()

    return checkpoint


def resume_checkpoint(ckpt_path, model, optimizer, scheduler=None, scaler=None, model_ema=None):

    ckpt = torch.load(ckpt_path, map_location='cpu')

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt.get('epoch', 0) + 1
    score = ckpt.get('score', 0)

    print(f'resume training from {ckpt_path}')
    print(f'start training from epoch={epoch}')

    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])
    if model_ema is not None:
        model_ema.module.load_state_dict(ckpt['model_ema'])
    ret = (model, optimizer, epoch, score, scheduler, scaler, model_ema)

    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return ret


def batch_to_device(batch, device, mixed_precision=False):
    batch_dict = {}
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch_dict[k] = batch[k].to(device, non_blocking=True)
        else:
            batch_dict[k] = batch[k]

        if mixed_precision and (k == 'features'):
            batch_dict[k] = batch_dict[k].half()
        # elif isinstance(batch[k], torch.Tensor):
        #     batch_dict[k] = batch_dict[k].float()
    return batch_dict


def nms(predictions, scores, nms_threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        keep_pred = predictions[[i]]
        other_preds = predictions[order]
        enough_far = np.abs(other_preds - keep_pred) > nms_threshold
        order = order[enough_far]
    return keep


def log_results(all_results, train_results, val_results):
    def _add_text(text, key, value):
        if isinstance(value, float):
            text += f'{key}:{value:.3} '
        elif isinstance(value, (int, str)):
            text += f'{key}:{value} '
        else:
            print(key, value)
            # raise NotImplementedError
        return text

    text = "train "
    for k, v in all_results.items():
        text = _add_text(text, k, v)
    for k, v in train_results.items():
        text = _add_text(text, k, v)
    print(text)

    text = "valid "
    for k, v in val_results.items():
        text = _add_text(text, k, v)
    print(text)

    all_results.update({f'train_{k}': v for k, v in train_results.items()})
    all_results.update({f'val_{k}': v for k, v in val_results.items()})
    wandb.log(all_results)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
