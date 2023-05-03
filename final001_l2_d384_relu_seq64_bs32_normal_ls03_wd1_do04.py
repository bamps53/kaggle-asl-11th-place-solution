import argparse
import gc
import os
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import wandb
from configs.base import cfg
from datasets.base import TrainDataset as Dataset
from datasets.common import get_train_dataloader, get_val_dataloader
from models.clip import SimpleCLIP as Net
from optimizers import get_optimizer
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.common import (batch_to_device, create_checkpoint, log_results,
                          resume_checkpoint, set_seed)
from utils.debugger import set_debugger
from utils.ema import ModelEmaV2

warnings.simplefilter(action='ignore', category=FutureWarning)

MAX_LEN = 64

cfg = deepcopy(cfg)
cfg.project = 'kaggle-asl'
cfg.exp_name = 'final001_l2_d384_relu_seq64_bs32_normal_ls03_wd1_do04'
cfg.exp_id = cfg.exp_name.split('_')[0]
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/preprocessed_data/train_df_with_folds.csv'
cfg.train.max_len = MAX_LEN
cfg.train.batch_size = 32
cfg.train.num_workers = 4
cfg.train.fillna_mode = 'zero'
cfg.train.normalize_mode = 'mean_std'
cfg.train.feature_path = '../input/preprocessed_data//pre_select_features.pkl'
cfg.train.resize_rate = 0.8
cfg.train.resize_range = 0.5
cfg.train.drop_frame_rate = 0.5
cfg.train.angle_range = 45
cfg.train.scale_range = 0.5
cfg.train.shift_range = 0.3
cfg.train.random_flip = True
cfg.train.motion_features = False

cfg.valid.df_path = '../input/preprocessed_data/train_df_with_folds.csv'
cfg.valid.max_len = MAX_LEN
cfg.valid.batch_size = 128
cfg.valid.num_workers = 4
cfg.valid.fillna_mode = 'zero'
cfg.valid.normalize_mode = 'mean_std'
cfg.valid.feature_path = '../input/preprocessed_data//pre_select_features.pkl'
cfg.valid.motion_features = False

cfg.model.max_len = MAX_LEN
cfg.model.num_features = 61
cfg.model.num_classes = 250
cfg.model.loss_type = 'ls_bce'
cfg.model.label_smoothing = 0.3
cfg.model.final_drop_rate = 0.4

cfg.model.clip.hidden_size = 384
cfg.model.clip.intermediate_size = 768
cfg.model.clip.num_attention_heads = 16
cfg.model.clip.num_hidden_layers = 2
cfg.model.clip.act_name = 'relu'

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-4
cfg.wd = 1.0e-4
cfg.min_lr = 1.0e-4
cfg.warmup_lr = 1.0e-5
cfg.warmup_epochs = 1
cfg.warmup = 1
cfg.epochs = 300
cfg.eval_intervals = 1
cfg.mixed_precision = True
cfg.ema_start_epoch = 5


def get_model(cfg, weight_path=None, export=False):
    if export:
        cfg.model.export = True
    model = Net(cfg.model)
    if cfg.model.resume_exp is not None:
        weight_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location='cpu')
        epoch = state_dict['epoch']
        model_key = 'model_ema'
        if model_key not in state_dict.keys():
            model_key = 'model'
            print(f'load epoch {epoch} model from {weight_path}')
        else:
            print(f'load epoch {epoch} ema model from {weight_path}')

        model.load_state_dict(state_dict[model_key])

    return model.to(cfg.device)


def save_val_results(targets, preds, save_path):
    num_classes = targets.shape[1]
    df = pd.DataFrame()
    for c in range(num_classes):
        df[f'target_{c}'] = targets[:, c]
        df[f'pred_{c}'] = preds[:, c]
    df.to_csv(save_path, index=False)


def my_collate_fn(data):
    features = [torch.from_numpy(d['features']) for d in data]
    # dist_features = [torch.from_numpy(d['dist_features']) for d in data]
    masks = [torch.from_numpy(d['masks']) for d in data]
    labels = torch.LongTensor([d['labels'] for d in data])

    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=False)

    if not cfg.train.motion_features:
        return {'features': padded_features, 'masks': padded_masks, 'labels': labels}

    else:
        # padded_dist_features = pad_sequence(dist_features, batch_first=True, padding_value=0)
        motion_features = [torch.from_numpy(d['motion_features']) for d in data]
        padded_motion_features = pad_sequence(motion_features, batch_first=True, padding_value=0)
        return {'features': padded_features, 'motion_features': padded_motion_features, 'masks': padded_masks, 'labels': labels}


def train(cfg, fold):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)
    cfg.fold = fold
    mode = 'disabled' if cfg.debug else None
    wandb.init(project=cfg.project,
               name=f'{cfg.exp_name}_fold{fold}', config=cfg, reinit=True, mode=mode)
    set_seed(cfg.seed)
    train_dataloader = get_train_dataloader(
        Dataset(cfg.train, fold=fold, mode="train"),
        cfg.train,
        collate_fn=my_collate_fn,
    )
    if fold != -1:
        valid_dataloader = get_val_dataloader(
            Dataset(cfg.valid, fold=fold, mode="valid"),
            cfg.valid,
            collate_fn=my_collate_fn,
        )
    model = get_model(cfg)

    # if cfg.model.grad_checkpointing:
    #     model.set_grad_checkpointing(enable=True)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = ModelEmaV2(model, decay=0.999)

    optimizer = get_optimizer(model, cfg)
    steps_per_epoch = len(train_dataloader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs*steps_per_epoch,
        lr_min=cfg.min_lr,
        warmup_lr_init=cfg.warmup_lr,
        warmup_t=cfg.warmup_epochs*steps_per_epoch,
        k_decay=1.0,
    )

    scaler = GradScaler(enabled=cfg.mixed_precision)
    init_epoch = 0
    best_val_score = 0
    ckpt_path = f"{cfg.output_dir}/last_fold{fold}.pth"
    if cfg.resume and os.path.exists(ckpt_path):
        model, optimizer, init_epoch, best_val_score, scheduler, scaler, model_ema = resume_checkpoint(
            f"{cfg.output_dir}/last_fold{fold}.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            model_ema
        )

    cfg.curr_step = 0
    i = init_epoch * steps_per_epoch

    optimizer.zero_grad()
    for epoch in range(init_epoch, cfg.epochs):
        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        progress_bar = tqdm(range(len(train_dataloader)),
                            leave=True,  dynamic_ncols=True)
        tr_it = iter(train_dataloader)

        train_outputs = defaultdict(list)
        gc.collect()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.train.batch_size

            model.train()
            torch.set_grad_enabled(True)

            inputs = next(tr_it)
            inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)

            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(inputs)
                loss_dict = model.get_loss(outputs, inputs)
                loss = loss_dict['loss']

            train_outputs['loss'].append(loss.item())
            train_outputs['preds'].append(outputs['preds'].sigmoid().detach().cpu().numpy())
            train_outputs['labels'].append(inputs['labels'][outputs['is_valids']].cpu().numpy())

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            NUM_RECENT = 10
            labels = np.concatenate(train_outputs["labels"][-NUM_RECENT:], axis=0).reshape(-1)
            preds = np.concatenate(train_outputs["preds"][-NUM_RECENT:], axis=0).argmax(1).reshape(-1)
            score = np.mean(labels == preds)
            train_loss = np.mean(train_outputs["loss"][-NUM_RECENT:])
            lr = optimizer.param_groups[0]['lr']
            text = f"step:{i} loss: {train_loss:.4f} acc: {score:.3f} lr:{lr:.6}"

            progress_bar.set_description(text)

        labels = np.concatenate(train_outputs["labels"], axis=0).reshape(-1)
        preds = np.concatenate(train_outputs["preds"], axis=0).argmax(1).reshape(-1)
        score = np.mean(labels == preds)

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")

        if fold == -1:
            val_results = {}
        else:
            if epoch % cfg.eval_intervals == 0:
                if model_ema is not None:
                    val_results = validate(cfg, fold, model_ema.module, valid_dataloader)
                else:
                    val_results = validate(cfg, fold, model, valid_dataloader)
            else:
                val_results = {}
        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {'score': score}
        log_results(all_results, train_results, val_results)

        val_score = val_results.get('score', 0.0)
        if best_val_score < val_score:
            best_val_score = val_score
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler, score=best_val_score,
                model_ema=model_ema
            )
            torch.save(checkpoint, f"{cfg.output_dir}/best_fold{fold}.pth")

        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint, f"{cfg.output_dir}/epoch{epoch}_fold{fold}.pth")


def validate(cfg, fold, model=None, valid_dataloader=None):
    if model is None:
        weight_path = f"{cfg.output_dir}/last_fold{fold}.pth"
        model = get_model(cfg, weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if valid_dataloader is None:
        valid_dataloader = get_val_dataloader(
            Dataset(cfg.valid, fold=fold, mode="valid"),
            cfg.valid,
            collate_fn=my_collate_fn,
        )

    val_outputs = defaultdict(list)
    for i, inputs in enumerate(tqdm(valid_dataloader)):
        inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)
        with torch.no_grad() and autocast(cfg.mixed_precision):
            outputs = model(inputs)

        val_outputs['preds'].append(outputs['preds'].sigmoid().detach().cpu().numpy())
        val_outputs['labels'].append(inputs['labels'].cpu().numpy())

    labels = np.concatenate(val_outputs["labels"], axis=0).reshape(-1)
    preds = np.concatenate(val_outputs["preds"], axis=0).argmax(1).reshape(-1)
    np.save(f"{cfg.output_dir}/val_preds_fold{fold}.npy", preds)
    score = np.mean(labels == preds)
    print('score=', score)
    val_results = {'score': score}
    return val_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=5, type=int)
    parser.add_argument("--seed", "-se", default=42, type=int)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")

    return parser.parse_args()


def update_cfg(cfg, args, fold):
    if args.debug:
        cfg.debug = True
        cfg.train.num_workers = 4 if not cfg.debug else 0
        cfg.valid.num_workers = 4 if not cfg.debug else 0
        set_debugger()

    cfg.fold = fold

    if args.resume:
        cfg.resume = True

    cfg.root = args.root

    cfg.output_dir = os.path.join(args.root, cfg.output_dir)

    if cfg.model.resume_exp is not None:
        cfg.model.pretrained_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')

    cfg.seed = args.seed
    if cfg.seed != 42:
        cfg.exp_name = f'{cfg.exp_name}_seed{cfg.seed}'
    cfg.exp_id = cfg.exp_name.split('_')[0]
    cfg.output_dir = f'output/{cfg.exp_name}'
    return cfg


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    for fold in range(args.start_fold, args.end_fold):
        cfg = update_cfg(cfg, args, fold)
        if args.validate:
            validate(cfg, fold)
        else:
            train(cfg, fold)
