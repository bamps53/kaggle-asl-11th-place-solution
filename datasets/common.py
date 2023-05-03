from torch.utils.data import DataLoader


def get_train_dataloader(dataset, data_cfg, collate_fn=None):
    train_dataloader = DataLoader(
        dataset,
        sampler=None,
        shuffle=True,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_dataloader


def get_val_dataloader(dataset, data_cfg, collate_fn=None):
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return val_dataloader
