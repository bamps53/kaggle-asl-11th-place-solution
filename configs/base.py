from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.train = SimpleNamespace(**{})
cfg.train.normalize_mode = 'none'
cfg.train.resize_rate = 0.5
cfg.train.resize_range = 0.2
cfg.train.drop_frame_rate = 0.0
cfg.train.noise_rate = 0.0
cfg.train.angle_range = 0.0
cfg.train.scale_range = 0.0
cfg.train.shift_range = 0.0
cfg.train.random_flip = False
cfg.train.resize_range = 0.2
cfg.train.motion_features = False
cfg.train.distance_features = False
cfg.train.normalize_mode = 'mean_std'

cfg.valid = SimpleNamespace(**{})
cfg.valid.normalize_mode = 'none'
cfg.valid.drop_frame_rate = 0.0
cfg.valid.noise_rate = 0.0
cfg.valid.angle_range = 0.0
cfg.valid.scale_range = 0.0
cfg.valid.shift_range = 0.0
cfg.valid.random_flip = False
cfg.valid.motion_features = False
cfg.valid.distance_features = False
cfg.valid.normalize_mode = 'me an_std'

cfg.model = SimpleNamespace(**{})
cfg.model.resume_exp = None
cfg.model.pretrained = True

cfg.model.num_classes = 250
cfg.model.loss_type = 'bce'
cfg.model.label_smoothing = 0.0
cfg.model.kernel_size = 3
cfg.model.export = False
cfg.model.final_drop_rate = 0.4
cfg.model.max_len = 64
cfg.model.num_coords = 3

cfg.model.clip = SimpleNamespace(**{})
cfg.model.clip.attention_dropout = 0.0
cfg.model.clip.dropout = 0.0
cfg.model.clip.hidden_size = 512
cfg.model.clip.intermediate_size = 2048
cfg.model.clip.num_attention_heads = 8
cfg.model.clip.num_hidden_layers = 12
cfg.model.clip.initializer_range = 0.02
cfg.model.clip.initializer_factor = 1.0
cfg.model.clip.layer_norm_eps = 0.00001

cfg.resume = False
cfg.optimizer = 'adamw'
