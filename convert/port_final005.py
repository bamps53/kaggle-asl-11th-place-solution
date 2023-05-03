import tensorflow as tf
import numpy as np
import torch

from convert.common import get_pt_state_dict
from models.tf_clip_export import MotionEyeCLIP
from final005_l3_d256_gelu_seq48_bs128_motion_eye_ls03_wd1_do04 import cfg, get_model

N = cfg.model.num_features * 3

tf_net = MotionEyeCLIP(cfg.model)
tf_net(tf.keras.Input(shape=((None, N, 3))))

def convert_to_tf_saved_model(weight_path, saved_model_dir):
    pt_net = get_model(cfg, weight_path, export=True)
    pt_net = pt_net.cpu().eval()
    pt_state_dict = get_pt_state_dict(pt_net)

    for var in tf_net.weights:
        pt_w = pt_state_dict[var.name]
        if 'kernel' in var.name:
            tf_w = np.transpose(pt_w, [1, 0])
        else:
            tf_w = pt_w
        var.assign(tf_w)
    tf.saved_model.save(tf_net, saved_model_dir)

convert_to_tf_saved_model('output/final005_l3_d256_gelu_seq48_bs128_motion_eye_ls03_wd1_do04_seed4/epoch299_fold-1.pth', 'output_models/final005_seed4')