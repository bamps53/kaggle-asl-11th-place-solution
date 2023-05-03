import numpy as np
import tensorflow as tf


def pt2tf(name):
    name = name.replace('layers.', 'layers')
    name = name.replace('.', '/')
    name = name.replace('weight', 'kernel')
    if 'layer_norm' in name:
        name = name.replace('kernel', 'gamma')
        name = name.replace('bias', 'beta')
    if 'pos_embed' in name:
        return f'{name}:0'
    else:
        return f'simple_clip/{name}:0'


def get_pt_state_dict(pt_model):
    pt_state_dict = {}
    for n, p in pt_model.named_parameters():
        pt_state_dict[pt2tf(n)] = p.detach().numpy()
    return pt_state_dict


def get_tf_state_dict(tf_model):
    tf_state_dict = {}
    for var in tf_model.weights:
        tf_state_dict[var.name] = var.numpy()
    return tf_state_dict


def convert_weights(tf_model, pt_state_dict, key_map):
    for layer in tf_model.layers:
        for var in layer.weights:
            pt_key = key_map[var.name]
            pt_w = pt_state_dict[pt_key]
            if ('dense' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [1, 0])
            elif ('depthwise_conv2d' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [2, 3, 0, 1])
            elif ('conv2d' in var.name) and ('kernel' in var.name):
                tf_w = np.transpose(pt_w, [2, 3, 1, 0])
            else:
                tf_w = pt_w
            var.assign(tf_w)


def convert_to_tf_saved_model(pt_net, tf_net, saved_model_dir):
    pt_state_dict = get_pt_state_dict(pt_net)

    for var in tf_net.weights:
        pt_w = pt_state_dict[var.name]
        if 'kernel' in var.name:
            tf_w = np.transpose(pt_w, [1, 0])
        else:
            tf_w = pt_w
        var.assign(tf_w)
    tf.saved_model.save(tf_net, saved_model_dir)
