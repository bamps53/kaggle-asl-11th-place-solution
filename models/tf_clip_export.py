# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 CLIP model."""


import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

CONFIG = SimpleNamespace(**{})
CONFIG.attention_dropout = 0.0
CONFIG.dropout = 0.0
CONFIG.hidden_size = 512
CONFIG.intermediate_size = 2048
CONFIG.num_attention_heads = 8
CONFIG.num_hidden_layers = 12
CONFIG.initializer_range = 0.02
CONFIG.initializer_factor = 1.0
CONFIG.layer_norm_eps = 0.00001
CONFIG.max_position_embeddings = 64

LARGE_NEGATIVE = -1e8


def quick_gelu(x):
    x = tf.convert_to_tensor(x)
    coeff = tf.cast(1.702, x.dtype)
    return x * tf.math.sigmoid(coeff * x)


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    """
    Creates a `tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# Copied from transformers.models.bart.modeling_tf_bart._expand_mask
def _expand_mask(mask: tf.Tensor):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, src_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TFCLIPAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_attention_heads})."
            )

        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.q_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # copied from transformers.models.bert.modeling_tf_bert.TFBertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        mixed_key_layer = self.k_proj(inputs=hidden_states)
        mixed_value_layer = self.v_proj(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))
        attention_output = self.out_proj(attention_output, training=training)
        return attention_output


class TFCLIPMLP(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.activation_fn = tf.keras.activations.relu

        factor = config.initializer_factor
        in_proj_std = (config.hidden_size**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        fc_std = (2 * config.hidden_size) ** -0.5 * factor

        self.fc1 = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(fc_std), name="fc1"
        )
        self.fc2 = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(in_proj_std), name="fc2"
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:

        hidden_states = self.fc1(inputs=hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(inputs=hidden_states)
        return hidden_states


class TFCLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.self_attn = TFCLIPAttention(config, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        self.mlp = TFCLIPMLP(config, name="mlp")
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(inputs=hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            training=training,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(inputs=hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TFCLIPEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layers = [TFCLIPEncoderLayer(config, name=f"layers{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(
                hidden_states=hidden_states,
                training=training,
            )
        return hidden_states


class TFCLIPEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size
        self.config = config

        factor = config.initializer_factor
        fc_std = (2 * config.hidden_size) ** -0.5 * factor
        self.coords_embedding = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(fc_std), name="coords_embedding"
        )

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        coords: tf.Tensor,
    ) -> tf.Tensor:
        coords_embeds = self.coords_embedding(coords)
        input_shape = shape_list(coords)[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = coords_embeds + position_embeds
        return final_embeddings


class SimpleCLIP(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pos_embed = tf.Variable(initial_value=tf.ones((cfg.max_len, cfg.clip.hidden_size)), name='pos_embed')
        self.left_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='left_embed')
        self.right_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='right_embed')
        self.lip_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='lip_embed')
        self.transformer = TFCLIPEncoder(cfg.clip, name='transformer')
        self.logit = tf.keras.layers.Dense(cfg.num_classes, name='logit')

    def call(self, x):
        # B, L, N, C = x.shape
        L = shape_list(x)[1]
        left_nans = tf.math.reduce_sum(tf.cast(x[:, :, 40, 0] == 0, tf.int32), axis=1)
        right_nans = tf.math.reduce_sum(tf.cast(x[:, :, 61, 0] == 0, tf.int32), axis=1)

        lips = self.lip_embed(tf.reshape(x[:, :, :40], (-1, L, 40*3)))
        left = self.left_embed(tf.reshape(x[:, :, 40:61], (-1, L, 21*3)))
        right = self.right_embed(tf.reshape(x[:, :, 61:], (-1, L, 21*3)))

        hands = tf.where(tf.expand_dims(tf.expand_dims(left_nans < right_nans, -1), -1), left, right)

        x = lips + hands
        x = x + self.pos_embed[:L][tf.newaxis, ...]
        x = self.transformer(x)
        x = tf.reduce_mean(x, axis=1)
        logits = self.logit(x)
        return logits


class MotionEyeCLIP(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.pos_embed = tf.Variable(initial_value=tf.ones((cfg.max_len, cfg.clip.hidden_size)), name='pos_embed')
        self.left_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='left_embed')
        self.right_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='right_embed')
        self.lip_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='lip_embed')
        self.eyes_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='eyes_embed')
        self.hand_motion_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='hand_motion_embed')
        self.lip_motion_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='lip_motion_embed')
        self.eyes_motion_embed = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='eyes_motion_embed')
        self.summarize = tf.keras.layers.Dense(cfg.clip.hidden_size, use_bias=True, name='summarize')
        self.transformer = TFCLIPEncoder(cfg.clip, name='transformer')
        self.logit = tf.keras.layers.Dense(cfg.num_classes, name='logit')

    def call(self, features, motion_features):
        # B, L, N, C = x.shape
        L = shape_list(features)[1]
        left_nans = tf.math.reduce_sum(tf.cast(features[:, :, 40, 0] == 0, tf.int32), axis=1)
        right_nans = tf.math.reduce_sum(tf.cast(features[:, :, 61, 0] == 0, tf.int32), axis=1)

        lips = self.lip_embed(tf.reshape(features[:, :, :40], (-1, L, 40*3)))
        left = self.left_embed(tf.reshape(features[:, :, 40:61], (-1, L, 21*3)))
        right = self.right_embed(tf.reshape(features[:, :, 61:82], (-1, L, 21*3)))
        eyes = self.eyes_embed(tf.reshape(features[:, :, 82:114], (-1, L, 32*3)))
        hands = tf.where(tf.expand_dims(tf.expand_dims(left_nans < right_nans, -1), -1), left, right)

        lips_motion = self.lip_motion_embed(tf.reshape(motion_features[:, :, :40], (-1, L, 40*9)))
        left_motion = tf.reshape(motion_features[:, :, 40:61], (-1, L, 21*9))
        right_motion = tf.reshape(motion_features[:, :, 61:82], (-1, L, 21*9))
        eyes_motion = self.eyes_motion_embed(tf.reshape(motion_features[:, :, 82:114], (-1, L, 32*9)))
        hand_motion = tf.where(tf.expand_dims(tf.expand_dims(
            left_nans < right_nans, -1), -1), left_motion, right_motion)
        hand_motion = self.hand_motion_embed(hand_motion)

        coords = lips + hands + eyes
        motion = lips_motion + hand_motion + eyes_motion

        features = tf.concat([coords, motion], axis=2)
        features = self.summarize(features)

        features = features + self.pos_embed[:L][tf.newaxis, ...]
        features = self.transformer(features)
        features = tf.reduce_mean(features, axis=1)
        logits = self.logit(features)
        return logits
