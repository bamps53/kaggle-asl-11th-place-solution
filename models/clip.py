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
""" PyTorch CLIP model."""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import numpy as np

from .losses import LabelSmoothBCEWithLogitsLoss


def positional_encoding(length, embed_dim):
    dim = embed_dim//2

    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)

    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)

    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, src_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


ACT2FN = {
    "relu": nn.ReLU(),
    "quick_gelu": QuickGELUActivation(),
}


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.act_name]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask)
        return hidden_states


class MotionEyeCLIP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        pos_embed = positional_encoding(self.model_cfg.max_len, model_cfg.clip.hidden_size)
        self.pos_embed = nn.Parameter(pos_embed)
        self.left_embed = nn.Linear(21 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.right_embed = nn.Linear(21 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.lip_embed = nn.Linear(40 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.eyes_embed = nn.Linear(32 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)

        self.hand_motion_embed = nn.Linear(21 * self.model_cfg.num_coords * 3, model_cfg.clip.hidden_size, bias=True)
        self.lip_motion_embed = nn.Linear(40 * self.model_cfg.num_coords * 3, model_cfg.clip.hidden_size, bias=True)
        self.eyes_motion_embed = nn.Linear(32 * self.model_cfg.num_coords * 3, model_cfg.clip.hidden_size, bias=True)

        self.summarize = nn.Linear(model_cfg.clip.hidden_size * 2, model_cfg.clip.hidden_size)

        self.transformer = CLIPEncoder(model_cfg.clip)
        self.logit = nn.Linear(model_cfg.clip.hidden_size, model_cfg.num_classes)

        self.cls_loss = LabelSmoothBCEWithLogitsLoss(model_cfg.label_smoothing)

        self.export = model_cfg.export
        self.final_drop_rate = model_cfg.final_drop_rate

    def forward(self, inputs):
        x = inputs['features']
        B, L, N, C = x.shape
        left_nans = (x[:, :, 40, 0] == 0).sum(1)
        right_nans = (x[:, :, 61, 0] == 0).sum(1)

        lips = self.lip_embed(x[:, :, :40].view(B, L, -1))
        left = self.left_embed(x[:, :, 40:61].view(B, L, -1))
        right = self.right_embed(x[:, :, 61:82].view(B, L, -1))
        eyes = self.eyes_embed(x[:, :, 82:114].view(B, L, -1))

        hands = torch.where((left_nans < right_nans)[:, None, None], left, right)

        motion_x = inputs['motion_features']
        lips_motion = self.lip_motion_embed(motion_x[:, :, :40].view(B, L, -1))
        left_motion = self.hand_motion_embed(motion_x[:, :, 40:61].view(B, L, -1))
        right_motion = self.hand_motion_embed(motion_x[:, :, 61:82].view(B, L, -1))
        eyes_motion = self.eyes_motion_embed(motion_x[:, :, 82:114].view(B, L, -1))

        hand_motion = torch.where((left_nans < right_nans)[:, None, None], left_motion, right_motion)

        coords = lips + hands + eyes
        motion = lips_motion + hand_motion + eyes_motion

        x = torch.cat([coords, motion], dim=-1)
        x = self.summarize(x)

        if self.export:
            x = x + self.pos_embed[:L].unsqueeze(0)
            x = self.transformer(x, attention_mask=None)
            x = x.mean(1)
            logits = self.logit(x)
            return {'preds': logits}
        else:
            masks = inputs['masks']
            num_tokens = inputs['masks'].sum(1)
            is_valids = num_tokens > 0
            x = x[is_valids]
            masks = masks[is_valids]
            num_tokens = num_tokens[is_valids]

            attention_mask = _expand_mask(masks, x.dtype)

            x = x + self.pos_embed[:L].unsqueeze(0)
            x = self.transformer(x, attention_mask=attention_mask)
            x = (x * masks[..., None]).sum(1) / num_tokens[..., None]
            x = F.dropout(x, p=self.final_drop_rate, training=self.training)
            logits = self.logit(x)
            outputs = {
                'preds': logits,
                'is_valids': is_valids,
            }
            return outputs

    def get_loss(self, outputs, inputs):
        y_pred = outputs['preds']
        y_true = inputs['labels'][outputs['is_valids']]
        y_true = F.one_hot(y_true, num_classes=self.model_cfg.num_classes).float()
        loss = self.cls_loss(y_pred, y_true, )
        loss_dict = {
            'loss': loss,
        }
        return loss_dict


class SimpleCLIP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        pos_embed = positional_encoding(self.model_cfg.max_len, model_cfg.clip.hidden_size)
        self.pos_embed = nn.Parameter(pos_embed)
        self.left_embed = nn.Linear(21 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.right_embed = nn.Linear(21 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.lip_embed = nn.Linear(40 * self.model_cfg.num_coords, model_cfg.clip.hidden_size, bias=True)
        self.transformer = CLIPEncoder(model_cfg.clip)
        self.logit = nn.Linear(model_cfg.clip.hidden_size, model_cfg.num_classes)
        self.cls_loss = LabelSmoothBCEWithLogitsLoss(model_cfg.label_smoothing)

        self.export = model_cfg.export

    def forward(self, inputs):
        x = inputs['features']
        B, L, N, C = x.shape
        left_nans = (x[:, :, 40, 0] == 0).sum(1)
        right_nans = (x[:, :, 61, 0] == 0).sum(1)

        lips = self.lip_embed(x[:, :, :40].view(B, L, -1))
        left = self.left_embed(x[:, :, 40:61].view(B, L, -1))
        right = self.right_embed(x[:, :, 61:82].view(B, L, -1))

        hands = torch.where((left_nans < right_nans)[:, None, None], left, right)

        x = lips + hands

        if self.export:
            x = x + self.pos_embed[:L].unsqueeze(0)
            x = self.transformer(x, attention_mask=None)
            x = x.mean(1)
            logits = self.logit(x)
            return {'preds': logits}
        else:
            masks = inputs['masks']
            num_tokens = inputs['masks'].sum(1)
            is_valids = num_tokens > 0
            x = x[is_valids]
            masks = masks[is_valids]
            num_tokens = num_tokens[is_valids]

            attention_mask = _expand_mask(masks, x.dtype)

            x = x + self.pos_embed[:L].unsqueeze(0)
            x = self.transformer(x, attention_mask=attention_mask)
            x = (x * masks[..., None]).sum(1) / num_tokens[..., None]
            x = F.dropout(x, p=0.4, training=self.training)
            logits = self.logit(x)
            outputs = {
                'preds': logits,
                'is_valids': is_valids,
            }
            return outputs

    def get_loss(self, outputs, inputs):
        y_pred = outputs['preds']
        y_true = inputs['labels'][outputs['is_valids']]
        y_true = F.one_hot(y_true, num_classes=self.model_cfg.num_classes).float()
        loss = self.cls_loss(y_pred, y_true, )
        loss_dict = {
            'loss': loss,
        }
        return loss_dict
