import math

import keras
import numpy as np
from keras import layers, ops


def _inverse_sigmoid(x, eps=1e-3):
    x = ops.clip(x, eps, 1.0 - eps)
    return ops.log(x / (1.0 - x))


def _box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = ops.split(boxes, 4, axis=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return ops.concatenate([x0, y0, x1, y1], axis=-1)


def _rotate_pairwise(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = ops.stack([-x2, x1], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(rotated, shape)


def _apply_rotary_pos_emb_2d(q, k, cos, sin):
    q_embed = q * cos + _rotate_pairwise(q) * sin
    k_embed = k * cos + _rotate_pairwise(k) * sin
    return q_embed, k_embed


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTRotaryEmbedding(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        end_x,
        end_y,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.end_x = end_x
        self.end_y = end_y
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_attention_heads

        dim = self.head_dim
        freqs = 1.0 / (rope_theta ** (np.arange(0, dim, 4).astype("float32") / dim))

        t_x = np.arange(end_x, dtype="float32")
        t_y = np.arange(end_y, dtype="float32")

        freqs_x = np.outer(t_x, freqs)
        freqs_y = np.outer(t_y, freqs)

        freqs_x = np.tile(freqs_x[:, np.newaxis, :], (1, end_y, 1))
        freqs_y = np.tile(freqs_y[np.newaxis, :, :], (end_x, 1, 1))

        freqs_xy = np.concatenate([freqs_x, freqs_y], axis=-1)
        freqs_xy = freqs_xy.reshape(end_x * end_y, -1)
        freqs_xy = np.repeat(freqs_xy, 2, axis=-1)

        self._cos = np.cos(freqs_xy).astype("float32")
        self._sin = np.sin(freqs_xy).astype("float32")

    def call(self, inputs=None):
        cos = ops.convert_to_tensor(self._cos)
        sin = ops.convert_to_tensor(self._sin)
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "end_x": self.end_x,
                "end_y": self.end_y,
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTRoPEAttention(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.hidden_size, name="q_proj")
        self.q_proj.build(input_shape)
        self.k_proj = layers.Dense(self.hidden_size, name="k_proj")
        self.k_proj.build(input_shape)
        self.v_proj = layers.Dense(self.hidden_size, name="v_proj")
        self.v_proj.build(input_shape)
        self.o_proj = layers.Dense(self.hidden_size, name="o_proj")
        self.o_proj.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = ops.reshape(
            q, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )
        k = ops.reshape(
            k, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )
        v = ops.reshape(
            v, (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        )

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(ops.expand_dims(cos, 0), 0)
        sin = ops.expand_dims(ops.expand_dims(sin, 0), 0)
        q, k = _apply_rotary_pos_emb_2d(q, k, cos, sin)

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = self.o_proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTLayerScale(layers.Layer):
    def __init__(self, hidden_size, init_value, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.init_value = init_value

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(self.hidden_size,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        return x * self.lambda1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "init_value": self.init_value,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3ViTLayer(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        window_size=0,
        image_size=72,
        layer_norm_eps=1e-6,
        layer_scale_init_value=None,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.rope_theta = rope_theta

    def build(self, input_shape):
        seq_shape = (None, None, self.hidden_size)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.attn = SAM3ViTRoPEAttention(
            self.hidden_size, self.num_attention_heads, name="attention"
        )
        self.attn.build(seq_shape)

        end = self.window_size if self.window_size > 0 else self.image_size
        self.rotary_emb = SAM3ViTRotaryEmbedding(
            self.hidden_size,
            self.num_attention_heads,
            end_x=end,
            end_y=end,
            rope_theta=self.rope_theta,
            name="rotary_emb",
        )

        self.mlp_fc1 = layers.Dense(self.intermediate_size, name="mlp_fc1")
        self.mlp_fc1.build(seq_shape)
        self.mlp_fc2 = layers.Dense(self.hidden_size, name="mlp_fc2")
        self.mlp_fc2.build((None, None, self.intermediate_size))

        if self.layer_scale_init_value is not None:
            self.layer_scale1 = SAM3ViTLayerScale(
                self.hidden_size, self.layer_scale_init_value, name="layer_scale1"
            )
            self.layer_scale1.build(seq_shape)
            self.layer_scale2 = SAM3ViTLayerScale(
                self.hidden_size, self.layer_scale_init_value, name="layer_scale2"
            )
            self.layer_scale2.build(seq_shape)

        self.built = True

    def _window_partition(self, x, window_size):
        shape = ops.shape(x)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        padded_h = height + pad_h
        padded_w = width + pad_w

        x = ops.reshape(
            x,
            (
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, window_size * window_size, channels))
        return x, (padded_h, padded_w)

    def _window_unpartition(self, windows, window_size, pad_hw, original_hw):
        padded_h, padded_w = pad_hw
        height, width = original_hw
        num_h = padded_h // window_size
        num_w = padded_w // window_size

        channels = ops.shape(windows)[-1]
        batch_size = ops.shape(windows)[0] // (num_h * num_w)

        x = ops.reshape(
            windows, (batch_size, num_h, num_w, window_size, window_size, channels)
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, padded_h, padded_w, channels))
        x = x[:, :height, :width, :]
        return x

    def call(self, hidden_states):
        shape = ops.shape(hidden_states)
        batch_size, height, width = shape[0], shape[1], shape[2]

        residual = hidden_states

        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_size))
        x = self.layer_norm1(x)

        if self.window_size > 0:
            x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
            x, pad_hw = self._window_partition(x, self.window_size)
        else:
            pass

        pos_emb = self.rotary_emb()
        x = self.attn(x, pos_emb)

        if self.window_size > 0:
            x = self._window_unpartition(x, self.window_size, pad_hw, (height, width))
            x = ops.reshape(x, (batch_size, height * width, self.hidden_size))

        if self.layer_scale_init_value is not None:
            x = self.layer_scale1(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
        hidden_states = residual + x

        residual = hidden_states
        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_size))
        x = self.layer_norm2(x)
        x = self.mlp_fc1(x)
        x = ops.nn.gelu(x, approximate=False)
        x = self.mlp_fc2(x)

        if self.layer_scale_init_value is not None:
            x = self.layer_scale2(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_size))
        hidden_states = residual + x

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
                "layer_scale_init_value": self.layer_scale_init_value,
                "rope_theta": self.rope_theta,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3SinePositionEmbedding(layers.Layer):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs
    ):
        super().__init__(**kwargs)
        if scale is None:
            scale = 2 * math.pi
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def call(self, height, width, batch_size=1):
        not_mask = ops.ones((batch_size, height, width), dtype="float32")
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, :, 1::2])
        pos_y_sin = ops.sin(pos_y[:, :, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, :, 1::2])

        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos

    def encode_boxes(self, boxes):
        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        boxes_scaled = boxes * self.scale
        pos = ops.expand_dims(boxes_scaled, axis=-1) / dim_t

        pos_sin = ops.sin(pos[..., 0::2])
        pos_cos = ops.cos(pos[..., 1::2])
        pos_embed = ops.reshape(
            ops.stack([pos_sin, pos_cos], axis=-1),
            ops.shape(pos)[:-1] + (self.num_pos_feats,),
        )
        shape = ops.shape(pos_embed)
        return ops.reshape(pos_embed, (shape[0], shape[1], shape[2] * shape[3]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3MultiHeadAttention(layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout

    def build(self, input_shape):
        dim = self.hidden_size
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.q_proj.build((None, None, dim))
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.k_proj.build((None, None, dim))
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.v_proj.build((None, None, dim))
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.o_proj.build((None, None, dim))
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(self, query, key, value, attention_mask=None, training=None):
        batch_size = ops.shape(query)[0]
        seq_q = ops.shape(query)[1]
        seq_k = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, (batch_size, seq_q, self.num_attention_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_k, self.num_attention_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_k, self.num_attention_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, seq_q, self.hidden_size))
        return self.o_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DetrEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="cross_attn"
        )
        self.cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        vision_feats,
        text_feats,
        vision_pos,
        text_mask=None,
        training=None,
    ):
        q = k = vision_feats + vision_pos
        x = self.self_attn(q, k, vision_feats, training=training)
        x = self.dropout1(x, training=training)
        vision_feats = self.layer_norm1(vision_feats + x)

        x = self.cross_attn(
            vision_feats,
            text_feats,
            text_feats,
            attention_mask=text_mask,
            training=training,
        )
        x = self.dropout2(x, training=training)
        vision_feats = self.layer_norm2(vision_feats + x)

        x = self.fc1(vision_feats)
        x = ops.nn.relu(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        vision_feats = self.layer_norm3(vision_feats + x)

        return vision_feats

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DetrDecoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.text_cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="text_cross_attn"
        )
        self.text_cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.vision_cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="vision_cross_attn"
        )
        self.vision_cross_attn.build(seq_shape)
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm4 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm4"
        )
        self.layer_norm4.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.dropout4 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        hidden_states,
        query_pos,
        text_feats,
        vision_feats,
        vision_pos,
        text_mask=None,
        vision_mask=None,
        training=None,
    ):
        q = k = hidden_states + query_pos
        x = self.self_attn(q, k, hidden_states, training=training)
        x = self.dropout1(x, training=training)
        hidden_states = self.layer_norm1(hidden_states + x)

        x = self.text_cross_attn(
            hidden_states,
            text_feats,
            text_feats,
            attention_mask=text_mask,
            training=training,
        )
        x = self.dropout2(x, training=training)
        hidden_states = self.layer_norm2(hidden_states + x)

        x = self.vision_cross_attn(
            hidden_states,
            vision_feats + vision_pos,
            vision_feats,
            attention_mask=vision_mask,
            training=training,
        )
        x = self.dropout3(x, training=training)
        hidden_states = self.layer_norm3(hidden_states + x)

        x = self.fc1(hidden_states)
        x = ops.nn.relu(x)
        x = self.dropout4(x, training=training)
        x = self.fc2(x)
        hidden_states = self.layer_norm4(hidden_states + x)

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3GeometryEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.cross_attn = SAM3MultiHeadAttention(
            dim, self.num_attention_heads, self.dropout_rate, name="cross_attn"
        )
        self.cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.fc1 = layers.Dense(self.intermediate_size, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.intermediate_size))
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        prompt_feats,
        vision_feats,
        vision_pos,
        prompt_mask=None,
        training=None,
    ):
        x = self.self_attn(
            prompt_feats,
            prompt_feats,
            prompt_feats,
            attention_mask=prompt_mask,
            training=training,
        )
        x = self.dropout1(x, training=training)
        prompt_feats = self.layer_norm1(prompt_feats + x)

        k = vision_feats + vision_pos
        x = self.cross_attn(
            prompt_feats,
            k,
            vision_feats,
            training=training,
        )
        x = self.dropout2(x, training=training)
        prompt_feats = self.layer_norm2(prompt_feats + x)

        x = self.fc1(prompt_feats)
        x = ops.nn.relu(x)
        x = self.dropout3(x, training=training)
        x = self.fc2(x)
        prompt_feats = self.layer_norm3(prompt_feats + x)

        return prompt_feats

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3MaskEmbedder(layers.Layer):
    def __init__(self, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        dim = self.hidden_size
        seq_shape = (None, None, dim)
        self.linear1 = layers.Dense(dim, name="linear1")
        self.linear1.build(seq_shape)
        self.linear2 = layers.Dense(dim, name="linear2")
        self.linear2.build(seq_shape)
        self.linear3 = layers.Dense(dim, name="linear3")
        self.linear3.build(seq_shape)
        self.built = True

    def call(self, x):
        x = self.linear1(x)
        x = ops.nn.relu(x)
        x = self.linear2(x)
        x = ops.nn.relu(x)
        x = self.linear3(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DotProductScoring(layers.Layer):
    def __init__(
        self,
        hidden_size=256,
        text_hidden_size=1024,
        text_projection_dim=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.text_hidden_size = text_hidden_size
        self.text_projection_dim = text_projection_dim

    def build(self, input_shape):
        self.text_mlp_fc1 = layers.Dense(self.text_hidden_size, name="text_mlp_fc1")
        self.text_mlp_fc1.build((None, None, self.text_hidden_size))
        self.text_mlp_fc2 = layers.Dense(self.text_hidden_size, name="text_mlp_fc2")
        self.text_mlp_fc2.build((None, None, self.text_hidden_size))

        self.text_proj = layers.Dense(
            self.text_projection_dim, use_bias=False, name="text_proj"
        )
        self.text_proj.build((None, self.text_hidden_size))
        self.query_proj = layers.Dense(
            self.text_projection_dim, use_bias=False, name="query_proj"
        )
        self.query_proj.build((None, None, self.hidden_size))

        self._scale = self.text_projection_dim**-0.5
        self.built = True

    def call(self, decoder_hidden_states, text_features, text_mask=None):
        x = self.text_mlp_fc1(text_features)
        x = ops.nn.gelu(x, approximate=False)
        x = self.text_mlp_fc2(x)
        text_feats = text_features + x

        if text_mask is not None:
            mask_expanded = ops.expand_dims(ops.cast(text_mask, "float32"), axis=-1)
            text_pooled = ops.sum(text_feats * mask_expanded, axis=1) / (
                ops.sum(mask_expanded, axis=1) + 1e-8
            )
        else:
            text_pooled = ops.mean(text_feats, axis=1)

        text_proj = self.text_proj(text_pooled)
        text_proj = text_proj / (
            ops.sqrt(ops.sum(ops.square(text_proj), axis=-1, keepdims=True)) + 1e-8
        )

        query_proj = self.query_proj(decoder_hidden_states)
        query_proj = query_proj / (
            ops.sqrt(ops.sum(ops.square(query_proj), axis=-1, keepdims=True)) + 1e-8
        )

        logits = ops.matmul(query_proj, ops.expand_dims(text_proj, axis=-1))
        logits = ops.squeeze(logits, axis=-1) * self._scale
        logits = ops.clip(logits, -12.0, 12.0)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "text_hidden_size": self.text_hidden_size,
                "text_projection_dim": self.text_projection_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3DecoderMLP(layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def build(self, input_shape):
        dims = (
            [self.input_dim]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [self.output_dim]
        )
        self._layers = []
        for i in range(self.num_layers):
            dense = layers.Dense(dims[i + 1], name=f"dense_{i}")
            dense.build((None, dims[i]) if i == 0 else (None, None, dims[i]))
            self._layers.append(dense)
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config
