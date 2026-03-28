import math

import keras
from keras import initializers, layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETRChannelLayerNorm(layers.Layer):
    """Layer normalization applied over the channel dimension.

    Unlike standard LayerNormalization which expects (B, ..., C) inputs,
    this layer explicitly normalizes along the last axis of (B, H, W, C)
    tensors using learnable gamma and beta parameters.

    Args:
        epsilon: Float, small constant for numerical stability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        4D tensor: `(batch_size, height, width, channels)`.

    Output Shape:
        Same as input shape.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer="ones",
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(dim,),
            initializer="zeros",
        )

    def call(self, x):
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / ops.sqrt(variance + self.epsilon)
        return x_norm * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2PatchEmbeddings(layers.Layer):
    """Converts pixel values to patch embeddings via a non-overlapping convolution.

    Splits the input image into fixed-size patches and projects each patch
    into an embedding vector using a single Conv2D with kernel and stride
    equal to the patch size.

    Args:
        hidden_size: Integer, dimensionality of the output embeddings.
        patch_size: Integer, size of each square patch (height and width).
        num_channels: Integer, number of input channels. Defaults to `3`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        4D tensor: `(batch_size, height, width, channels)`.

    Output Shape:
        3D tensor: `(batch_size, num_patches, hidden_size)`, where
        `num_patches = (height // patch_size) * (width // patch_size)`.
    """

    def __init__(self, hidden_size, patch_size, num_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection = layers.Conv2D(
            self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            data_format="channels_last",
            name="conv_projection",
        )

    def call(self, pixel_values):
        x = self.projection(pixel_values)
        shape = ops.shape(x)
        x = ops.reshape(x, [shape[0], shape[1] * shape[2], self.hidden_size])
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2Embeddings(layers.Layer):
    """DINOv2 embedding layer with CLS token, position embeddings, and windowing.

    Converts input images into a sequence of patch embeddings with learnable
    CLS token and position embeddings. Optionally splits patches into spatial
    windows for windowed attention and prepends register tokens.

    Position embeddings are interpolated via bicubic resize when the input
    resolution differs from the pretrained positional encoding grid size.

    Args:
        hidden_size: Integer, dimensionality of the embeddings.
        patch_size: Integer, size of each square patch.
        num_channels: Integer, number of input image channels. Defaults to `3`.
        num_register_tokens: Integer, number of register tokens to prepend.
            Defaults to `4`.
        num_windows: Integer, number of spatial windows per axis. When greater
            than 1, patches are partitioned into `num_windows^2` windows.
            Defaults to `1` (no windowing).
        positional_encoding_size: Integer, grid size used for the pretrained
            positional encoding. Defaults to `37`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        4D tensor: `(batch_size, height, width, channels)`.

    Output Shape:
        3D tensor: `(batch_size [* num_windows^2], seq_len, hidden_size)`, where
        `seq_len = tokens_per_window + 1 + num_register_tokens`.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        num_channels=3,
        num_register_tokens=4,
        num_windows=1,
        positional_encoding_size=37,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_register_tokens = num_register_tokens
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.num_patches = positional_encoding_size * positional_encoding_size
        self.patch_embeddings = DinoV2PatchEmbeddings(
            self.hidden_size,
            self.patch_size,
            self.num_channels,
            name="patch_embeddings",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches + 1, self.hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, self.num_register_tokens, self.hidden_size),
                initializer="zeros",
            )
        super().build(input_shape)

    def _interpolate_pos_encoding(self, embeddings, height, width):
        num_positions = self.num_patches

        class_pos_embed = self.position_embeddings[:, :1, :]
        patch_pos_embed = self.position_embeddings[:, 1:, :]

        h = height // self.patch_size
        w = width // self.patch_size
        sqrt_n = int(math.sqrt(num_positions))

        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, sqrt_n, sqrt_n, self.hidden_size]
        )
        patch_pos_embed = ops.cast(patch_pos_embed, "float32")
        patch_pos_embed = ops.image.resize(
            patch_pos_embed, (h, w), interpolation="bicubic", antialias=True
        )
        patch_pos_embed = ops.cast(patch_pos_embed, embeddings.dtype)
        patch_pos_embed = ops.reshape(patch_pos_embed, [1, h * w, self.hidden_size])
        return ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def call(self, pixel_values):
        batch_size = ops.shape(pixel_values)[0]
        height = ops.shape(pixel_values)[1]
        width = ops.shape(pixel_values)[2]

        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        embeddings = ops.concatenate([cls_tokens, embeddings], axis=1)

        pos_embed = self._interpolate_pos_encoding(embeddings, height, width)
        embeddings = embeddings + pos_embed

        if self.num_windows > 1:
            num_h = height // self.patch_size
            num_w = width // self.patch_size
            cls_tok = embeddings[:, :1, :]
            patch_tok = embeddings[:, 1:, :]
            patch_tok = ops.reshape(
                patch_tok, [batch_size, num_h, num_w, self.hidden_size]
            )
            nw = self.num_windows
            h_pw = num_h // nw
            w_pw = num_w // nw
            patch_tok = ops.reshape(
                patch_tok, [batch_size * nw, h_pw, nw, w_pw, self.hidden_size]
            )
            patch_tok = ops.transpose(patch_tok, [0, 2, 1, 3, 4])
            patch_tok = ops.reshape(
                patch_tok, [batch_size * nw * nw, h_pw * w_pw, self.hidden_size]
            )
            cls_tok = ops.tile(cls_tok, [nw * nw, 1, 1])
            embeddings = ops.concatenate([cls_tok, patch_tok], axis=1)

        if self.num_register_tokens > 0:
            reg = ops.broadcast_to(
                self.register_tokens,
                (ops.shape(embeddings)[0], self.num_register_tokens, self.hidden_size),
            )
            embeddings = ops.concatenate(
                [embeddings[:, :1, :], reg, embeddings[:, 1:, :]], axis=1
            )

        return embeddings

    def compute_output_spec(self, input_spec, **kwargs):
        if not self.patch_embeddings.projection.built:
            self.patch_embeddings.projection.build(input_spec.shape)
        h = input_spec.shape[1] // self.patch_size
        w = input_spec.shape[2] // self.patch_size
        tokens_per_window = (
            (h // self.num_windows) * (w // self.num_windows)
            if self.num_windows > 1
            else h * w
        )
        seq_len = tokens_per_window + 1 + self.num_register_tokens
        batch = input_spec.shape[0]
        if self.num_windows > 1:
            batch = None
        return keras.KerasTensor(
            shape=(batch, seq_len, self.hidden_size),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
                "num_register_tokens": self.num_register_tokens,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class DinoV2LayerScale(layers.Layer):
    """Learnable per-channel scaling applied to residual branch outputs.

    Multiplies input element-wise by a learnable vector (lambda1) of shape
    `(hidden_size,)`, initialized to a constant value. Used in DINOv2
    transformer blocks to stabilize training.

    Args:
        hidden_size: Integer, number of channels (must match the last dimension
            of the input).
        init_value: Float, initial value for the scale parameters.
            Defaults to `1.0`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        Tensor of shape `(..., hidden_size)`.

    Output Shape:
        Same as input shape.
    """

    def __init__(self, hidden_size, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.init_value = init_value

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(self.hidden_size,),
            initializer=initializers.Constant(self.init_value),
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.lambda1

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "init_value": self.init_value})
        return config


def _ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure implementation of multi-scale deformable attention core.

    Args:
        value: (B, n_heads, head_dim, N_total)
        value_spatial_shapes: list of (H, W) tuples
        sampling_locations: (B, Len_q, n_heads, L, P, 2) in [0, 1]
        attention_weights: (B, Len_q, n_heads, L*P) after softmax
    """
    B = ops.shape(value)[0]
    n_heads = ops.shape(value)[1]
    head_dim = ops.shape(value)[2]
    Len_q = ops.shape(sampling_locations)[1]
    L = len(value_spatial_shapes)
    P = ops.shape(sampling_locations)[4]

    sampling_grids = 2 * sampling_locations - 1

    splits = [h * w for h, w in value_spatial_shapes]
    value_list = ops.split(value, splits, axis=3)

    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        value_l = ops.reshape(value_list[lid], [B * n_heads, head_dim, H, W])
        value_l = ops.transpose(value_l, [0, 2, 3, 1])
        val_flat = ops.reshape(value_l, [B * n_heads, H * W, head_dim])

        grid_l = sampling_grids[:, :, :, lid, :, :]
        grid_l = ops.transpose(grid_l, [0, 2, 1, 3, 4])
        grid_l = ops.reshape(grid_l, [B * n_heads, Len_q, P, 2])

        grid_x = grid_l[..., 0]
        grid_y = grid_l[..., 1]

        W_f = ops.cast(W, grid_x.dtype)
        H_f = ops.cast(H, grid_y.dtype)
        ix = ((grid_x + 1) * W_f - 1) / 2.0
        iy = ((grid_y + 1) * H_f - 1) / 2.0

        ix0 = ops.cast(ops.floor(ix), "int32")
        iy0 = ops.cast(ops.floor(iy), "int32")
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        fx = ix - ops.cast(ix0, ix.dtype)
        fy = iy - ops.cast(iy0, iy.dtype)

        valid_00 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_01 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_10 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)
        valid_11 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)

        ix0_c = ops.clip(ix0, 0, W - 1)
        ix1_c = ops.clip(ix1, 0, W - 1)
        iy0_c = ops.clip(iy0, 0, H - 1)
        iy1_c = ops.clip(iy1, 0, H - 1)

        def _gather(val_flat, iy, ix, BN, Len_q, P, H, W, head_dim):
            idx = iy * W + ix
            idx_flat = ops.reshape(idx, [BN, Len_q * P])
            idx_flat = ops.expand_dims(idx_flat, axis=-1)
            idx_flat = ops.repeat(idx_flat, head_dim, axis=-1)
            gathered = ops.take_along_axis(val_flat, idx_flat, axis=1)
            return ops.reshape(gathered, [BN, Len_q, P, head_dim])

        BN = B * n_heads
        v00 = _gather(val_flat, iy0_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v01 = _gather(val_flat, iy0_c, ix1_c, BN, Len_q, P, H, W, head_dim)
        v10 = _gather(val_flat, iy1_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v11 = _gather(val_flat, iy1_c, ix1_c, BN, Len_q, P, H, W, head_dim)

        v00 = v00 * ops.expand_dims(valid_00, axis=-1)
        v01 = v01 * ops.expand_dims(valid_01, axis=-1)
        v10 = v10 * ops.expand_dims(valid_10, axis=-1)
        v11 = v11 * ops.expand_dims(valid_11, axis=-1)

        fx = ops.expand_dims(fx, axis=-1)
        fy = ops.expand_dims(fy, axis=-1)

        w00 = (1 - fx) * (1 - fy)
        w01 = fx * (1 - fy)
        w10 = (1 - fx) * fy
        w11 = fx * fy

        sampled = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        sampled = ops.transpose(sampled, [0, 3, 1, 2])
        sampling_value_list.append(sampled)

    sampling_values = ops.stack(sampling_value_list, axis=-2)
    sampling_values = ops.reshape(
        sampling_values, [B * n_heads, head_dim, Len_q, L * P]
    )

    attn = ops.transpose(attention_weights, [0, 2, 1, 3])
    attn = ops.reshape(attn, [B * n_heads, 1, Len_q, L * P])

    output = ops.sum(sampling_values * attn, axis=-1)
    output = ops.reshape(output, [B, n_heads * head_dim, Len_q])
    output = ops.transpose(output, [0, 2, 1])
    return output


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETRMSDeformableAttention(layers.Layer):
    """Multi-Scale Deformable Attention module.

    Implements deformable attention where each query attends to a small set of
    learned sampling locations around reference points, rather than all spatial
    positions. Uses bilinear interpolation to sample values at continuous
    locations, enabling efficient attention over large feature maps.

    Reference:
        - [Deformable DETR](https://arxiv.org/abs/2010.04159)

    Args:
        d_model: Integer, dimensionality of the model. Defaults to `256`.
        n_levels: Integer, number of feature levels. Defaults to `1`.
        n_heads: Integer, number of attention heads. Defaults to `8`.
        n_points: Integer, number of sampling points per head per level.
            Defaults to `4`.
        spatial_shapes: List of `(height, width)` tuples for each feature
            level. Defaults to `None`.
        level_start_index: List of integers indicating the start index of each
            level in the flattened feature sequence. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        - query: `(batch_size, num_queries, d_model)`.
        - reference_points: `(batch_size, num_queries, n_levels, 2 or 4)`.
        - input_flatten: `(batch_size, total_tokens, d_model)`.

    Output Shape:
        3D tensor: `(batch_size, num_queries, d_model)`.
    """

    def __init__(
        self,
        d_model=256,
        n_levels=1,
        n_heads=8,
        n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]

        self.sampling_offsets = layers.Dense(
            self.n_heads * self.n_levels * self.n_points * 2,
            name="sampling_offsets",
        )
        self.attention_weights_proj = layers.Dense(
            self.n_heads * self.n_levels * self.n_points,
            name="attention_weights",
        )
        self.value_proj = layers.Dense(self.d_model, name="value_proj")
        self.output_proj = layers.Dense(self.d_model, name="output_proj")

    def call(self, query, reference_points, input_flatten, input_padding_mask=None):
        input_spatial_shapes = self.spatial_shapes
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        Len_in = ops.shape(input_flatten)[1]

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            mask = ops.expand_dims(ops.cast(input_padding_mask, value.dtype), axis=-1)
            value = value * (1.0 - mask)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets, [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2]
        )

        attention_weights = self.attention_weights_proj(query)
        attention_weights = ops.reshape(
            attention_weights, [N, Len_q, self.n_heads, self.n_levels * self.n_points]
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)

        if reference_points.shape[-1] == 2:
            spatial_shapes_wh = [[w, h] for h, w in input_spatial_shapes]
            offset_normalizer = ops.cast(
                ops.convert_to_tensor(spatial_shapes_wh, dtype="float32"),
                sampling_offsets.dtype,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        value = ops.transpose(value, [0, 2, 1])
        value = ops.reshape(
            value, [N, self.n_heads, self.d_model // self.n_heads, Len_in]
        )

        output = _ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def compute_output_spec(self, query_spec, *args, **kwargs):
        return keras.KerasTensor(
            shape=(query_spec.shape[0], query_spec.shape[1], self.d_model),
            dtype=query_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_levels": self.n_levels,
                "n_heads": self.n_heads,
                "n_points": self.n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETRDecoderLayer(layers.Layer):
    """Single decoder layer for RF-DETR with self-attention, deformable
    cross-attention, and feed-forward network.

    Each layer applies:
    1. Multi-head self-attention over object queries.
    2. Multi-scale deformable cross-attention to encoder memory.
    3. Feed-forward network with ReLU activation.

    All sub-blocks use pre-norm residual connections with dropout.

    Args:
        d_model: Integer, model hidden dimension.
        sa_nhead: Integer, number of self-attention heads.
        ca_nhead: Integer, number of cross-attention heads.
        dim_feedforward: Integer, hidden dimension of the FFN.
            Defaults to `2048`.
        dropout: Float, dropout rate for all sub-blocks. Defaults to `0.0`.
        num_feature_levels: Integer, number of multi-scale feature levels.
            Defaults to `1`.
        dec_n_points: Integer, number of sampling points per head in
            deformable cross-attention. Defaults to `4`.
        spatial_shapes: List of `(height, width)` tuples for each feature
            level. Defaults to `None`.
        level_start_index: List of integers for each level's start index
            in the flattened feature sequence. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        - tgt: `(batch_size, num_queries, d_model)`.
        - memory: `(batch_size, total_tokens, d_model)`.
        - query_pos: `(batch_size, num_queries, d_model)`.
        - reference_points: `(batch_size, num_queries, 1, 4)`.

    Output Shape:
        3D tensor: `(batch_size, num_queries, d_model)`.
    """

    def __init__(
        self,
        d_model,
        sa_nhead,
        ca_nhead,
        dim_feedforward=2048,
        dropout=0.0,
        num_feature_levels=1,
        dec_n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sa_nhead = sa_nhead
        self.ca_nhead = ca_nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.num_feature_levels = num_feature_levels
        self.dec_n_points = dec_n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]

        self.self_attn_q = layers.Dense(self.d_model, name="self_attn_q_proj")
        self.self_attn_k = layers.Dense(self.d_model, name="self_attn_k_proj")
        self.self_attn_v = layers.Dense(self.d_model, name="self_attn_v_proj")
        self.self_attn_out = layers.Dense(self.d_model, name="self_attn_out_proj")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.dropout1 = layers.Dropout(self.dropout_rate)

        self.cross_attn = RFDETRMSDeformableAttention(
            d_model=self.d_model,
            n_levels=self.num_feature_levels,
            n_heads=self.ca_nhead,
            n_points=self.dec_n_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            name="cross_attn",
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.dropout2 = layers.Dropout(self.dropout_rate)

        self.linear1 = layers.Dense(self.dim_feedforward, name="linear1")
        self.linear2 = layers.Dense(self.d_model, name="linear2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="norm3")
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.dropout_ffn = layers.Dropout(self.dropout_rate)

    def _self_attention(self, q, k, v, training=None):
        batch = ops.shape(q)[0]
        seq_len = ops.shape(q)[1]
        head_dim = self.d_model // self.sa_nhead

        q = self.self_attn_q(q)
        k = self.self_attn_k(k)
        v = self.self_attn_v(v)

        q = ops.reshape(q, [batch, seq_len, self.sa_nhead, head_dim])
        k = ops.reshape(k, [batch, seq_len, self.sa_nhead, head_dim])
        v = ops.reshape(v, [batch, seq_len, self.sa_nhead, head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        scale = ops.cast(head_dim, q.dtype) ** -0.5
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * scale
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, [batch, seq_len, self.d_model])
        return self.self_attn_out(out)

    def call(
        self,
        tgt,
        memory,
        query_pos,
        reference_points,
        memory_key_padding_mask=None,
        training=None,
    ):
        q = k = tgt + query_pos
        v = tgt

        tgt2 = self._self_attention(q, k, v, training=training)
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            tgt + query_pos,
            reference_points,
            memory,
            input_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)

        tgt2 = self.linear1(tgt)
        tgt2 = ops.relu(tgt2)
        tgt2 = self.dropout_ffn(tgt2, training=training)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)
        return tgt

    def compute_output_spec(self, tgt_spec, *args, **kwargs):
        d = self.d_model
        shape = (None, d)
        for layer in [
            self.self_attn_q,
            self.self_attn_k,
            self.self_attn_v,
            self.self_attn_out,
            self.norm1,
            self.linear1,
        ]:
            if not layer.built:
                layer.build(shape)
        if not self.linear2.built:
            self.linear2.build((None, self.dim_feedforward))
        if not self.norm2.built:
            self.norm2.build(shape)
        if not self.norm3.built:
            self.norm3.build(shape)
        ca = self.cross_attn
        for sub in [
            ca.sampling_offsets,
            ca.attention_weights_proj,
            ca.value_proj,
            ca.output_proj,
        ]:
            if not sub.built:
                sub.build(shape)
        return keras.KerasTensor(
            shape=tgt_spec.shape,
            dtype=tgt_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "sa_nhead": self.sa_nhead,
                "ca_nhead": self.ca_nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "num_feature_levels": self.num_feature_levels,
                "dec_n_points": self.dec_n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETRLearnedEmbedding(layers.Layer):
    """Learnable embedding table that broadcasts to the input batch size.

    Holds a single weight matrix of shape `(num_embeddings, embedding_dim)` and
    replicates it across the batch dimension at call time. Used for query
    feature embeddings and reference point embeddings in RF-DETR.

    Args:
        num_embeddings: Integer, number of embedding vectors (e.g., num_queries).
        embedding_dim: Integer, dimensionality of each embedding vector.
        initializer: String or initializer, weight initializer.
            Defaults to `"zeros"`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        Any tensor (only the batch dimension is used).

    Output Shape:
        3D tensor: `(batch_size, num_embeddings, embedding_dim)`.
    """

    def __init__(self, num_embeddings, embedding_dim, initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.initializer = initializer

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=self.initializer,
        )
        self.built = True

    def call(self, x):
        batch_size = ops.shape(x)[0]
        return ops.repeat(ops.expand_dims(self.weight, axis=0), batch_size, axis=0)

    def compute_output_spec(self, input_spec, **kwargs):
        return keras.KerasTensor(
            shape=(None, self.num_embeddings, self.embedding_dim),
            dtype=input_spec.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "initializer": self.initializer,
            }
        )
        return config
