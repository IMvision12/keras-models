import math

import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2SinePositionEmbedding(layers.Layer):
    """2-D sine-cosine positional encoding for FPN feature maps.

    Generates dense positional encodings using sine and cosine
    functions, similar to the attention-is-all-you-need positional
    encoding generalised to 2-D grids. The coordinate grid is
    normalized to ``[0, scale]`` and encoded via alternating sine
    and cosine functions at different frequencies.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        num_pos_feats: Integer, number of positional features per
            spatial dimension. The output channel dimension is
            ``2 * num_pos_feats``. Defaults to ``128``.
        temperature: Integer, temperature scaling for the sine/cosine
            frequencies. Defaults to ``10000``.
        normalize: Boolean, whether to normalize the coordinate grid
            to ``[0, scale]``. Defaults to ``True``.
        scale: Float or ``None``, scaling factor applied to the
            normalised coordinates. If ``None``, defaults to
            ``2 * pi``. Defaults to ``None``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, H, W, C)``.

    Output Shape:
        4D tensor: ``(1, 2 * num_pos_feats, H, W)``.
    """

    def __init__(
        self,
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if scale is None:
            scale = 2 * math.pi
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

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
class SAM2MultiScaleAttention(layers.Layer):
    """Multi-head attention with optional query pooling for Hiera.

    Standard multi-head attention where the query can optionally be
    spatially downsampled via max-pooling at stage transitions. This
    enables efficient hierarchical feature extraction in the Hiera
    backbone by reducing the spatial resolution of queries while
    maintaining full-resolution keys and values.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_
        - `Hiera <https://arxiv.org/abs/2306.00989>`_

    Args:
        dim: Integer, input embedding dimension.
        dim_out: Integer, output embedding dimension.
        num_heads: Integer, number of attention heads.
        query_stride: Integer or ``None``. When set, applies max
            pooling with this stride to the query before attention.
            Defaults to ``None``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, H, W, dim)``.

    Output Shape:
        4D tensor: ``(batch_size, H', W', dim_out)`` where ``H'``
        and ``W'`` are ``H // query_stride`` and ``W // query_stride``
        when query pooling is applied.
    """

    def __init__(self, dim, dim_out, num_heads, query_stride=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.query_stride = query_stride
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.qkv = layers.Dense(self.dim_out * 3, use_bias=True, name="qkv")
        self.qkv.build(input_shape)
        self.proj = layers.Dense(self.dim_out, use_bias=True, name="proj")
        self.proj.build((*input_shape[:-1], self.dim_out))
        if self.query_stride is not None:
            self._q_pool = layers.MaxPool2D(
                pool_size=self.query_stride,
                strides=self.query_stride,
                name="q_pool",
            )
        self.built = True

    def call(self, hidden_states):
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(
            qkv,
            (batch_size, -1, 3, self.num_heads, self.head_dim),
        )

        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        if self.query_stride is not None:
            q = ops.reshape(
                q, (batch_size, height, width, self.num_heads * self.head_dim)
            )
            q = self._q_pool(q)
            new_h = ops.shape(q)[1]
            new_w = ops.shape(q)[2]
            q = ops.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))
        else:
            new_h = height
            new_w = width

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q * self.scale, ops.transpose(k, (0, 1, 3, 2)))
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, new_h, new_w, self.dim_out))

        attn_output = self.proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "query_stride": self.query_stride,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2MultiScaleBlock(layers.Layer):
    """Hiera transformer block with windowed or global attention.

    Pre-norm transformer block supporting window-partitioned
    attention, global attention, and optional query pooling at stage
    transitions. When the input and output dimensions differ (stage
    boundary), a linear projection is applied to the residual path.
    The block consists of layer normalization, multi-scale attention,
    another layer normalization, and a two-layer MLP with GELU
    activation.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_
        - `Hiera <https://arxiv.org/abs/2306.00989>`_

    Args:
        dim: Integer, input channel dimension.
        dim_out: Integer, output channel dimension.
        num_heads: Integer, number of attention heads.
        mlp_ratio: Float, expansion ratio for the MLP hidden
            dimension. Defaults to ``4.0``.
        window_size: Integer, window size for windowed attention.
            ``0`` means global attention. Defaults to ``0``.
        query_stride: Integer or ``None``. When set, applies query
            pooling at this stride. Defaults to ``None``.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, H, W, dim)``.

    Output Shape:
        4D tensor: ``(batch_size, H', W', dim_out)``.
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        window_size=0,
        query_stride=None,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.query_stride = query_stride
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(input_shape)
        self.attn = SAM2MultiScaleAttention(
            self.dim,
            self.dim_out,
            self.num_heads,
            query_stride=self.query_stride,
            name="attn",
        )
        self.attn.build(input_shape)
        mlp_dim = int(self.dim_out * self.mlp_ratio)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build((*input_shape[:-1], self.dim_out))
        self.mlp_lin1 = layers.Dense(mlp_dim, name="mlp_proj_in")
        self.mlp_lin1.build((*input_shape[:-1], self.dim_out))
        self.mlp_lin2 = layers.Dense(self.dim_out, name="mlp_proj_out")
        self.mlp_lin2.build((*input_shape[:-1], mlp_dim))

        if self.dim != self.dim_out:
            self.proj = layers.Dense(self.dim_out, name="proj")
            self.proj.build(input_shape)

        if self.query_stride is not None:
            self._residual_pool = layers.MaxPool2D(
                pool_size=self.query_stride,
                strides=self.query_stride,
                name="residual_pool",
            )

        self.built = True

    def _window_partition(self, hidden_states, window_size):
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            hidden_states = ops.pad(
                hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
            )

        padded_h = height + pad_h
        padded_w = width + pad_w

        hidden_states = ops.reshape(
            hidden_states,
            (
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ),
        )
        hidden_states = ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5))
        hidden_states = ops.reshape(
            hidden_states, (-1, window_size, window_size, channels)
        )
        return hidden_states, (padded_h, padded_w)

    def _window_unpartition(self, windows, window_size, pad_hw, original_hw):
        padded_h, padded_w = pad_hw
        height, width = original_hw
        num_windows_h = padded_h // window_size
        num_windows_w = padded_w // window_size
        channels = ops.shape(windows)[-1]
        batch_size = ops.shape(windows)[0] // (num_windows_h * num_windows_w)

        x = ops.reshape(
            windows,
            (
                batch_size,
                num_windows_h,
                num_windows_w,
                window_size,
                window_size,
                channels,
            ),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, padded_h, padded_w, channels))
        x = x[:, :height, :width, :]
        return x

    def call(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.dim != self.dim_out:
            residual = self.proj(hidden_states)
            if self.query_stride is not None:
                residual = self._residual_pool(residual)

        window_size = self.window_size
        if window_size > 0:
            H = ops.shape(hidden_states)[1]
            W = ops.shape(hidden_states)[2]
            hidden_states, pad_hw = self._window_partition(hidden_states, window_size)

        hidden_states = self.attn(hidden_states)

        if self.query_stride is not None and window_size > 0:
            window_size = window_size // self.query_stride
            H_new = ops.shape(residual)[1]
            W_new = ops.shape(residual)[2]

            pad_h = (window_size - H_new % window_size) % window_size
            pad_w = (window_size - W_new % window_size) % window_size
            pad_hw = (H_new + pad_h, W_new + pad_w)
            H = H_new
            W = W_new

        if window_size > 0:
            hidden_states = self._window_unpartition(
                hidden_states, window_size, pad_hw, (H, W)
            )

        hidden_states = residual + hidden_states

        ln_out = self.layer_norm2(hidden_states)
        mlp_out = self.mlp_lin1(ln_out)
        mlp_out = ops.nn.gelu(mlp_out, approximate=False)
        mlp_out = self.mlp_lin2(mlp_out)
        hidden_states = hidden_states + mlp_out

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "window_size": self.window_size,
                "query_stride": self.query_stride,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2PositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding for 2-D coordinates.

    Projects normalised coordinates through a random (frozen)
    Gaussian matrix, then applies sine and cosine to produce the
    final encoding. Used by the prompt encoder for point and box
    prompts and by the model for image-wide positional embeddings.
    The random Gaussian matrix is initialized once and remains
    frozen during training.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        num_pos_feats: Integer, half the output feature dimension.
            Defaults to ``128``.
        scale: Float, standard deviation of the random Gaussian
            initialisation. Defaults to ``1.0``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Tensor of shape ``(..., 2)`` with normalized coordinates.

    Output Shape:
        Tensor of shape ``(..., 2 * num_pos_feats)``.
    """

    def __init__(self, num_pos_feats=128, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )
        self.built = True

    def call(self, coordinates):
        coordinates = 2 * coordinates - 1
        coordinates = ops.cast(coordinates, dtype=self.positional_embedding.dtype)
        coordinates = ops.matmul(coordinates, self.positional_embedding)
        coordinates = 2 * np.pi * coordinates
        return ops.concatenate([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2ImagePositionalEmbeddings(layers.Layer):
    """Grid-based positional embeddings for the image feature map.

    Builds a normalized ``[0, 1]`` coordinate grid over the image
    embedding spatial dimensions and encodes it with the shared
    random Fourier feature layer (``SAM2PositionalEmbedding``). The
    resulting tensor provides fixed positional information that is
    added to the keys in the mask decoder's cross-attention layers.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        image_embedding_size: Integer, spatial size of the image
            embedding grid (both height and width).
        shared_embedding: A ``SAM2PositionalEmbedding`` layer
            instance used to encode the coordinate grid. Shared
            with the prompt encoder.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Dummy input tensor (not used for computation).

    Output Shape:
        4D tensor: ``(1, image_embedding_size, image_embedding_size,
        2 * num_pos_feats)``.
    """

    def __init__(self, image_embedding_size, shared_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self.shared_embedding = shared_embedding

    def call(self, inputs):
        size = self.image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        coords = ops.stack([x_embed, y_embed], axis=-1)
        pe = self.shared_embedding(coords)
        pe = ops.expand_dims(pe, axis=0)
        return pe

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_embedding_size": self.image_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2PromptEncoderLayer(layers.Layer):
    """Encodes sparse and dense prompts for SAM2 mask prediction.

    Sparse prompts (points, boxes) are encoded using Fourier
    positional encoding with learned type embeddings that distinguish
    foreground points, background points, and box corners. Dense
    prompts (masks) are encoded through a small convolutional network.
    When no mask prompt is provided, a learned ``no_mask_embed`` is
    broadcast to the image embedding spatial size.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        hidden_size: Integer, embedding dimension for prompt tokens.
            Defaults to ``256``.
        image_embedding_size: Integer, spatial size of the image
            embeddings from the vision encoder.
            Defaults to ``64``.
        image_size: Integer, input image spatial size.
            Defaults to ``1024``.
        num_point_embeddings: Integer, number of learned point-type
            embeddings. Defaults to ``4``.
        shared_embedding: A ``SAM2PositionalEmbedding`` layer
            instance shared with the image positional embedding
            generator.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        List of two tensors:
        - ``input_points``: ``(batch_size, num_prompts, num_points, 2)``
        - ``input_labels``: ``(batch_size, num_prompts, num_points)``

    Output Shape:
        Dictionary with:
        - ``"sparse_embeddings"``: ``(batch_size, num_prompts,
          num_points + 1, hidden_size)``
        - ``"dense_embeddings"``: ``(batch_size, image_embedding_size,
          image_embedding_size, hidden_size)``
    """

    def __init__(
        self,
        hidden_size=256,
        image_embedding_size=64,
        image_size=1024,
        num_point_embeddings=4,
        shared_embedding=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.num_point_embeddings = num_point_embeddings
        self.shared_embedding = shared_embedding

    def build(self, input_shape):
        self.point_embeddings = []
        for i in range(self.num_point_embeddings):
            w = self.add_weight(
                name=f"point_embed_{i}",
                shape=(1, self.hidden_size),
                initializer="zeros",
            )
            self.point_embeddings.append(w)
        self.not_a_point_embed = self.add_weight(
            name="not_a_point_embed",
            shape=(1, self.hidden_size),
            initializer="zeros",
        )
        self.no_mask_embed = self.add_weight(
            name="no_mask_embed",
            shape=(1, self.hidden_size),
            initializer="zeros",
        )
        self.built = True

    def _embed_points(self, points, labels, pad=True):
        points = points + 0.5
        if pad:
            points = ops.pad(points, [[0, 0], [0, 0], [0, 1], [0, 0]])
            labels = ops.pad(labels, [[0, 0], [0, 0], [0, 1]], constant_values=-1)
        point_embedding = self.shared_embedding(
            points / ops.cast(self.image_size, dtype=points.dtype)
        )

        point_embedding = ops.where(
            ops.expand_dims(labels, axis=-1) == -1,
            ops.broadcast_to(self.not_a_point_embed, ops.shape(point_embedding)),
            point_embedding,
        )
        point_embedding = ops.where(
            ops.expand_dims(labels, axis=-1) != -10,
            point_embedding,
            ops.zeros_like(point_embedding),
        )

        for i in range(self.num_point_embeddings):
            mask = ops.expand_dims(
                ops.cast(labels == i, dtype=point_embedding.dtype), axis=-1
            )
            point_embedding = point_embedding + mask * ops.broadcast_to(
                self.point_embeddings[i], ops.shape(point_embedding)
            )

        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        batch_size = ops.shape(boxes)[0]
        num_boxes = ops.shape(boxes)[1]
        coords = ops.reshape(boxes, (batch_size, num_boxes, 2, 2))
        coords = ops.pad(coords, [[0, 0], [0, 0], [0, 1], [0, 0]])
        corner_embedding = self.shared_embedding(
            coords / ops.cast(self.image_size, dtype=coords.dtype)
        )
        corner_embedding_0 = corner_embedding[:, :, 0:1, :] + ops.broadcast_to(
            self.point_embeddings[2],
            ops.shape(corner_embedding[:, :, 0:1, :]),
        )
        corner_embedding_1 = corner_embedding[:, :, 1:2, :] + ops.broadcast_to(
            self.point_embeddings[3],
            ops.shape(corner_embedding[:, :, 1:2, :]),
        )
        corner_embedding_2 = ops.broadcast_to(
            self.not_a_point_embed,
            ops.shape(corner_embedding[:, :, 2:3, :]),
        )
        corner_embedding = ops.concatenate(
            [corner_embedding_0, corner_embedding_1, corner_embedding_2],
            axis=2,
        )
        return corner_embedding

    def call(self, inputs):
        input_points, input_labels = inputs[0], inputs[1]
        sparse_embeddings = self._embed_points(input_points, input_labels)

        dense_embeddings = ops.broadcast_to(
            ops.reshape(
                self.no_mask_embed,
                (1, 1, 1, self.hidden_size),
            ),
            (
                ops.shape(input_points)[0],
                self.image_embedding_size,
                self.image_embedding_size,
                self.hidden_size,
            ),
        )

        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "image_embedding_size": self.image_embedding_size,
                "image_size": self.image_size,
                "num_point_embeddings": self.num_point_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2TwoWayAttention(layers.Layer):
    """Attention layer for the two-way mask decoder transformer.

    Supports downsampling the internal dimension by a configurable
    rate, enabling efficient cross-attention between prompt tokens
    and image embeddings. The internal dimension is computed as
    ``hidden_size // downsample_rate``, reducing memory and
    computation for cross-attention operations.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        hidden_size: Integer, input/output embedding dimension.
            Defaults to ``256``.
        num_heads: Integer, number of attention heads.
            Defaults to ``8``.
        downsample_rate: Integer, factor by which the internal
            dimension is reduced. Defaults to ``1``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        Three tensors:
        - ``query``: ``(batch_size, point_batch, num_queries, hidden_size)``
        - ``key``: ``(batch_size, point_batch, num_keys, hidden_size)``
        - ``value``: ``(batch_size, point_batch, num_values, hidden_size)``

    Output Shape:
        Tensor of shape ``(batch_size, point_batch, num_queries,
        hidden_size)``.
    """

    def __init__(self, hidden_size=256, num_heads=8, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_size // downsample_rate
        self.head_dim = self.internal_dim // num_heads

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.q_proj.build(input_shape)
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.k_proj.build(input_shape)
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.v_proj.build(input_shape)
        self.out_proj = layers.Dense(self.hidden_size, name="out_proj")
        self.out_proj.build((*input_shape[:-1], self.internal_dim))
        self.built = True

    def call(self, query, key, value, attention_similarity=None):
        batch_size = ops.shape(query)[0]
        point_batch = ops.shape(query)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        new_shape = (
            batch_size * point_batch,
            -1,
            self.num_heads,
            self.head_dim,
        )
        q = ops.reshape(q, new_shape)
        k = ops.reshape(k, new_shape)
        v = ops.reshape(v, new_shape)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        scale = self.head_dim**-0.5
        attn_weights = ops.matmul(q * scale, ops.transpose(k, (0, 1, 3, 2)))

        if attention_similarity is not None:
            attn_weights = attn_weights + attention_similarity

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output,
            (batch_size, point_batch, -1, self.internal_dim),
        )
        return self.out_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2MaskDecoderLayer(layers.Layer):
    """Mask decoder with two-way transformer and object scoring.

    Jointly attends between prompt tokens and image embeddings using
    a lightweight two-way transformer, then predicts segmentation
    masks via hypernetwork MLPs and quality scores via an IoU
    prediction head. Extends SAM v1 with object score prediction and
    high-resolution feature skip connections. The decoder first
    concatenates learned object score, IoU, and mask tokens with the
    sparse prompt embeddings, then alternates self-attention on
    tokens, cross-attention from tokens to image features, an MLP
    block, and cross-attention from image features back to tokens.
    After the transformer, image features are upscaled via two
    transposed convolutions with high-resolution skip connections,
    and per-mask predictions are generated by hypernetwork MLPs.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_

    Args:
        hidden_size: Integer, embedding dimension.
            Defaults to ``256``.
        num_hidden_layers: Integer, number of two-way transformer
            blocks. Defaults to ``2``.
        num_attention_heads: Integer, number of attention heads.
            Defaults to ``8``.
        mlp_dim: Integer, hidden dimension of the transformer MLP.
            Defaults to ``2048``.
        num_multimask_outputs: Integer, number of mask outputs
            beyond the single-mask token. Defaults to ``3``.
        iou_head_depth: Integer, number of layers in the IoU
            prediction head. Defaults to ``3``.
        iou_head_hidden_dim: Integer, hidden dimension of the IoU
            prediction head. Defaults to ``256``.
        attention_downsample_rate: Integer, downsample rate for
            cross-attention. Defaults to ``2``.
        layer_norm_eps: Float, epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        List of six tensors:
        - ``image_embeddings``: ``(batch_size, H, W, hidden_size)``
        - ``image_pe``: ``(batch_size, H, W, hidden_size)``
        - ``sparse_embeddings``: ``(batch_size, num_prompts,
          num_tokens, hidden_size)``
        - ``dense_embeddings``: ``(batch_size, H, W, hidden_size)``
        - ``high_res_feat_s0``: ``(batch_size, 4*H, 4*W, hidden_size)``
        - ``high_res_feat_s1``: ``(batch_size, 2*H, 2*W, hidden_size)``

    Output Shape:
        Dictionary with:
        - ``"pred_masks"``: ``(batch_size, num_prompts,
          num_mask_tokens, 4*H, 4*W)``
        - ``"iou_scores"``: ``(batch_size, num_prompts,
          num_mask_tokens)``
        - ``"object_score_logits"``: ``(batch_size, num_prompts, 1)``
    """

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        mlp_dim=2048,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        attention_downsample_rate=2,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        hs = self.hidden_size
        nm = self.num_mask_tokens
        ds = self.attention_downsample_rate

        self.obj_score_token = self.add_weight(
            name="obj_score_token",
            shape=(1, hs),
            initializer="zeros",
        )
        self.iou_token = self.add_weight(
            name="iou_token",
            shape=(1, hs),
            initializer="zeros",
        )
        self.mask_tokens = self.add_weight(
            name="mask_tokens",
            shape=(nm, hs),
            initializer="zeros",
        )

        # Two-way transformer layers
        self.transformer_self_attns = []
        self.transformer_layer_norm1s = []
        self.transformer_cross_attn_token_to_images = []
        self.transformer_layer_norm2s = []
        self.transformer_mlp_lin1s = []
        self.transformer_mlp_lin2s = []
        self.transformer_layer_norm3s = []
        self.transformer_cross_attn_image_to_tokens = []
        self.transformer_layer_norm4s = []

        dummy_shape = (None, None, None, hs)
        for i in range(self.num_hidden_layers):
            self_attn = SAM2TwoWayAttention(
                hs,
                self.num_attention_heads,
                downsample_rate=1,
                name=f"transformer_self_attns_{i}",
            )
            self_attn.build(dummy_shape)
            self.transformer_self_attns.append(self_attn)

            ln1 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm1s_{i}",
            )
            ln1.build(dummy_shape)
            self.transformer_layer_norm1s.append(ln1)

            cross_t2i = SAM2TwoWayAttention(
                hs,
                self.num_attention_heads,
                downsample_rate=ds,
                name=f"transformer_cross_attn_token_to_images_{i}",
            )
            cross_t2i.build(dummy_shape)
            self.transformer_cross_attn_token_to_images.append(cross_t2i)

            ln2 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm2s_{i}",
            )
            ln2.build(dummy_shape)
            self.transformer_layer_norm2s.append(ln2)

            mlp1 = layers.Dense(self.mlp_dim, name=f"transformer_mlp_lin1s_{i}")
            mlp1.build(dummy_shape)
            self.transformer_mlp_lin1s.append(mlp1)

            mlp2 = layers.Dense(hs, name=f"transformer_mlp_lin2s_{i}")
            mlp2.build((*dummy_shape[:-1], self.mlp_dim))
            self.transformer_mlp_lin2s.append(mlp2)

            ln3 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm3s_{i}",
            )
            ln3.build(dummy_shape)
            self.transformer_layer_norm3s.append(ln3)

            cross_i2t = SAM2TwoWayAttention(
                hs,
                self.num_attention_heads,
                downsample_rate=ds,
                name=f"transformer_cross_attn_image_to_tokens_{i}",
            )
            cross_i2t.build(dummy_shape)
            self.transformer_cross_attn_image_to_tokens.append(cross_i2t)

            ln4 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm4s_{i}",
            )
            ln4.build(dummy_shape)
            self.transformer_layer_norm4s.append(ln4)

        self.final_attn_token_to_image = SAM2TwoWayAttention(
            hs,
            self.num_attention_heads,
            downsample_rate=ds,
            name="final_attn_token_to_image",
        )
        self.final_attn_token_to_image.build(dummy_shape)
        self.layer_norm_final_attn = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="layer_norm_final_attn",
        )
        self.layer_norm_final_attn.build(dummy_shape)

        # Upscale path
        self.upscale_conv1 = layers.Conv2DTranspose(
            hs // 4, kernel_size=2, strides=2, name="upscale_conv1"
        )
        self.upscale_conv1.build((None, None, None, hs))
        self.upscale_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="upscale_layer_norm"
        )
        self.upscale_layer_norm.build((None, None, None, hs // 4))
        self.upscale_conv2 = layers.Conv2DTranspose(
            hs // 8, kernel_size=2, strides=2, name="upscale_conv2"
        )
        self.upscale_conv2.build((None, None, None, hs // 4))

        # High-res skip convolutions
        self.conv_s0 = layers.Conv2D(hs // 8, kernel_size=1, name="conv_s0")
        self.conv_s0.build((None, None, None, hs))
        self.conv_s1 = layers.Conv2D(hs // 4, kernel_size=1, name="conv_s1")
        self.conv_s1.build((None, None, None, hs))

        # Hypernetwork MLPs
        out_dim = hs // 8
        self.output_hypernetworks_mlps_proj_ins = []
        self.output_hypernetworks_mlps_hidden_layers = []
        self.output_hypernetworks_mlps_proj_outs = []
        for i in range(nm):
            pi = layers.Dense(hs, name=f"output_hypernetworks_mlps_proj_ins_{i}")
            pi.build((None, None, hs))
            self.output_hypernetworks_mlps_proj_ins.append(pi)

            hl = layers.Dense(hs, name=f"output_hypernetworks_mlps_layers_{i}")
            hl.build((None, None, hs))
            self.output_hypernetworks_mlps_hidden_layers.append(hl)

            po = layers.Dense(
                out_dim,
                name=f"output_hypernetworks_mlps_proj_outs_{i}",
            )
            po.build((None, None, hs))
            self.output_hypernetworks_mlps_proj_outs.append(po)

        # IoU prediction head (with sigmoid)
        self.iou_head_proj_in = layers.Dense(
            self.iou_head_hidden_dim, name="iou_head_proj_in"
        )
        self.iou_head_proj_in.build((None, None, hs))
        self.iou_head_hidden_layers = []
        for i in range(self.iou_head_depth - 2):
            hl = layers.Dense(
                self.iou_head_hidden_dim,
                name=f"iou_head_hidden_layers_{i}",
            )
            hl.build((None, None, self.iou_head_hidden_dim))
            self.iou_head_hidden_layers.append(hl)
        self.iou_head_proj_out = layers.Dense(nm, name="iou_head_proj_out")
        self.iou_head_proj_out.build((None, None, self.iou_head_hidden_dim))

        # Object score prediction head
        self.obj_score_proj_in = layers.Dense(hs, name="obj_score_proj_in")
        self.obj_score_proj_in.build((None, None, hs))
        self.obj_score_hidden_layers = []
        for i in range(1):
            hl = layers.Dense(hs, name=f"obj_score_hidden_layers_{i}")
            hl.build((None, None, hs))
            self.obj_score_hidden_layers.append(hl)
        self.obj_score_proj_out = layers.Dense(1, name="obj_score_proj_out")
        self.obj_score_proj_out.build((None, None, hs))

        self.built = True

    def call(self, inputs):
        (
            image_embeddings,
            image_pe,
            sparse_embeddings,
            dense_embeddings,
            high_res_feat_s0,
            high_res_feat_s1,
        ) = inputs

        batch_size = ops.shape(image_embeddings)[0]
        num_channels = ops.shape(image_embeddings)[3]
        height = ops.shape(image_embeddings)[1]
        width = ops.shape(image_embeddings)[2]
        point_batch_size = ops.shape(sparse_embeddings)[1]

        # Build output tokens: [obj_score, iou, mask_0, ..., mask_N]
        output_tokens = ops.concatenate(
            [self.obj_score_token, self.iou_token, self.mask_tokens],
            axis=0,
        )
        output_tokens = ops.broadcast_to(
            ops.reshape(
                output_tokens,
                (1, 1, 2 + self.num_mask_tokens, self.hidden_size),
            ),
            (
                batch_size,
                point_batch_size,
                2 + self.num_mask_tokens,
                self.hidden_size,
            ),
        )
        tokens = ops.concatenate([output_tokens, sparse_embeddings], axis=2)

        # Add dense prompt to image embeddings
        image_embeddings_with_dense = image_embeddings + dense_embeddings

        # Flatten to (B, H*W, C) then broadcast to (B, P, H*W, C)
        ie_flat = ops.reshape(
            image_embeddings_with_dense,
            (batch_size, height * width, num_channels),
        )
        ie_flat = ops.expand_dims(ie_flat, axis=1)
        ie_flat = ops.broadcast_to(
            ie_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        # Broadcast PE to batch size, flatten to (B, H*W, C),
        # then broadcast to (B, P, H*W, C)
        image_pe = ops.broadcast_to(image_pe, (batch_size, height, width, num_channels))
        pe_flat = ops.reshape(image_pe, (batch_size, height * width, num_channels))
        pe_flat = ops.expand_dims(pe_flat, axis=1)
        pe_flat = ops.broadcast_to(
            pe_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        queries = tokens
        keys = ie_flat

        # Two-way transformer
        for i in range(self.num_hidden_layers):
            if i == 0:
                queries = self.transformer_self_attns[i](queries, queries, queries)
            else:
                queries_with_pe = queries + tokens
                attn_out = self.transformer_self_attns[i](
                    queries_with_pe, queries_with_pe, queries
                )
                queries = queries + attn_out
            queries = self.transformer_layer_norm1s[i](queries)

            queries_with_pe = queries + tokens
            keys_with_pe = keys + pe_flat

            attn_out = self.transformer_cross_attn_token_to_images[i](
                queries_with_pe, keys_with_pe, keys
            )
            queries = queries + attn_out
            queries = self.transformer_layer_norm2s[i](queries)

            mlp_out = self.transformer_mlp_lin1s[i](queries)
            mlp_out = ops.nn.relu(mlp_out)
            mlp_out = self.transformer_mlp_lin2s[i](mlp_out)
            queries = queries + mlp_out
            queries = self.transformer_layer_norm3s[i](queries)

            queries_with_pe = queries + tokens
            keys_with_pe = keys + pe_flat

            attn_out = self.transformer_cross_attn_image_to_tokens[i](
                keys_with_pe, queries_with_pe, queries
            )
            keys = keys + attn_out
            keys = self.transformer_layer_norm4s[i](keys)

        # Final attention
        queries_with_pe = queries + tokens
        keys_with_pe = keys + pe_flat
        attn_out = self.final_attn_token_to_image(queries_with_pe, keys_with_pe, keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        # Extract token outputs (obj_score=0, iou=1, masks=2:)
        iou_token_out = queries[:, :, 1, :]
        mask_tokens_out = queries[:, :, 2 : 2 + self.num_mask_tokens, :]

        # Upscale with high-res skip connections
        keys_spatial = ops.reshape(
            keys,
            (batch_size * point_batch_size, height, width, num_channels),
        )

        # High-res features are NHWC from FPN (no transpose needed)
        feat_s1 = ops.expand_dims(high_res_feat_s1, axis=1)
        feat_s1 = ops.broadcast_to(
            feat_s1,
            (
                batch_size,
                point_batch_size,
                ops.shape(feat_s1)[2],
                ops.shape(feat_s1)[3],
                ops.shape(feat_s1)[4],
            ),
        )
        feat_s1 = ops.reshape(
            feat_s1,
            (-1, ops.shape(feat_s1)[2], ops.shape(feat_s1)[3], ops.shape(feat_s1)[4]),
        )

        feat_s0 = ops.expand_dims(high_res_feat_s0, axis=1)
        feat_s0 = ops.broadcast_to(
            feat_s0,
            (
                batch_size,
                point_batch_size,
                ops.shape(feat_s0)[2],
                ops.shape(feat_s0)[3],
                ops.shape(feat_s0)[4],
            ),
        )
        feat_s0 = ops.reshape(
            feat_s0,
            (-1, ops.shape(feat_s0)[2], ops.shape(feat_s0)[3], ops.shape(feat_s0)[4]),
        )

        upscaled = self.upscale_conv1(keys_spatial) + self.conv_s1(feat_s1)
        upscaled = ops.nn.gelu(self.upscale_layer_norm(upscaled), approximate=False)
        upscaled = ops.nn.gelu(
            self.upscale_conv2(upscaled) + self.conv_s0(feat_s0), approximate=False
        )

        up_shape = ops.shape(upscaled)
        up_h = up_shape[1]
        up_w = up_shape[2]
        up_c = up_shape[3]

        # Hypernetwork MLPs
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            h = self.output_hypernetworks_mlps_proj_ins[i](mask_tokens_out[:, :, i, :])
            h = ops.nn.relu(h)
            h = ops.nn.relu(self.output_hypernetworks_mlps_hidden_layers[i](h))
            h = self.output_hypernetworks_mlps_proj_outs[i](h)
            hyper_in_list.append(h)
        hyper_in = ops.stack(hyper_in_list, axis=2)

        upscaled_flat = ops.reshape(
            upscaled,
            (batch_size, point_batch_size, up_h * up_w, up_c),
        )
        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 1, 3, 2)))
        masks = ops.reshape(
            masks,
            (batch_size, point_batch_size, -1, up_h, up_w),
        )

        # IoU prediction (sigmoid)
        iou_out = self.iou_head_proj_in(iou_token_out)
        iou_out = ops.nn.relu(iou_out)
        for hl in self.iou_head_hidden_layers:
            iou_out = ops.nn.relu(hl(iou_out))
        iou_pred = ops.sigmoid(self.iou_head_proj_out(iou_out))

        # Object score prediction
        obj_score = self.obj_score_proj_in(queries[:, :, 0, :])
        obj_score = ops.nn.relu(obj_score)
        for hl in self.obj_score_hidden_layers:
            obj_score = ops.nn.relu(hl(obj_score))
        object_score_logits = self.obj_score_proj_out(obj_score)

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "object_score_logits": object_score_logits,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "mlp_dim": self.mlp_dim,
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "attention_downsample_rate": self.attention_downsample_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kmodels")
class SAM2HieraPositionEmbedding(layers.Layer):
    """Windowed positional embedding for the Hiera backbone.

    Combines a global positional embedding (bicubic-interpolated to
    match the input spatial size) with a tiled window-level
    positional embedding. The global embedding provides coarse
    positional information across the entire feature map, while the
    window embedding adds fine-grained positional information within
    each window. Both embeddings are learned parameters that are
    added to the input features.

    Reference:
        - `SAM 2 <https://arxiv.org/abs/2408.00714>`_
        - `Hiera <https://arxiv.org/abs/2306.00989>`_

    Args:
        hidden_size: Integer, channel dimension.
        spatial_size: Tuple ``(H, W)`` of the feature map.
        window_size: Integer, first-stage window size for the tiled
            window embedding.
        bg_size: Tuple ``(H, W)`` background size for the global
            embedding. Defaults to ``(7, 7)``.
        **kwargs: Additional keyword arguments passed to the
            ``Layer`` class.

    Input Shape:
        4D tensor: ``(batch_size, H, W, hidden_size)``.

    Output Shape:
        4D tensor: ``(batch_size, H, W, hidden_size)``.
    """

    def __init__(
        self, hidden_size, spatial_size, window_size, bg_size=(7, 7), **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.spatial_size = tuple(spatial_size)
        self.window_size = window_size
        self.bg_size = tuple(bg_size)

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.bg_size[0], self.bg_size[1], self.hidden_size),
            initializer="zeros",
        )
        self.pos_embed_window = self.add_weight(
            name="pos_embed_window",
            shape=(1, self.window_size, self.window_size, self.hidden_size),
            initializer="zeros",
        )
        self.built = True

    def call(self, hidden_states):
        # Use concrete spatial_size (known at build time) to avoid
        # symbolic shape issues with JAX backend.
        h, w = self.spatial_size

        pos = ops.image.resize(
            ops.convert_to_tensor(self.pos_embed),
            size=(h, w),
            interpolation="bicubic",
            antialias=False,
        )

        tile_h = h // self.window_size
        tile_w = w // self.window_size
        window_pos = ops.tile(self.pos_embed_window, (1, tile_h, tile_w, 1))
        pos = pos + window_pos

        return hidden_states + pos

    def compute_output_spec(self, hidden_states):
        return keras.KerasTensor(hidden_states.shape, dtype=hidden_states.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "spatial_size": self.spatial_size,
                "window_size": self.window_size,
                "bg_size": self.bg_size,
            }
        )
        return config
