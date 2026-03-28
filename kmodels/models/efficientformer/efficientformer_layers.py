import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kmodels")
class Attention4D(layers.Layer):
    """Multi-head self-attention with learnable relative position bias.

    Computes multi-head attention using a fused QKV projection and adds
    a trainable relative position bias to the attention logits. Used in
    the transformer (1D) stages of EfficientFormer.

    When loading pretrained weights from a different resolution, the
    attention bias table is automatically interpolated via bilinear
    resizing in ``load_own_variables``, following the same approach
    used by ViT's ``AddPositionEmbs``.

    Reference:
    - [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

    Args:
        dim: Integer, total input/output feature dimension. Only used for
            serialization; the actual input dimension is inferred during
            `build`. Defaults to `384`.
        key_dim: Integer, per-head dimension for query and key projections.
            Defaults to `32`.
        num_heads: Integer, number of attention heads.
            Defaults to `8`.
        attn_ratio: Integer, ratio of value dimension to key dimension.
            The per-head value dimension is `key_dim * attn_ratio`.
            Defaults to `4`.
        resolution: Integer, spatial resolution of the feature map. The
            relative position bias table has shape
            `(num_heads, resolution * resolution)`. Defaults to `7`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        3D tensor: `(batch_size, seq_len, dim)` where
        `seq_len = resolution * resolution`.

    Output Shape:
        3D tensor: `(batch_size, seq_len, dim)`.
    """

    def __init__(
        self,
        dim=384,
        key_dim=32,
        num_heads=8,
        attn_ratio=4,
        resolution=7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution

    def build(self, input_shape):
        dim = input_shape[-1]
        qkv_out_dim = self.key_attn_dim * 2 + self.val_attn_dim

        self.qkv_kernel = self.add_weight(
            name="qkv_kernel",
            shape=(dim, qkv_out_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.qkv_bias = self.add_weight(
            name="qkv_bias",
            shape=(qkv_out_dim,),
            initializer="zeros",
            trainable=True,
        )

        self.proj_kernel = self.add_weight(
            name="proj_kernel",
            shape=(self.val_attn_dim, dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.proj_bias = self.add_weight(
            name="proj_bias",
            shape=(dim,),
            initializer="zeros",
            trainable=True,
        )

        resolution = self.resolution
        pos_h = ops.arange(resolution)
        pos_w = ops.arange(resolution)
        grid_h, grid_w = ops.meshgrid(pos_h, pos_w, indexing="ij")
        pos = ops.stack([grid_h, grid_w], axis=0)
        pos = ops.reshape(pos, (2, -1))

        rel_pos = ops.abs(ops.expand_dims(pos, -1) - ops.expand_dims(pos, -2))
        rel_pos = rel_pos[0] * resolution + rel_pos[1]

        self.attention_biases = self.add_weight(
            name="attention_biases",
            shape=(self.num_heads, resolution * resolution),
            initializer="zeros",
            trainable=True,
        )
        self.attention_bias_idxs = ops.cast(rel_pos, "int32")

    def call(self, x):
        B = ops.shape(x)[0]
        N = ops.shape(x)[1]

        qkv = ops.matmul(x, self.qkv_kernel) + self.qkv_bias
        qkv = ops.reshape(qkv, (B, N, self.num_heads, -1))
        qkv = ops.transpose(qkv, (0, 2, 1, 3))

        q = qkv[:, :, :, : self.key_dim]
        k = qkv[:, :, :, self.key_dim : 2 * self.key_dim]
        v = qkv[:, :, :, 2 * self.key_dim :]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        bias = ops.take(self.attention_biases, self.attention_bias_idxs, axis=1)
        attn = attn + bias

        attn = ops.softmax(attn, axis=-1)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, self.val_attn_dim))
        x = ops.matmul(x, self.proj_kernel) + self.proj_bias
        return x

    def load_own_variables(self, store):
        # Infer source resolution from the saved attention_biases shape.
        bias_idx = None
        target_vars = self.trainable_variables + self.non_trainable_variables
        for i, var in enumerate(target_vars):
            if var is self.attention_biases:
                bias_idx = i
                break
        if bias_idx is not None and str(bias_idx) in store:
            source_bias_size = store[str(bias_idx)].shape[-1]
            import math

            source_resolution = int(math.isqrt(source_bias_size))
        else:
            source_resolution = self.resolution

        if source_resolution == self.resolution:
            super().load_own_variables(store)
            return

        target_vars = self.trainable_variables + self.non_trainable_variables
        for i, var in enumerate(target_vars):
            if var is self.attention_biases:
                continue
            var.assign(store[str(i)])

        bias_idx = None
        for i, var in enumerate(target_vars):
            if var is self.attention_biases:
                bias_idx = i
                break

        source_biases = store[str(bias_idx)]
        source_biases = ops.cast(source_biases, dtype="float32")
        source_biases = ops.reshape(
            source_biases,
            (self.num_heads, source_resolution, source_resolution),
        )
        source_biases = ops.expand_dims(source_biases, axis=-1)
        source_biases = ops.image.resize(
            source_biases,
            size=(self.resolution, self.resolution),
            interpolation="bilinear",
            antialias=True,
        )
        source_biases = ops.reshape(
            source_biases,
            (self.num_heads, self.resolution * self.resolution),
        )
        self.attention_biases.assign(source_biases)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.key_attn_dim + self.val_attn_dim,
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "attn_ratio": self.attn_ratio,
                "resolution": self.resolution,
            }
        )
        return config
