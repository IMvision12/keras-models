import math

import keras
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import RF_DETR_MODEL_CONFIG, RF_DETR_WEIGHTS_CONFIG
from .rf_detr_layers import (
    DinoV2Embeddings,
    DinoV2LayerScale,
    RFDETRChannelLayerNorm,
    RFDETRDecoderLayer,
    RFDETRLearnedEmbedding,
)


def _sincos_interleave(x):
    """Interleave sine and cosine components for positional encoding.

    Converts alternating sin/cos components into an interleaved format,
    matching the standard sinusoidal positional encoding format.

    Args:
        x: Input tensor with even indices containing angles for sin,
            odd indices for cos.

    Returns:
        Tensor with interleaved sin/cos values.
    """
    sin_part = ops.sin(x[..., 0::2])
    cos_part = ops.cos(x[..., 1::2])
    stacked = ops.stack([sin_part, cos_part], axis=-1)
    target_shape = [-1 if s is None else s for s in stacked.shape]
    target_shape[-2] = target_shape[-2] * 2
    target_shape.pop()
    return ops.reshape(stacked, target_shape)


def rf_detr_position_embedding_sine(
    x, num_pos_feats=128, temperature=10000, normalize=True
):
    """Generate 2D sinusoidal positional embeddings for feature maps.

    Creates a positional encoding using sine and cosine functions at different
    frequencies, suitable for transformer-based vision models.

    Args:
        x: Input tensor used only for shape reference. Shape is expected to be
            ``(B, H, W, C)`` where H and W determine the spatial dimensions.
        num_pos_feats: Number of positional features per dimension. The output
            will have ``2 * num_pos_feats`` channels. Default 128.
        temperature: Temperature for scaling the frequency bands. Default 10000.
        normalize: Whether to normalize coordinates to [0, 2π]. Default True.

    Returns:
        Positional embedding tensor of shape ``(1, H, W, 2 * num_pos_feats)``.
    """
    scale = 2.0 * math.pi
    shape = ops.shape(x)
    h, w = shape[1], shape[2]

    y_range = ops.cast(ops.arange(h), "float32")
    x_range = ops.cast(ops.arange(w), "float32")

    if normalize:
        eps = 1e-6
        y_range = y_range / (ops.cast(h, "float32") - 1 + eps) * scale
        x_range = x_range / (ops.cast(w, "float32") - 1 + eps) * scale
    else:
        y_range = (y_range + 1) * scale
        x_range = (x_range + 1) * scale

    dim_t = ops.cast(ops.arange(num_pos_feats), "float32")
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_range[:, None] / dim_t[None, :]
    pos_y = y_range[:, None] / dim_t[None, :]

    pos_x_sin = ops.sin(pos_x[:, 0::2])
    pos_x_cos = ops.cos(pos_x[:, 1::2])
    pos_x = ops.reshape(ops.stack([pos_x_sin, pos_x_cos], axis=-1), [w, num_pos_feats])

    pos_y_sin = ops.sin(pos_y[:, 0::2])
    pos_y_cos = ops.cos(pos_y[:, 1::2])
    pos_y = ops.reshape(ops.stack([pos_y_sin, pos_y_cos], axis=-1), [h, num_pos_feats])

    pos_y = ops.expand_dims(pos_y, axis=1)
    pos_y = ops.tile(pos_y, [1, w, 1])
    pos_x = ops.expand_dims(pos_x, axis=0)
    pos_x = ops.tile(pos_x, [h, 1, 1])
    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    pos = ops.expand_dims(pos, axis=0)
    return pos


def rf_detr_gen_sineembed_for_position(pos_tensor, dim=128):
    """Generate sine positional embeddings for bounding box coordinates.

    Creates sinusoidal positional embeddings from normalized coordinate tensors,
    used for query position encoding in the DETR decoder.

    Args:
        pos_tensor: Position tensor of shape ``(..., 2)`` for (x, y) coordinates
            or ``(..., 4)`` for (x, y, w, h) coordinates. Values should be in
            range [0, 1] representing normalized positions.
        dim: Embedding dimension per coordinate axis. Default 128.

    Returns:
        Positional embedding tensor. Shape depends on input:
        - For 2D input: ``(..., 2 * dim)``
        - For 4D input: ``(..., 4 * dim)``

    Raises:
        ValueError: If pos_tensor has last dimension other than 2 or 4.
    """
    scale = 2 * math.pi
    dim_t = ops.cast(ops.arange(dim), "float32")
    dim_t = 10000.0 ** (2 * (dim_t // 2) / dim)

    x_embed = pos_tensor[..., 0:1] * scale
    y_embed = pos_tensor[..., 1:2] * scale

    pos_x = _sincos_interleave(x_embed / dim_t)
    pos_y = _sincos_interleave(y_embed / dim_t)

    if pos_tensor.shape[-1] == 2:
        return ops.concatenate([pos_y, pos_x], axis=-1)
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[..., 2:3] * scale
        h_embed = pos_tensor[..., 3:4] * scale
        pos_w = _sincos_interleave(w_embed / dim_t)
        pos_h = _sincos_interleave(h_embed / dim_t)
        return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)
    else:
        raise ValueError(
            f"pos_tensor last dim must be 2 or 4, got {pos_tensor.shape[-1]}"
        )


def rf_detr_unwindow_features(
    hidden_state, num_h, num_w, num_windows, hidden_size, num_register_tokens
):
    """Convert windowed features back to spatial feature map format.

    Transforms features from window-based representation (used in windowed
    attention) back to a standard spatial layout, removing register tokens
    and reorganizing patches.

    Args:
        hidden_state: Windowed feature tensor from DINOv2 encoder.
        num_h: Number of patches in height dimension.
        num_w: Number of patches in width dimension.
        num_windows: Number of windows along each spatial dimension.
            Windowed attention divides the spatial grid into windows.
        hidden_size: Feature dimension.
        num_register_tokens: Number of DINOv2 register tokens to remove
            from the beginning of the sequence.

    Returns:
        Unwindowed feature tensor of shape ``(B, num_h, num_w, hidden_size)``.
    """
    hidden_state = hidden_state[:, num_register_tokens + 1 :, :]

    if num_windows > 1:
        nw2 = num_windows**2
        shape = ops.shape(hidden_state)
        HW_win = shape[1]
        C = shape[2]

        hidden_state = ops.reshape(hidden_state, [-1, nw2 * HW_win, C])
        h_pw = num_h // num_windows
        w_pw = num_w // num_windows
        hidden_state = ops.reshape(hidden_state, [-1, num_windows, h_pw, w_pw, C])
        hidden_state = ops.transpose(hidden_state, [0, 2, 1, 3, 4])

    hidden_state = ops.reshape(hidden_state, [-1, num_h, num_w, hidden_size])
    return hidden_state


def rf_detr_encoder_output_proposals(memory, spatial_shapes, bbox_reparam=True):
    """Generate encoder output proposals for two-stage DETR initialization.

    Creates initial reference points (proposals) from spatial locations across
    feature levels, used in the two-stage DETR architecture for query selection.

    Args:
        memory: Encoder output features of shape ``(B, total_tokens, C)``.
        spatial_shapes: List of ``(H, W)`` tuples for each feature level's
            spatial dimensions.
        bbox_reparam: If True, use bbox reparameterization (cx, cy, w, h
            format directly). If False, use sigmoid-sigmoid format for
            numerically stable sigmoid inverse. Default True.

    Returns:
        Tuple of:
        - output_memory: Filtered memory features with invalid proposals zeroed.
        - output_proposals: Generated proposals as (cx, cy, w, h) tensors
            in normalized coordinates [0, 1].
    """
    proposals = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        y_range = ops.cast(ops.arange(H_), "float32")
        x_range = ops.cast(ops.arange(W_), "float32")
        grid_y, grid_x = ops.meshgrid(y_range, x_range, indexing="ij")
        grid = ops.stack([grid_x, grid_y], axis=-1)
        grid = ops.reshape(grid, [1, H_ * W_, 2])

        scale = ops.convert_to_tensor([[W_, H_]], dtype="float32")
        scale = ops.reshape(scale, [1, 1, 2])
        grid = (grid + 0.5) / scale

        wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = ops.concatenate([grid, wh], axis=-1)
        proposals.append(proposal)

    output_proposals = ops.concatenate(proposals, axis=1)
    # Broadcast proposals to match memory's batch dimension.
    # Multiply by 0 and sum to get a (B, 1, 1) zero tensor that
    # carries the dynamic batch dimension without slicing.
    batch_zero = ops.sum(memory * 0, axis=(1, 2), keepdims=True)
    output_proposals = batch_zero + output_proposals

    valid = ops.all(
        (output_proposals > 0.01) & (output_proposals < 0.99),
        axis=-1,
        keepdims=True,
    )

    if bbox_reparam:
        output_proposals = ops.where(
            valid, output_proposals, ops.zeros_like(output_proposals)
        )
    else:
        eps = 1e-7
        clamped = ops.clip(output_proposals, eps, 1.0 - eps)
        unsig = ops.log(clamped / (1.0 - clamped))
        inf_val = ops.full_like(unsig, 1e8)
        output_proposals = ops.where(valid, unsig, inf_val)

    output_memory = ops.where(valid, memory, ops.zeros_like(memory))
    return output_memory, output_proposals


def rf_detr_two_stage_refine_refpoints(
    refpoint_embed, refpoint_embed_ts, bbox_reparam=True, num_queries=300
):
    """Refine reference points for two-stage query initialization.

    Combines learned query reference points with encoder proposals selected
    in the first stage, using either bbox reparameterization or direct addition.

    Args:
        refpoint_embed: Learned query embeddings of shape ``(B, num_queries, 4)``.
        refpoint_embed_ts: Two-stage proposals from encoder of shape
            ``(B, num_proposals, 4)``.
        bbox_reparam: If True, use multiplicative refinement for center and
            exponential for size. If False, use additive refinement. Default True.
        num_queries: Number of queries to select from proposals. Default 300.

    Returns:
        Refined reference points of shape ``(B, num_queries, 4)`` as
        (cx, cy, w, h) in normalized coordinates.
    """
    refpoint_embed_subset = refpoint_embed[:, :num_queries, :]
    if bbox_reparam:
        ref_cxcy = (
            refpoint_embed_subset[..., :2] * refpoint_embed_ts[..., 2:]
            + refpoint_embed_ts[..., :2]
        )
        ref_wh = ops.exp(refpoint_embed_subset[..., 2:]) * refpoint_embed_ts[..., 2:]
        return ops.concatenate([ref_cxcy, ref_wh], axis=-1)
    else:
        return refpoint_embed_subset + refpoint_embed_ts


def rf_detr_conv_bn(
    x,
    filters,
    kernel_size=3,
    strides=1,
    groups=1,
    activation="relu",
    use_layer_norm=False,
    name="conv_bn",
):
    """Apply convolution followed by batch normalization and activation.

    A convenience layer that combines Conv2D, BatchNormalization (or
    RFDETRChannelLayerNorm), and activation into a single block.

    Args:
        x: Input tensor of shape ``(B, H, W, C_in)``.
        filters: Number of output filters (channels).
        kernel_size: Convolution kernel size. Default 3.
        strides: Convolution stride. Default 1.
        groups: Number of groups for grouped convolution. Default 1.
        activation: Activation function name. Default "relu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm instead of BatchNorm.
            Default False.
        name: Layer name prefix. Default "conv_bn".

    Returns:
        Output tensor of shape ``(B, H', W', filters)``.
    """
    padding = (kernel_size // 2, kernel_size // 2)
    x = layers.ZeroPadding2D(padding=padding, name=f"{name}_pad")(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="valid",
        groups=groups,
        use_bias=False,
        data_format="channels_last",
        name=f"{name}_conv",
    )(x)
    if use_layer_norm:
        x = RFDETRChannelLayerNorm(name=f"{name}_ln")(x)
    else:
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.1,
            name=f"{name}_bn",
        )(x)
    x = layers.Activation(activation, name=f"{name}_{activation}")(x)
    return x


def rf_detr_bottleneck(
    x,
    out_channels,
    shortcut=True,
    expansion=1.0,
    activation="silu",
    use_layer_norm=False,
    name="bottleneck",
):
    """Apply a bottleneck block with optional residual connection.

    A lightweight bottleneck that applies two 3x3 convolutions with an
    optional skip connection, used in C2F modules.

    Args:
        x: Input tensor of shape ``(B, H, W, C_in)``.
        out_channels: Number of output channels.
        shortcut: Whether to add residual connection. Only applied when
            input and output channels match. Default True.
        expansion: Hidden channel expansion ratio. The intermediate channels
            are ``out_channels * expansion``. Default 1.0.
        activation: Activation function name. Default "silu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm instead of BatchNorm.
            Default False.
        name: Layer name prefix. Default "bottleneck".

    Returns:
        Output tensor of shape ``(B, H, W, out_channels)``.
    """
    hidden = int(out_channels * expansion)
    in_channels = x.shape[-1]
    residual = x
    x = rf_detr_conv_bn(
        x,
        hidden,
        3,
        activation=activation,
        use_layer_norm=use_layer_norm,
        name=f"{name}_cv1",
    )
    x = rf_detr_conv_bn(
        x,
        out_channels,
        3,
        activation=activation,
        use_layer_norm=use_layer_norm,
        name=f"{name}_cv2",
    )
    if shortcut and in_channels == out_channels:
        x = x + residual
    return x


def rf_detr_c2f(
    x,
    out_channels,
    num_blocks=1,
    shortcut=False,
    expansion=0.5,
    activation="silu",
    use_layer_norm=False,
    name="c2f",
):
    """Apply a C2F (CSP Bottleneck with 2 convolutions) module from YOLO.

    Implements the C2F module that splits features and processes through
    a series of bottleneck blocks, then concatenates and fuses outputs.

    Args:
        x: Input tensor of shape ``(B, H, W, C_in)``.
        out_channels: Number of output channels.
        num_blocks: Number of bottleneck blocks to apply. Default 1.
        shortcut: Whether to use residual connections in bottlenecks.
            Default False.
        expansion: Hidden channel expansion ratio for the split. Default 0.5.
        activation: Activation function name. Default "silu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm instead of BatchNorm.
            Default False.
        name: Layer name prefix. Default "c2f".

    Returns:
        Output tensor of shape ``(B, H, W, out_channels)``.
    """
    c = int(out_channels * expansion)
    x = rf_detr_conv_bn(
        x,
        2 * c,
        1,
        activation=activation,
        use_layer_norm=use_layer_norm,
        name=f"{name}_cv1",
    )
    chunks = ops.split(x, 2, axis=-1)
    y = [chunks[0], chunks[1]]
    for i in range(num_blocks):
        y.append(
            rf_detr_bottleneck(
                y[-1],
                c,
                shortcut=shortcut,
                expansion=1.0,
                activation=activation,
                use_layer_norm=use_layer_norm,
                name=f"{name}_bottleneck_{i}",
            )
        )
    x = ops.concatenate(y, axis=-1)
    x = rf_detr_conv_bn(
        x,
        out_channels,
        1,
        activation=activation,
        use_layer_norm=use_layer_norm,
        name=f"{name}_cv2",
    )
    return x


def rf_detr_simple_projector(x, out_channels, name="projector"):
    """Apply a simple 2-layer projector with convolution and normalization.

    Projects features from backbone dimension to decoder dimension using
    two convolutional layers with SwiGLU-style activation and LayerNorm.

    Args:
        x: Input tensor of shape ``(B, H, W, C_in)``.
        out_channels: Number of output channels (decoder hidden dimension).
        name: Layer name prefix. Default "projector".

    Returns:
        Output tensor of shape ``(B, H, W, out_channels)``.
    """
    in_dim = x.shape[-1]
    x = rf_detr_conv_bn(
        x,
        in_dim * 2,
        3,
        activation="silu",
        use_layer_norm=True,
        name=f"{name}_convx1",
    )
    x = rf_detr_conv_bn(
        x,
        out_channels,
        3,
        activation="silu",
        use_layer_norm=True,
        name=f"{name}_convx2",
    )
    x = RFDETRChannelLayerNorm(name=f"{name}_ln")(x)
    return x


def rf_detr_dinov2_swiglu_ffn(x, hidden_size, mlp_ratio=4, name="mlp"):
    """Apply SwiGLU feed-forward network from DINOv2.

    Implements the SwiGLU variant of FFN which uses a gated linear unit
    with Swish activation, providing better performance than standard FFN.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Model dimension (input and output size).
        mlp_ratio: Expansion ratio for hidden dimension. Default 4.
        name: Layer name prefix. Default "mlp".

    Returns:
        Output tensor of shape ``(B, seq_len, hidden_size)``.
    """
    hidden_features = int(hidden_size * mlp_ratio)
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    x = layers.Dense(2 * hidden_features, name=f"{name}_weights_in")(x)
    x1, x2 = ops.split(x, 2, axis=-1)
    x = ops.silu(x1) * x2
    x = layers.Dense(hidden_size, name=f"{name}_weights_out")(x)
    return x


def rf_detr_dinov2_mlp(x, hidden_size, mlp_ratio=4, name="mlp"):
    """Apply standard MLP feed-forward network with GELU activation.

    Implements the classic transformer FFN: two linear layers with GELU
    activation in between.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)``.
        hidden_size: Model dimension (input and output size).
        mlp_ratio: Expansion ratio for hidden dimension. Default 4.
        name: Layer name prefix. Default "mlp".

    Returns:
        Output tensor of shape ``(B, seq_len, hidden_size)``.
    """
    hidden_features = int(hidden_size * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{name}_fc1")(x)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(hidden_size, name=f"{name}_fc2")(x)
    return x


def rf_detr_dinov2_block(
    x,
    hidden_size,
    num_heads,
    mlp_ratio=4,
    use_swiglu=False,
    run_full_attention=False,
    num_windows=1,
    name="layer",
):
    """Apply a single DINOv2 transformer block.

    Implements self-attention followed by feed-forward network with
    pre-norm and residual connections. Supports both windowed and full attention.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_size)`` for full attention
            or ``(B, num_windows^2, seq_per_window, hidden_size)`` for windowed.
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for FFN hidden dimension. Default 4.
        use_swiglu: If True, use SwiGLU FFN; otherwise use GELU MLP. Default False.
        run_full_attention: If True and num_windows > 1, reshape for full attention.
            Default False.
        num_windows: Number of windows per dimension for windowed attention.
            Default 1 (full attention).
        name: Layer name prefix. Default "layer".

    Returns:
        Output tensor of same shape as input.
    """
    head_dim = hidden_size // num_heads
    shortcut = x

    if run_full_attention and num_windows > 1:
        nw2 = num_windows**2
        shape = ops.shape(x)
        x = ops.reshape(x, [-1, nw2 * shape[1], shape[2]])

    normed = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)

    q = layers.Dense(hidden_size, name=f"{name}_attention_query")(normed)
    k = layers.Dense(hidden_size, name=f"{name}_attention_key")(normed)
    v = layers.Dense(hidden_size, name=f"{name}_attention_value")(normed)

    seq_len = ops.shape(normed)[1]
    q = ops.reshape(q, [-1, seq_len, num_heads, head_dim])
    k = ops.reshape(k, [-1, seq_len, num_heads, head_dim])
    v = ops.reshape(v, [-1, seq_len, num_heads, head_dim])
    q = ops.transpose(q, [0, 2, 1, 3])
    k = ops.transpose(k, [0, 2, 1, 3])
    v = ops.transpose(v, [0, 2, 1, 3])

    scale = ops.cast(head_dim, q.dtype) ** -0.5
    attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * scale
    attn_weights = ops.softmax(attn_weights, axis=-1)
    attn_output = ops.matmul(attn_weights, v)
    attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
    attn_output = ops.reshape(attn_output, [-1, seq_len, hidden_size])
    attn_out = layers.Dense(
        hidden_size,
        name=f"{name}_attention_out_proj",
    )(attn_output)

    if run_full_attention and num_windows > 1:
        full_shape = ops.shape(attn_out)
        attn_out = ops.reshape(
            attn_out,
            [-1, full_shape[1] // nw2, full_shape[2]],
        )

    attn_out = DinoV2LayerScale(hidden_size, name=f"{name}_layer_scale1")(attn_out)
    x = attn_out + shortcut

    shortcut2 = x
    normed2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(x)
    if use_swiglu:
        mlp_out = rf_detr_dinov2_swiglu_ffn(
            normed2,
            hidden_size,
            mlp_ratio,
            name=f"{name}_mlp",
        )
    else:
        mlp_out = rf_detr_dinov2_mlp(
            normed2,
            hidden_size,
            mlp_ratio,
            name=f"{name}_mlp",
        )
    mlp_out = DinoV2LayerScale(hidden_size, name=f"{name}_layer_scale2")(mlp_out)
    x = mlp_out + shortcut2
    return x


def rf_detr_windowed_dinov2_encoder(
    x,
    hidden_size,
    num_heads,
    num_layers,
    mlp_ratio=4,
    use_swiglu=False,
    num_windows=1,
    out_feature_indexes=None,
    window_block_indexes=None,
    name="backbone_encoder",
):
    """Apply a stack of DINOv2 transformer blocks with windowed attention.

    Runs a sequence of DINOv2 blocks, alternating between windowed and full
    attention based on the window_block_indexes, and extracts features at
    specified layer indices.

    Args:
        x: Input tensor from DINOv2 embeddings.
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Total number of transformer blocks.
        mlp_ratio: Expansion ratio for FFN hidden dimension. Default 4.
        use_swiglu: If True, use SwiGLU FFN. Default False.
        num_windows: Number of windows per dimension for windowed attention.
            Default 1.
        out_feature_indexes: Layer indices (1-indexed) at which to extract
            intermediate features. Default None.
        window_block_indexes: Layer indices that should use windowed attention.
            Others will use full attention. Default None.
        name: Layer name prefix. Default "backbone_encoder".

    Returns:
        List of feature tensors extracted at out_feature_indexes (after layer
        norm). If out_feature_indexes is empty, returns empty list.
    """
    out_feature_indexes = out_feature_indexes or []
    window_block_indexes = window_block_indexes or []
    max_layer = max(out_feature_indexes) + 1 if out_feature_indexes else num_layers

    shared_layernorm = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_layernorm",
    )

    features = []
    for i in range(max_layer):
        run_full = i not in window_block_indexes
        x = rf_detr_dinov2_block(
            x,
            hidden_size,
            num_heads,
            mlp_ratio,
            use_swiglu=use_swiglu,
            run_full_attention=run_full,
            num_windows=num_windows,
            name=f"{name}_layer_{i}",
        )
        if (i + 1) in out_feature_indexes:
            features.append(shared_layernorm(x))
    return features


@keras.saving.register_keras_serializable(package="kmodels")
class RFDETR(keras.Model):
    """RF-DETR: Real-Time Detection Transformer.

    A real-time object detection model based on DINOv2 backbone with windowed
    attention, a simple projector, and a deformable DETR decoder. Uses two-stage
    query initialization and iterative bounding box refinement.

    Reference:
        - RF-DETR (Roboflow, ICLR 2026)
        - https://github.com/roboflow/rf-detr

    Args:
        hidden_dim: Transformer decoder hidden dimension.
        backbone_hidden_size: DINOv2 backbone hidden size.
        backbone_num_heads: Number of attention heads in backbone.
        backbone_num_layers: Number of transformer layers in backbone.
        backbone_mlp_ratio: MLP expansion ratio in backbone.
        backbone_use_swiglu: Whether backbone uses SwiGLU FFN.
        num_register_tokens: Number of register tokens in DINOv2.
        out_feature_indexes: Backbone layer indices to extract features from.
        patch_size: Patch size for DINOv2 patch embeddings.
        num_windows: Number of windows for windowed attention.
        positional_encoding_size: Size of positional encoding grid.
        resolution: Input image resolution.
        dec_layers: Number of decoder layers.
        sa_nheads: Number of self-attention heads in decoder.
        ca_nheads: Number of cross-attention heads in decoder.
        dec_n_points: Number of sampling points in deformable attention.
        num_queries: Number of object queries.
        num_classes: Number of object classes (COCO: 91).
        two_stage: Whether to use two-stage query initialization.
        bbox_reparam: Whether to use bbox reparameterization.
        lite_refpoint_refine: Whether to use lite reference point refinement.
        group_detr: Number of DETR groups (training only, inference uses 1).
        dim_feedforward: FFN dimension in decoder.
        weights: Pre-trained weight identifier or file path.
        input_shape: Input image shape as (H, W, C).
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    def __init__(
        self,
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=True,
        num_register_tokens=4,
        out_feature_indexes=None,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        resolution=560,
        dec_layers=3,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=300,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        weights="coco",
        input_shape=None,
        input_tensor=None,
        name="RFDETR",
        **kwargs,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]

        if input_shape is None:
            input_shape = (resolution, resolution, 3)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        inputs = img_input

        embeddings = DinoV2Embeddings(
            hidden_size=backbone_hidden_size,
            patch_size=patch_size,
            num_channels=3,
            num_register_tokens=num_register_tokens,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            name="backbone_embeddings",
        )(inputs)

        window_block_indexes = sorted(
            set(range(max(out_feature_indexes) + 1)) - set(out_feature_indexes)
        )

        encoder_features = rf_detr_windowed_dinov2_encoder(
            embeddings,
            hidden_size=backbone_hidden_size,
            num_heads=backbone_num_heads,
            num_layers=backbone_num_layers,
            mlp_ratio=backbone_mlp_ratio,
            use_swiglu=backbone_use_swiglu,
            num_windows=num_windows,
            out_feature_indexes=out_feature_indexes,
            window_block_indexes=window_block_indexes,
            name="backbone_encoder",
        )

        num_h = input_shape[0] // patch_size
        num_w = input_shape[1] // patch_size

        unwindowed_features = []
        for i, feat in enumerate(encoder_features):
            uw = rf_detr_unwindow_features(
                feat,
                num_h,
                num_w,
                num_windows,
                backbone_hidden_size,
                num_register_tokens,
            )
            unwindowed_features.append(uw)

        concat_feat = layers.Concatenate(axis=-1, name="concat_features")(
            unwindowed_features
        )
        projected = rf_detr_c2f(
            concat_feat,
            hidden_dim,
            num_blocks=3,
            shortcut=False,
            expansion=0.5,
            activation="silu",
            use_layer_norm=True,
            name="projector_c2f",
        )
        projected = RFDETRChannelLayerNorm(name="projector_ln")(projected)

        proj_shape = (input_shape[0] // patch_size, input_shape[1] // patch_size)
        num_feature_levels = 1
        spatial_shapes = [proj_shape]
        level_start_index = [0]

        src_flat = ops.reshape(
            projected, [-1, proj_shape[0] * proj_shape[1], hidden_dim]
        )
        memory = src_flat

        tgt = RFDETRLearnedEmbedding(
            num_queries,
            hidden_dim,
            initializer="glorot_uniform",
            name="query_feat_embed",
        )(memory)
        refpoint_embed = RFDETRLearnedEmbedding(
            num_queries,
            4,
            initializer="zeros",
            name="refpoint_embed_layer",
        )(memory)

        if two_stage:
            output_memory_filtered, output_proposals = rf_detr_encoder_output_proposals(
                memory,
                spatial_shapes=spatial_shapes,
                bbox_reparam=bbox_reparam,
            )

            enc_output_proj = layers.Dense(hidden_dim, name="enc_output_0")(
                output_memory_filtered
            )
            enc_output_norm = layers.LayerNormalization(
                epsilon=1e-5, name="enc_output_norm_0"
            )
            output_memory_proj = enc_output_norm(enc_output_proj)

            enc_cls = layers.Dense(num_classes, name="enc_out_class_embed_0")(
                output_memory_proj
            )

            bbox_embed_enc_0 = layers.Dense(
                hidden_dim, activation="relu", name="enc_bbox_0"
            )
            bbox_embed_enc_1 = layers.Dense(
                hidden_dim, activation="relu", name="enc_bbox_1"
            )
            bbox_embed_enc_2 = layers.Dense(4, name="enc_bbox_2")

            enc_bbox_delta = bbox_embed_enc_2(
                bbox_embed_enc_1(bbox_embed_enc_0(output_memory_proj))
            )

            if bbox_reparam:
                enc_coord_cxcy = (
                    enc_bbox_delta[..., :2] * output_proposals[..., 2:]
                    + output_proposals[..., :2]
                )
                enc_coord_wh = (
                    ops.exp(enc_bbox_delta[..., 2:]) * output_proposals[..., 2:]
                )
                enc_coords = ops.concatenate([enc_coord_cxcy, enc_coord_wh], axis=-1)
            else:
                enc_coords = enc_bbox_delta + output_proposals

            enc_cls_max = ops.max(enc_cls, axis=-1)
            topk_indices = ops.top_k(
                enc_cls_max, k=min(num_queries, proj_shape[0] * proj_shape[1])
            )[1]
            topk_idx_4 = ops.repeat(ops.expand_dims(topk_indices, axis=-1), 4, axis=-1)
            refpoint_embed_ts = ops.take_along_axis(enc_coords, topk_idx_4, axis=1)
            refpoint_embed_ts = ops.stop_gradient(refpoint_embed_ts)

            refpoints_unsigmoid = rf_detr_two_stage_refine_refpoints(
                refpoint_embed,
                refpoint_embed_ts,
                bbox_reparam=bbox_reparam,
                num_queries=num_queries,
            )
        else:
            refpoints_unsigmoid = refpoint_embed

        ref_point_head_0 = layers.Dense(
            hidden_dim, activation="relu", name="ref_point_head_0"
        )
        ref_point_head_1 = layers.Dense(hidden_dim, name="ref_point_head_1")

        decoder_layers_list = []
        for i in range(dec_layers):
            decoder_layers_list.append(
                RFDETRDecoderLayer(
                    d_model=hidden_dim,
                    sa_nhead=sa_nheads,
                    ca_nhead=ca_nheads,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    num_feature_levels=num_feature_levels,
                    dec_n_points=dec_n_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    name=f"decoder_layer_{i}",
                )
            )

        decoder_norm = layers.LayerNormalization(epsilon=1e-5, name="decoder_norm")

        bbox_embed_0 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_0")
        bbox_embed_1 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_1")
        bbox_embed_2 = layers.Dense(4, name="bbox_embed_2")

        level_start_index = [0]

        if bbox_reparam:
            ref_for_query = refpoints_unsigmoid
        else:
            ref_for_query = ops.sigmoid(refpoints_unsigmoid)

        if lite_refpoint_refine:
            query_sine = rf_detr_gen_sineembed_for_position(
                ref_for_query[..., :4], dim=hidden_dim // 2
            )
            query_pos = ref_point_head_1(ref_point_head_0(query_sine))

        output = tgt
        for layer_id, dec_layer in enumerate(decoder_layers_list):
            if not lite_refpoint_refine:
                if bbox_reparam:
                    ref_for_query = refpoints_unsigmoid
                else:
                    ref_for_query = ops.sigmoid(refpoints_unsigmoid)
                query_sine = rf_detr_gen_sineembed_for_position(
                    ref_for_query[..., :4], dim=hidden_dim // 2
                )
                query_pos = ref_point_head_1(ref_point_head_0(query_sine))

            if bbox_reparam:
                ref_input = refpoints_unsigmoid[:, :, None, :]
            else:
                sig_ref = ops.sigmoid(refpoints_unsigmoid)
                ref_input = sig_ref[:, :, None, :]

            output = dec_layer(
                output,
                memory,
                query_pos,
                ref_input,
                training=False,
            )

            if not lite_refpoint_refine:
                new_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(output)))
                if bbox_reparam:
                    new_cxcy = (
                        new_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                        + refpoints_unsigmoid[..., :2]
                    )
                    new_wh = ops.exp(new_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
                    refpoints_unsigmoid = ops.concatenate([new_cxcy, new_wh], axis=-1)
                else:
                    refpoints_unsigmoid = refpoints_unsigmoid + new_delta
                refpoints_unsigmoid = ops.stop_gradient(refpoints_unsigmoid)

        output = decoder_norm(output)

        class_embed = layers.Dense(num_classes, name="class_embed")
        pred_logits = class_embed(output)

        pred_bbox_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(output)))
        if bbox_reparam:
            pred_cxcy = (
                pred_bbox_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )
            pred_wh = ops.exp(pred_bbox_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
            pred_boxes = ops.concatenate([pred_cxcy, pred_wh], axis=-1)
        else:
            pred_boxes = ops.sigmoid(pred_bbox_delta + refpoints_unsigmoid)

        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_heads = backbone_num_heads
        self.backbone_num_layers = backbone_num_layers
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.backbone_use_swiglu = backbone_use_swiglu
        self.num_register_tokens = num_register_tokens
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.resolution = resolution
        self.dec_layers = dec_layers
        self.sa_nheads = sa_nheads
        self.ca_nheads = ca_nheads
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.group_detr = group_detr
        self.dim_feedforward = dim_feedforward
        self._input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "backbone_hidden_size": self.backbone_hidden_size,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_num_layers": self.backbone_num_layers,
                "backbone_mlp_ratio": self.backbone_mlp_ratio,
                "backbone_use_swiglu": self.backbone_use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "out_feature_indexes": self.out_feature_indexes,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
                "resolution": self.resolution,
                "dec_layers": self.dec_layers,
                "sa_nheads": self.sa_nheads,
                "ca_nheads": self.ca_nheads,
                "dec_n_points": self.dec_n_points,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "two_stage": self.two_stage,
                "bbox_reparam": self.bbox_reparam,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "group_detr": self.group_detr,
                "dim_feedforward": self.dim_feedforward,
                "input_shape": self.input_shape[1:],
                "input_tensor": self._input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_rf_detr_model(
    variant,
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name=None,
    **kwargs,
):
    """Factory function to create RF-DETR model variants.

    Creates an RF-DETR model with configuration based on the variant name,
    loading pre-trained weights if available.

    Args:
        variant: Model variant name (e.g., "RFDETRNano", "RFDETRSmall").
        num_queries: Number of object queries. Default 300.
        num_classes: Number of object classes. Default 91 (COCO).
        weights: Weight identifier ("coco", None) or path to weights file.
            Default "coco".
        input_shape: Input shape as ``(H, W, C)``. If None, uses variant default.
        input_tensor: Optional input tensor for functional API usage.
        name: Model name. If None, uses variant name.
        **kwargs: Additional arguments passed to RFDETR constructor.

    Returns:
        RF-DETR model instance with loaded weights.
    """
    config = RF_DETR_MODEL_CONFIG[variant]

    if input_shape is None:
        res = config["resolution"]
        input_shape = (res, res, 3)

    model = RFDETR(
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=False,
        num_register_tokens=0,
        out_feature_indexes=config.get("out_feature_indexes", [3, 6, 9, 12]),
        patch_size=config.get("patch_size", 16),
        num_windows=config.get("num_windows", 2),
        positional_encoding_size=config["positional_encoding_size"],
        resolution=config["resolution"],
        dec_layers=config["dec_layers"],
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=num_queries,
        num_classes=num_classes,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name or variant,
        **kwargs,
    )

    if weights in RF_DETR_WEIGHTS_CONFIG.get(variant, {}):
        load_weights_from_config(variant, weights, model, RF_DETR_WEIGHTS_CONFIG)
    elif weights is not None and weights != "coco":
        model.load_weights(weights)

    return model


@register_model
def RFDETRNano(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRNano",
    **kwargs,
):
    return _create_rf_detr_model(
        "RFDETRNano",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRSmall(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRSmall",
    **kwargs,
):
    return _create_rf_detr_model(
        "RFDETRSmall",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRMedium(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRMedium",
    **kwargs,
):
    return _create_rf_detr_model(
        "RFDETRMedium",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRBase(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRBase",
    **kwargs,
):
    return _create_rf_detr_model(
        "RFDETRBase",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )


@register_model
def RFDETRLarge(
    num_queries=300,
    num_classes=91,
    weights="coco",
    input_shape=None,
    input_tensor=None,
    name="RFDETRLarge",
    **kwargs,
):
    return _create_rf_detr_model(
        "RFDETRLarge",
        num_queries=num_queries,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=name,
        **kwargs,
    )
