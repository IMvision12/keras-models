import keras
import numpy as np
from keras import layers, ops, utils

from kmodels.model_registry import register_model
from kmodels.utils import load_weights_from_config

from .config import SAM3_MODEL_CONFIG, SAM3_WEIGHTS_CONFIG
from .sam3_layers import (
    SAM3DecoderMLP,
    SAM3DetrDecoderLayer,
    SAM3DetrEncoderLayer,
    SAM3DotProductScoring,
    SAM3MaskEmbedder,
    SAM3ViTLayer,
    _box_cxcywh_to_xyxy,
    _inverse_sigmoid,
)


def _sine_encode_boxes(boxes, num_pos_feats=128, temperature=10000):
    scale = 2 * 3.141592653589793
    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    boxes_scaled = boxes * scale
    pos = ops.expand_dims(boxes_scaled, axis=-1) / dim_t

    pos_sin = ops.sin(pos[..., 0::2])
    pos_cos = ops.cos(pos[..., 1::2])
    pos_embed = ops.reshape(
        ops.stack([pos_sin, pos_cos], axis=-1),
        ops.shape(pos)[:-1] + (num_pos_feats,),
    )
    shape = ops.shape(pos_embed)
    return ops.reshape(pos_embed, (shape[0], shape[1], shape[2] * shape[3]))


def _compute_sine_pos_encoding(
    height, width, num_pos_feats, temperature=10000, normalize=True
):
    import math

    scale = 2 * math.pi
    y_embed = np.cumsum(np.ones((1, height, width), dtype=np.float32), axis=1)
    x_embed = np.cumsum(np.ones((1, height, width), dtype=np.float32), axis=2)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2 * np.floor(dim_t / 2) / num_pos_feats)

    pos_x = x_embed[..., np.newaxis] / dim_t
    pos_y = y_embed[..., np.newaxis] / dim_t

    pos_x_sin = np.sin(pos_x[:, :, :, 0::2])
    pos_x_cos = np.cos(pos_x[:, :, :, 1::2])
    pos_y_sin = np.sin(pos_y[:, :, :, 0::2])
    pos_y_cos = np.cos(pos_y[:, :, :, 1::2])

    pos_x = np.stack([pos_x_sin, pos_x_cos], axis=4).reshape(
        1, height, width, num_pos_feats
    )
    pos_y = np.stack([pos_y_sin, pos_y_cos], axis=4).reshape(
        1, height, width, num_pos_feats
    )

    pos = np.concatenate([pos_y, pos_x], axis=-1)
    pos = pos.transpose(0, 3, 1, 2)  # (1, 2*num_pos_feats, H, W)
    return pos


@keras.saving.register_keras_serializable(package="kmodels")
class SAM3(keras.Model):
    LAYER_NORM_EPS = 1e-6

    def __init__(
        self,
        vit_hidden_size=1024,
        vit_intermediate_size=4736,
        vit_num_hidden_layers=32,
        vit_num_attention_heads=16,
        vit_image_size=1008,
        vit_patch_size=14,
        vit_window_size=24,
        vit_global_attn_indexes=(7, 15, 23, 31),
        vit_rope_theta=10000.0,
        vit_pretrain_image_size=336,
        fpn_hidden_size=256,
        fpn_scale_factors=(4.0, 2.0, 1.0, 0.5),
        detr_encoder_hidden_size=256,
        detr_encoder_num_layers=6,
        detr_encoder_num_attention_heads=8,
        detr_encoder_intermediate_size=2048,
        detr_encoder_dropout=0.1,
        detr_decoder_hidden_size=256,
        detr_decoder_num_layers=6,
        detr_decoder_num_queries=200,
        detr_decoder_num_attention_heads=8,
        detr_decoder_intermediate_size=2048,
        detr_decoder_dropout=0.1,
        mask_decoder_hidden_size=256,
        mask_decoder_num_upsampling_stages=3,
        mask_decoder_num_attention_heads=8,
        text_hidden_size=1024,
        text_projection_dim=512,
        input_shape=None,
        input_tensor=None,
        name="SAM3",
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (vit_image_size, vit_image_size, 3)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=input_shape, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=input_shape, name="pixel_values")

        text_features_input = layers.Input(
            shape=(None, text_hidden_size), name="text_features", dtype="float32"
        )
        text_attention_mask = layers.Input(
            shape=(None,), name="text_attention_mask", dtype="float32"
        )

        grid_size = vit_image_size // vit_patch_size

        patch_embed = layers.Conv2D(
            vit_hidden_size,
            kernel_size=vit_patch_size,
            strides=vit_patch_size,
            padding="valid",
            use_bias=False,
            name="backbone_patch_embed",
        )(pixel_values)

        pretrain_grid = vit_pretrain_image_size // vit_patch_size
        num_pretrain_patches = pretrain_grid * pretrain_grid

        hidden_states = layers.Reshape(
            (grid_size * grid_size, vit_hidden_size),
            name="backbone_flatten",
        )(patch_embed)

        pos_embed_layer = layers.Embedding(
            num_pretrain_patches,
            vit_hidden_size,
            name="backbone_position_embedding",
        )

        pos_indices = np.arange(num_pretrain_patches)
        pos_embed_static = pos_embed_layer(
            ops.convert_to_tensor(pos_indices, dtype="int32")
        )
        pos_embed_static = ops.reshape(
            pos_embed_static, (1, pretrain_grid, pretrain_grid, vit_hidden_size)
        )

        if grid_size != pretrain_grid:
            tiles_h = int(np.ceil(grid_size / pretrain_grid))
            tiles_w = int(np.ceil(grid_size / pretrain_grid))
            pos_embed_tiled = ops.tile(pos_embed_static, (1, tiles_h, tiles_w, 1))
            pos_embed_tiled = pos_embed_tiled[:, :grid_size, :grid_size, :]
        else:
            pos_embed_tiled = pos_embed_static

        pos_embed_flat = ops.reshape(
            pos_embed_tiled, (1, grid_size * grid_size, vit_hidden_size)
        )
        hidden_states = layers.Add(name="backbone_add_pos")(
            [hidden_states, pos_embed_flat]
        )

        hidden_states = layers.Reshape(
            (grid_size, grid_size, vit_hidden_size),
            name="backbone_to_spatial",
        )(hidden_states)

        for i in range(vit_num_hidden_layers):
            win = vit_window_size if i not in vit_global_attn_indexes else 0
            hidden_states = SAM3ViTLayer(
                hidden_size=vit_hidden_size,
                num_attention_heads=vit_num_attention_heads,
                intermediate_size=vit_intermediate_size,
                window_size=win,
                image_size=grid_size,
                layer_norm_eps=self.LAYER_NORM_EPS,
                rope_theta=vit_rope_theta,
                name=f"backbone_layers_{i}",
            )(hidden_states)

        backbone_out_flat = layers.Reshape(
            (grid_size * grid_size, vit_hidden_size),
            name="backbone_out_flatten",
        )(hidden_states)
        backbone_out_flat = layers.LayerNormalization(
            epsilon=self.LAYER_NORM_EPS, name="backbone_layer_norm"
        )(backbone_out_flat)
        backbone_out = layers.Reshape(
            (grid_size, grid_size, vit_hidden_size),
            name="backbone_out_spatial",
        )(backbone_out_flat)

        backbone_nchw = layers.Permute((3, 1, 2), name="backbone_to_nchw")(backbone_out)

        fpn_hidden_states = []

        for level_idx, scale_factor in enumerate(fpn_scale_factors):
            if scale_factor == 4.0:
                x = layers.Conv2DTranspose(
                    vit_hidden_size // 2,
                    kernel_size=2,
                    strides=2,
                    data_format="channels_first",
                    name=f"fpn_level_{level_idx}_deconv1",
                )(backbone_nchw)
                x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
                x = layers.Conv2DTranspose(
                    vit_hidden_size // 4,
                    kernel_size=2,
                    strides=2,
                    data_format="channels_first",
                    name=f"fpn_level_{level_idx}_deconv2",
                )(x)
                x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu2")(x)
            elif scale_factor == 2.0:
                x = layers.Conv2DTranspose(
                    vit_hidden_size // 2,
                    kernel_size=2,
                    strides=2,
                    data_format="channels_first",
                    name=f"fpn_level_{level_idx}_deconv1",
                )(backbone_nchw)
                x = layers.Activation("gelu", name=f"fpn_level_{level_idx}_gelu1")(x)
            elif scale_factor == 1.0:
                x = backbone_nchw
            else:  # 0.5
                x = layers.MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    data_format="channels_first",
                    name=f"fpn_level_{level_idx}_pool",
                )(backbone_nchw)

            x = layers.Conv2D(
                fpn_hidden_size,
                kernel_size=1,
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_proj1",
            )(x)
            x = layers.Conv2D(
                fpn_hidden_size,
                kernel_size=3,
                padding="same",
                data_format="channels_first",
                name=f"fpn_level_{level_idx}_proj2",
            )(x)

            fpn_hidden_states.append(x)

        text_projected = layers.Dense(detr_encoder_hidden_size, name="text_projection")(
            text_features_input
        )

        encoder_vision = fpn_hidden_states[-1]  # (B, C, H, W)

        enc_h = grid_size // 2 if 0.5 in fpn_scale_factors else grid_size

        encoder_vision_flat = layers.Permute((2, 3, 1), name="encoder_vision_to_nhwc")(
            encoder_vision
        )
        encoder_vision_flat = layers.Reshape(
            (enc_h * enc_h, fpn_hidden_size), name="encoder_vision_flatten"
        )(encoder_vision_flat)

        enc_pos_np = _compute_sine_pos_encoding(
            enc_h, enc_h, fpn_hidden_size // 2, normalize=True
        )  # (1, fpn_hidden_size, enc_h, enc_h)
        enc_pos_np = enc_pos_np.transpose(0, 2, 3, 1)  # (1, H, W, C)
        enc_pos_np = enc_pos_np.reshape(1, enc_h * enc_h, fpn_hidden_size)
        encoder_pos_flat = ops.convert_to_tensor(enc_pos_np, dtype="float32")

        text_attn_mask = layers.Lambda(
            lambda m: ops.expand_dims(
                ops.expand_dims((1.0 - m) * (-1e9), axis=1), axis=1
            ),
            name="text_attn_mask",
        )(text_attention_mask)

        encoder_output = encoder_vision_flat
        for i in range(detr_encoder_num_layers):
            encoder_output = SAM3DetrEncoderLayer(
                hidden_size=detr_encoder_hidden_size,
                num_attention_heads=detr_encoder_num_attention_heads,
                intermediate_size=detr_encoder_intermediate_size,
                dropout=detr_encoder_dropout,
                name=f"detr_encoder_layers_{i}",
            )(
                encoder_output,
                text_projected,
                encoder_pos_flat,
                text_mask=text_attn_mask,
            )

        query_embed_layer = layers.Embedding(
            detr_decoder_num_queries,
            detr_decoder_hidden_size,
            name="detr_decoder_query_embed",
        )
        reference_points_layer = layers.Embedding(
            detr_decoder_num_queries,
            4,
            name="detr_decoder_reference_points",
        )
        presence_token_layer = layers.Embedding(
            1,
            detr_decoder_hidden_size,
            name="detr_decoder_presence_token",
        )

        def _expand_embedding(args):
            embed, batch_ref = args
            batch_size = ops.shape(batch_ref)[0]
            return ops.broadcast_to(embed, (batch_size,) + ops.shape(embed)[1:])

        query_indices = ops.convert_to_tensor(
            np.arange(detr_decoder_num_queries), dtype="int32"
        )
        query_embed_raw = query_embed_layer(query_indices)
        query_embed_raw = ops.expand_dims(query_embed_raw, axis=0)

        reference_points_raw = reference_points_layer(query_indices)
        reference_points_raw = ops.sigmoid(reference_points_raw)
        reference_points_raw = ops.expand_dims(reference_points_raw, axis=0)

        presence_idx = ops.convert_to_tensor([0], dtype="int32")
        presence_raw = presence_token_layer(presence_idx)
        presence_raw = ops.expand_dims(presence_raw, axis=0)

        query_embed = layers.Lambda(_expand_embedding, name="expand_query_embed")(
            [query_embed_raw, encoder_output]
        )
        reference_points = layers.Lambda(_expand_embedding, name="expand_ref_points")(
            [reference_points_raw, encoder_output]
        )
        presence_token = layers.Lambda(_expand_embedding, name="expand_presence_token")(
            [presence_raw, encoder_output]
        )

        box_head = SAM3DecoderMLP(
            detr_decoder_hidden_size,
            detr_decoder_hidden_size,
            4,
            num_layers=3,
            name="detr_decoder_box_head",
        )
        presence_head = SAM3DecoderMLP(
            detr_decoder_hidden_size,
            detr_decoder_hidden_size,
            1,
            num_layers=3,
            name="detr_decoder_presence_head",
        )
        ref_point_head = SAM3DecoderMLP(
            2 * detr_decoder_hidden_size,
            detr_decoder_hidden_size,
            detr_decoder_hidden_size,
            num_layers=2,
            name="detr_decoder_ref_point_head",
        )

        decoder_hidden = query_embed
        all_reference_boxes = []
        all_presence_logits = []

        for i in range(detr_decoder_num_layers):
            ref_encoded = _sine_encode_boxes(
                reference_points,
                num_pos_feats=detr_decoder_hidden_size // 2,
            )
            query_pos = ref_point_head(ref_encoded)

            decoder_hidden = SAM3DetrDecoderLayer(
                hidden_size=detr_decoder_hidden_size,
                num_attention_heads=detr_decoder_num_attention_heads,
                intermediate_size=detr_decoder_intermediate_size,
                dropout=detr_decoder_dropout,
                name=f"detr_decoder_layers_{i}",
            )(
                decoder_hidden,
                query_pos,
                text_projected,
                encoder_output,
                encoder_pos_flat,
                text_mask=text_attn_mask,
            )

            box_delta = box_head(decoder_hidden)
            new_ref = ops.sigmoid(
                _inverse_sigmoid(ops.stop_gradient(reference_points)) + box_delta
            )
            reference_points = new_ref

            all_reference_boxes.append(reference_points)

            presence_logit = presence_head(presence_token)
            presence_logit = ops.squeeze(presence_logit, axis=-1)
            presence_logit = ops.clip(presence_logit, -10.0, 10.0)
            all_presence_logits.append(presence_logit)

        pred_boxes = _box_cxcywh_to_xyxy(all_reference_boxes[-1])

        scoring = SAM3DotProductScoring(
            hidden_size=detr_decoder_hidden_size,
            text_hidden_size=text_hidden_size,
            text_projection_dim=text_projection_dim,
            name="dot_product_scoring",
        )
        pred_logits = scoring(decoder_hidden, text_features_input, text_attention_mask)

        pixel_feat = layers.Reshape(
            (enc_h, enc_h, fpn_hidden_size),
            name="pixel_decoder_reshape_encoder",
        )(encoder_output)
        pixel_feat = layers.Permute((3, 1, 2), name="pixel_decoder_to_nchw")(pixel_feat)

        num_up = mask_decoder_num_upsampling_stages
        for stage_idx in range(num_up):
            pixel_feat_nhwc = layers.Permute(
                (2, 3, 1), name=f"pixel_decoder_stage_{stage_idx}_to_nhwc"
            )(pixel_feat)
            pixel_feat_nhwc = layers.UpSampling2D(
                size=2,
                interpolation="nearest",
                name=f"pixel_decoder_stage_{stage_idx}_upsample",
            )(pixel_feat_nhwc)

            skip_idx = len(fpn_hidden_states) - 2 - stage_idx
            if skip_idx >= 0:
                skip_nhwc = layers.Permute(
                    (2, 3, 1),
                    name=f"pixel_decoder_stage_{stage_idx}_skip_to_nhwc",
                )(fpn_hidden_states[skip_idx])
                pixel_feat_nhwc = layers.Add(
                    name=f"pixel_decoder_stage_{stage_idx}_add_skip",
                )([pixel_feat_nhwc, skip_nhwc])

            pixel_feat_nhwc = layers.Conv2D(
                mask_decoder_hidden_size,
                kernel_size=3,
                padding="same",
                name=f"pixel_decoder_stage_{stage_idx}_conv",
            )(pixel_feat_nhwc)
            pixel_feat_nhwc = layers.GroupNormalization(
                groups=8,
                name=f"pixel_decoder_stage_{stage_idx}_gn",
            )(pixel_feat_nhwc)
            pixel_feat_nhwc = layers.ReLU(
                name=f"pixel_decoder_stage_{stage_idx}_relu",
            )(pixel_feat_nhwc)

            pixel_feat = layers.Permute(
                (3, 1, 2),
                name=f"pixel_decoder_stage_{stage_idx}_to_nchw",
            )(pixel_feat_nhwc)

        instance_embed = layers.Conv2D(
            mask_decoder_hidden_size,
            kernel_size=1,
            data_format="channels_first",
            name="mask_decoder_instance_proj",
        )(pixel_feat)  # (B, C, H, W)

        semantic_seg = layers.Conv2D(
            1,
            kernel_size=1,
            data_format="channels_first",
            name="mask_decoder_semantic_proj",
        )(pixel_feat)  # (B, 1, H, W)

        mask_embedder = SAM3MaskEmbedder(
            hidden_size=mask_decoder_hidden_size,
            name="mask_embedder",
        )
        mask_embeddings = mask_embedder(decoder_hidden)  # (B, Q, C)

        instance_nhwc = layers.Permute((2, 3, 1), name="instance_to_nhwc")(
            instance_embed
        )
        pred_masks = ops.einsum("bqc,bhwc->bqhw", mask_embeddings, instance_nhwc)

        presence_logits_stacked = ops.concatenate(
            [ops.squeeze(p, axis=1) for p in all_presence_logits], axis=-1
        )

        outputs = {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits_stacked,
            "semantic_seg": semantic_seg,
        }

        super().__init__(
            inputs={
                "pixel_values": pixel_values,
                "text_features": text_features_input,
                "text_attention_mask": text_attention_mask,
            },
            outputs=outputs,
            name=name,
            **kwargs,
        )

        self.vit_hidden_size = vit_hidden_size
        self.vit_intermediate_size = vit_intermediate_size
        self.vit_num_hidden_layers = vit_num_hidden_layers
        self.vit_num_attention_heads = vit_num_attention_heads
        self.vit_image_size = vit_image_size
        self.vit_patch_size = vit_patch_size
        self.vit_window_size = vit_window_size
        self.vit_global_attn_indexes = list(vit_global_attn_indexes)
        self.vit_rope_theta = vit_rope_theta
        self.vit_pretrain_image_size = vit_pretrain_image_size
        self.fpn_hidden_size = fpn_hidden_size
        self.fpn_scale_factors = list(fpn_scale_factors)
        self.detr_encoder_hidden_size = detr_encoder_hidden_size
        self.detr_encoder_num_layers = detr_encoder_num_layers
        self.detr_encoder_num_attention_heads = detr_encoder_num_attention_heads
        self.detr_encoder_intermediate_size = detr_encoder_intermediate_size
        self.detr_encoder_dropout = detr_encoder_dropout
        self.detr_decoder_hidden_size = detr_decoder_hidden_size
        self.detr_decoder_num_layers = detr_decoder_num_layers
        self.detr_decoder_num_queries = detr_decoder_num_queries
        self.detr_decoder_num_attention_heads = detr_decoder_num_attention_heads
        self.detr_decoder_intermediate_size = detr_decoder_intermediate_size
        self.detr_decoder_dropout = detr_decoder_dropout
        self.mask_decoder_hidden_size = mask_decoder_hidden_size
        self.mask_decoder_num_upsampling_stages = mask_decoder_num_upsampling_stages
        self.mask_decoder_num_attention_heads = mask_decoder_num_attention_heads
        self.text_hidden_size = text_hidden_size
        self.text_projection_dim = text_projection_dim
        self._input_shape_val = input_shape
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vit_hidden_size": self.vit_hidden_size,
                "vit_intermediate_size": self.vit_intermediate_size,
                "vit_num_hidden_layers": self.vit_num_hidden_layers,
                "vit_num_attention_heads": self.vit_num_attention_heads,
                "vit_image_size": self.vit_image_size,
                "vit_patch_size": self.vit_patch_size,
                "vit_window_size": self.vit_window_size,
                "vit_global_attn_indexes": self.vit_global_attn_indexes,
                "vit_rope_theta": self.vit_rope_theta,
                "vit_pretrain_image_size": self.vit_pretrain_image_size,
                "fpn_hidden_size": self.fpn_hidden_size,
                "fpn_scale_factors": self.fpn_scale_factors,
                "detr_encoder_hidden_size": self.detr_encoder_hidden_size,
                "detr_encoder_num_layers": self.detr_encoder_num_layers,
                "detr_encoder_num_attention_heads": self.detr_encoder_num_attention_heads,
                "detr_encoder_intermediate_size": self.detr_encoder_intermediate_size,
                "detr_encoder_dropout": self.detr_encoder_dropout,
                "detr_decoder_hidden_size": self.detr_decoder_hidden_size,
                "detr_decoder_num_layers": self.detr_decoder_num_layers,
                "detr_decoder_num_queries": self.detr_decoder_num_queries,
                "detr_decoder_num_attention_heads": self.detr_decoder_num_attention_heads,
                "detr_decoder_intermediate_size": self.detr_decoder_intermediate_size,
                "detr_decoder_dropout": self.detr_decoder_dropout,
                "mask_decoder_hidden_size": self.mask_decoder_hidden_size,
                "mask_decoder_num_upsampling_stages": self.mask_decoder_num_upsampling_stages,
                "mask_decoder_num_attention_heads": self.mask_decoder_num_attention_heads,
                "text_hidden_size": self.text_hidden_size,
                "text_projection_dim": self.text_projection_dim,
                "input_shape": self._input_shape_val,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_sam3_model(
    variant,
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    config = SAM3_MODEL_CONFIG[variant]

    valid_model_weights = []
    if variant in SAM3_WEIGHTS_CONFIG:
        valid_model_weights = list(SAM3_WEIGHTS_CONFIG[variant].keys())

    valid_weights = [None] + valid_model_weights

    if weights not in valid_weights and not isinstance(weights, str):
        raise ValueError(
            f"Invalid weights: {weights}. "
            f"Supported weights for {variant} are "
            f"{', '.join([str(w) for w in valid_weights])}, "
            "a path to a weights file, or None."
        )

    if input_shape is None:
        image_size = config["vit_image_size"]
        input_shape = (image_size, image_size, 3)

    model = SAM3(
        vit_hidden_size=config["vit_hidden_size"],
        vit_intermediate_size=config["vit_intermediate_size"],
        vit_num_hidden_layers=config["vit_num_hidden_layers"],
        vit_num_attention_heads=config["vit_num_attention_heads"],
        vit_image_size=config["vit_image_size"],
        vit_patch_size=config["vit_patch_size"],
        vit_window_size=config["vit_window_size"],
        vit_global_attn_indexes=config["vit_global_attn_indexes"],
        vit_rope_theta=config["vit_rope_theta"],
        vit_pretrain_image_size=config["vit_pretrain_image_size"],
        fpn_hidden_size=config["fpn_hidden_size"],
        fpn_scale_factors=config["fpn_scale_factors"],
        detr_encoder_hidden_size=config["detr_encoder_hidden_size"],
        detr_encoder_num_layers=config["detr_encoder_num_layers"],
        detr_encoder_num_attention_heads=config["detr_encoder_num_attention_heads"],
        detr_encoder_intermediate_size=config["detr_encoder_intermediate_size"],
        detr_encoder_dropout=config["detr_encoder_dropout"],
        detr_decoder_hidden_size=config["detr_decoder_hidden_size"],
        detr_decoder_num_layers=config["detr_decoder_num_layers"],
        detr_decoder_num_queries=config["detr_decoder_num_queries"],
        detr_decoder_num_attention_heads=config["detr_decoder_num_attention_heads"],
        detr_decoder_intermediate_size=config["detr_decoder_intermediate_size"],
        detr_decoder_dropout=config["detr_decoder_dropout"],
        mask_decoder_hidden_size=config["mask_decoder_hidden_size"],
        mask_decoder_num_upsampling_stages=config["mask_decoder_num_upsampling_stages"],
        mask_decoder_num_attention_heads=config["mask_decoder_num_attention_heads"],
        text_hidden_size=config["text_hidden_size"],
        text_projection_dim=config["text_projection_dim"],
        input_shape=input_shape,
        input_tensor=input_tensor,
        name=variant,
        **kwargs,
    )

    if weights in valid_model_weights:
        url = SAM3_WEIGHTS_CONFIG[variant][weights].get("url", "")
        if url:
            load_weights_from_config(variant, weights, model, SAM3_WEIGHTS_CONFIG)
        else:
            print(
                f"Weight URL for '{weights}' is not yet available. "
                "Use the conversion script to generate weights."
            )
    elif weights is not None:
        print(f"Loading weights from file: {weights}")
        model.load_weights(weights)
    else:
        print("No weights loaded.")

    return model


@register_model
def Sam3(
    input_shape=None,
    input_tensor=None,
    weights=None,
    **kwargs,
):
    return _create_sam3_model(
        "Sam3",
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
        **kwargs,
    )
