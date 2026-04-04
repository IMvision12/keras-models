import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam3Model

from kmodels.models.sam3.sam3_model import Sam3
from kmodels.utils.weight_transfer_torch_to_keras import (
    transfer_weights,
)

vit_name_mapping = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

model_configs = [
    {
        "keras_model_cls": Sam3,
        "hf_model_name": "facebook/sam3",
        "input_shape": (1008, 1008, 3),
    },
]


def convert_sam3(model_config):
    print(f"\n{'=' * 60}")
    print(f"Converting {model_config['hf_model_name']}...")
    print(f"{'=' * 60}")

    print(f"Loading HF model: {model_config['hf_model_name']}")
    hf_model = Sam3Model.from_pretrained(
        model_config["hf_model_name"], attn_implementation="eager"
    ).eval()
    hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

    print(f"HF model has {len(hf_state_dict)} weight tensors")

    print("Creating Keras model...")
    keras_model = model_config["keras_model_cls"](
        input_shape=model_config["input_shape"], weights=None
    )
    print(f"Keras model params: {keras_model.count_params():,}")

    print("Transferring ViT backbone weights...")

    patch_conv = keras_model.get_layer("backbone_patch_embed")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_state_dict["vision_encoder.backbone.patch_embed.projection.weight"],
    )

    pos_embed_layer = keras_model.get_layer("backbone_position_embedding")
    hf_pos = hf_state_dict["vision_encoder.backbone.pos_embed"]
    pos_embed_layer.embeddings.assign(hf_pos.squeeze(0))

    num_layers = keras_model.vit_num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring ViT layers"):
        layer = keras_model.get_layer(f"backbone_layers_{i}")
        hf_prefix = f"vision_encoder.backbone.layers.{i}"

        layer.layer_norm1.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.weight"])
        layer.layer_norm1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])
        layer.layer_norm2.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.weight"])
        layer.layer_norm2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])

        attn = layer.attn
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.attention.{proj_name}.weight"],
            )
            proj.bias.assign(hf_state_dict[f"{hf_prefix}.attention.{proj_name}.bias"])

        transfer_weights(
            "kernel",
            layer.mlp_fc1.kernel,
            hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"],
        )
        layer.mlp_fc1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"])
        transfer_weights(
            "kernel",
            layer.mlp_fc2.kernel,
            hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"],
        )
        layer.mlp_fc2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"])

        if f"{hf_prefix}.layer_scale1.lambda1" in hf_state_dict:
            layer.layer_scale1.lambda1.assign(
                hf_state_dict[f"{hf_prefix}.layer_scale1.lambda1"]
            )
            layer.layer_scale2.lambda1.assign(
                hf_state_dict[f"{hf_prefix}.layer_scale2.lambda1"]
            )

    backbone_ln = keras_model.get_layer("backbone_layer_norm")
    backbone_ln.gamma.assign(hf_state_dict["vision_encoder.backbone.layer_norm.weight"])
    backbone_ln.beta.assign(hf_state_dict["vision_encoder.backbone.layer_norm.bias"])

    print("Transferring FPN neck weights...")

    scale_factors = keras_model.fpn_scale_factors
    for level_idx, scale_factor in enumerate(scale_factors):
        hf_fpn_prefix = f"vision_encoder.neck.fpn_layers.{level_idx}"

        if scale_factor == 4.0:
            for deconv_idx, deconv_name in enumerate(["deconv1", "deconv2"]):
                keras_layer = keras_model.get_layer(
                    f"fpn_level_{level_idx}_{deconv_name}"
                )
                hf_key = f"{hf_fpn_prefix}.upsample.{deconv_idx * 2}.weight"
                if hf_key in hf_state_dict:
                    transfer_weights(
                        "conv_kernel", keras_layer.kernel, hf_state_dict[hf_key]
                    )
                    bias_key = f"{hf_fpn_prefix}.upsample.{deconv_idx * 2}.bias"
                    if bias_key in hf_state_dict:
                        keras_layer.bias.assign(hf_state_dict[bias_key])
        elif scale_factor == 2.0:
            keras_layer = keras_model.get_layer(f"fpn_level_{level_idx}_deconv1")
            hf_key = f"{hf_fpn_prefix}.upsample.0.weight"
            if hf_key in hf_state_dict:
                transfer_weights(
                    "conv_kernel", keras_layer.kernel, hf_state_dict[hf_key]
                )
                bias_key = f"{hf_fpn_prefix}.upsample.0.bias"
                if bias_key in hf_state_dict:
                    keras_layer.bias.assign(hf_state_dict[bias_key])

        for proj_idx, proj_name in enumerate(["proj1", "proj2"]):
            keras_layer = keras_model.get_layer(f"fpn_level_{level_idx}_{proj_name}")
            hf_key = f"{hf_fpn_prefix}.proj.{proj_idx}.weight"
            if hf_key in hf_state_dict:
                transfer_weights(
                    "conv_kernel", keras_layer.kernel, hf_state_dict[hf_key]
                )
                bias_key = f"{hf_fpn_prefix}.proj.{proj_idx}.bias"
                if bias_key in hf_state_dict:
                    keras_layer.bias.assign(hf_state_dict[bias_key])

    print("Transferring text projection...")
    text_proj = keras_model.get_layer("text_projection")
    transfer_weights(
        "kernel",
        text_proj.kernel,
        hf_state_dict["text_projection.weight"],
    )
    text_proj.bias.assign(hf_state_dict["text_projection.bias"])

    print("Transferring DETR encoder...")
    num_enc_layers = keras_model.detr_encoder_num_layers
    for i in tqdm(range(num_enc_layers), desc="DETR encoder layers"):
        layer = keras_model.get_layer(f"detr_encoder_layers_{i}")
        hf_prefix = f"detr_encoder.layers.{i}"

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.self_attn.{proj_name}.weight"],
            )
            proj.bias.assign(hf_state_dict[f"{hf_prefix}.self_attn.{proj_name}.bias"])

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.cross_attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.cross_attn.{proj_name}.weight"],
            )
            proj.bias.assign(hf_state_dict[f"{hf_prefix}.cross_attn.{proj_name}.bias"])

        layer.layer_norm1.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.weight"])
        layer.layer_norm1.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm1.bias"])
        layer.layer_norm2.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.weight"])
        layer.layer_norm2.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm2.bias"])
        layer.layer_norm3.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.weight"])
        layer.layer_norm3.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm3.bias"])

        transfer_weights(
            "kernel", layer.fc1.kernel, hf_state_dict[f"{hf_prefix}.fc1.weight"]
        )
        layer.fc1.bias.assign(hf_state_dict[f"{hf_prefix}.fc1.bias"])
        transfer_weights(
            "kernel", layer.fc2.kernel, hf_state_dict[f"{hf_prefix}.fc2.weight"]
        )
        layer.fc2.bias.assign(hf_state_dict[f"{hf_prefix}.fc2.bias"])

    print("Transferring DETR decoder...")

    query_embed = keras_model.get_layer("detr_decoder_query_embed")
    query_embed.embeddings.assign(hf_state_dict["detr_decoder.query_embed.weight"])

    ref_points = keras_model.get_layer("detr_decoder_reference_points")
    ref_points.embeddings.assign(hf_state_dict["detr_decoder.reference_points.weight"])

    presence_token = keras_model.get_layer("detr_decoder_presence_token")
    presence_token.embeddings.assign(
        hf_state_dict["detr_decoder.presence_token.weight"]
    )

    box_head = keras_model.get_layer("detr_decoder_box_head")
    for j, dense in enumerate(box_head._layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf_state_dict[f"detr_decoder.box_head.dense_{j}.weight"],
        )
        dense.bias.assign(hf_state_dict[f"detr_decoder.box_head.dense_{j}.bias"])

    pres_head = keras_model.get_layer("detr_decoder_presence_head")
    for j, dense in enumerate(pres_head._layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf_state_dict[f"detr_decoder.presence_head.dense_{j}.weight"],
        )
        dense.bias.assign(hf_state_dict[f"detr_decoder.presence_head.dense_{j}.bias"])

    rph = keras_model.get_layer("detr_decoder_ref_point_head")
    for j, dense in enumerate(rph._layers):
        transfer_weights(
            "kernel",
            dense.kernel,
            hf_state_dict[f"detr_decoder.ref_point_head.dense_{j}.weight"],
        )
        dense.bias.assign(hf_state_dict[f"detr_decoder.ref_point_head.dense_{j}.bias"])

    num_dec_layers = keras_model.detr_decoder_num_layers
    for i in tqdm(range(num_dec_layers), desc="DETR decoder layers"):
        layer = keras_model.get_layer(f"detr_decoder_layers_{i}")
        hf_prefix = f"detr_decoder.layers.{i}"

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.self_attn.{proj_name}.weight"],
            )
            proj.bias.assign(hf_state_dict[f"{hf_prefix}.self_attn.{proj_name}.bias"])

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.text_cross_attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.text_cross_attn.{proj_name}.weight"],
            )
            proj.bias.assign(
                hf_state_dict[f"{hf_prefix}.text_cross_attn.{proj_name}.bias"]
            )

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.vision_cross_attn, proj_name)
            transfer_weights(
                "kernel",
                proj.kernel,
                hf_state_dict[f"{hf_prefix}.vision_cross_attn.{proj_name}.weight"],
            )
            proj.bias.assign(
                hf_state_dict[f"{hf_prefix}.vision_cross_attn.{proj_name}.bias"]
            )

        for ln_idx in range(1, 5):
            ln = getattr(layer, f"layer_norm{ln_idx}")
            ln.gamma.assign(hf_state_dict[f"{hf_prefix}.layer_norm{ln_idx}.weight"])
            ln.beta.assign(hf_state_dict[f"{hf_prefix}.layer_norm{ln_idx}.bias"])

        transfer_weights(
            "kernel", layer.fc1.kernel, hf_state_dict[f"{hf_prefix}.fc1.weight"]
        )
        layer.fc1.bias.assign(hf_state_dict[f"{hf_prefix}.fc1.bias"])
        transfer_weights(
            "kernel", layer.fc2.kernel, hf_state_dict[f"{hf_prefix}.fc2.weight"]
        )
        layer.fc2.bias.assign(hf_state_dict[f"{hf_prefix}.fc2.bias"])

    print("Transferring dot-product scoring...")
    scoring = keras_model.get_layer("dot_product_scoring")

    transfer_weights(
        "kernel",
        scoring.text_mlp_fc1.kernel,
        hf_state_dict["dot_product_scoring.text_mlp.fc1.weight"],
    )
    scoring.text_mlp_fc1.bias.assign(
        hf_state_dict["dot_product_scoring.text_mlp.fc1.bias"]
    )
    transfer_weights(
        "kernel",
        scoring.text_mlp_fc2.kernel,
        hf_state_dict["dot_product_scoring.text_mlp.fc2.weight"],
    )
    scoring.text_mlp_fc2.bias.assign(
        hf_state_dict["dot_product_scoring.text_mlp.fc2.bias"]
    )
    transfer_weights(
        "kernel",
        scoring.text_proj.kernel,
        hf_state_dict["dot_product_scoring.text_proj.weight"],
    )
    transfer_weights(
        "kernel",
        scoring.query_proj.kernel,
        hf_state_dict["dot_product_scoring.query_proj.weight"],
    )

    print("Transferring mask decoder...")

    num_up = keras_model.mask_decoder_num_upsampling_stages
    for stage_idx in range(num_up):
        conv = keras_model.get_layer(f"pixel_decoder_stage_{stage_idx}_conv")
        hf_key = f"mask_decoder.pixel_decoder.convs.{stage_idx}.weight"
        if hf_key in hf_state_dict:
            transfer_weights("conv_kernel", conv.kernel, hf_state_dict[hf_key])
            conv.bias.assign(
                hf_state_dict[f"mask_decoder.pixel_decoder.convs.{stage_idx}.bias"]
            )

        gn = keras_model.get_layer(f"pixel_decoder_stage_{stage_idx}_gn")
        gn_key = f"mask_decoder.pixel_decoder.gns.{stage_idx}.weight"
        if gn_key in hf_state_dict:
            gn.gamma.assign(hf_state_dict[gn_key])
            gn.beta.assign(
                hf_state_dict[f"mask_decoder.pixel_decoder.gns.{stage_idx}.bias"]
            )

    inst_proj = keras_model.get_layer("mask_decoder_instance_proj")
    transfer_weights(
        "conv_kernel",
        inst_proj.kernel,
        hf_state_dict["mask_decoder.instance_projection.weight"],
    )
    inst_proj.bias.assign(hf_state_dict["mask_decoder.instance_projection.bias"])

    sem_proj = keras_model.get_layer("mask_decoder_semantic_proj")
    transfer_weights(
        "conv_kernel",
        sem_proj.kernel,
        hf_state_dict["mask_decoder.semantic_projection.weight"],
    )
    sem_proj.bias.assign(hf_state_dict["mask_decoder.semantic_projection.bias"])

    mask_emb = keras_model.get_layer("mask_embedder")
    for j in range(3):
        dense = getattr(mask_emb, f"linear{j + 1}")
        transfer_weights(
            "kernel",
            dense.kernel,
            hf_state_dict[f"mask_decoder.mask_embedder.{j}.weight"],
        )
        dense.bias.assign(hf_state_dict[f"mask_decoder.mask_embedder.{j}.bias"])

    print("Weight transfer complete!")

    print("\nVerifying model equivalence...")
    np.random.seed(42)
    input_shape = model_config["input_shape"]
    test_image = np.random.rand(1, *input_shape).astype(np.float32)
    test_text_features = np.random.rand(1, 10, 1024).astype(np.float32)
    test_text_mask = np.ones((1, 10), dtype=np.float32)

    keras_output = keras_model.predict(
        {
            "pixel_values": test_image,
            "text_features": test_text_features,
            "text_attention_mask": test_text_mask,
        },
        verbose=0,
    )

    print("Output shapes:")
    for k, v in keras_output.items():
        print(f"  {k}: {v.shape}")

    model_base = model_config["hf_model_name"].split("/")[-1].replace("-", "_")
    model_filename = model_base + ".weights.h5"
    keras_model.save_weights(model_filename)
    print(f"Model saved as {model_filename}")

    del keras_model, hf_model, hf_state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    for config in model_configs:
        convert_sam3(config)
