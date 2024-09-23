# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from model.seg import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
)
from model.nn.norm import set_norm_eps
from model.utils.network import load_state_dict_from_file
import torch
from collections import OrderedDict
__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: dict[str, dict[str, str]] = {
    "bdd": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
    },

}


def create_seg_model(
    name: str, dataset: str, multitask: str, pretrained=True, 
    backbone_weight_url:str or None =None, 
    weight_url: str or None = None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset,multitask=multitask, **kwargs)

    if model_id in ["l1", "l2"]:
        set_norm_eps(model, 1e-7)

    if pretrained:
        if backbone_weight_url is None:
            weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
            if weight_url is None:
                raise ValueError(f"Do not find the pretrained weight of {name}.")
            else:
                weight = load_state_dict_from_file(weight_url)
                model.load_state_dict(weight)
        else:
            weights= load_state_dict_from_file(backbone_weight_url)
            backbone_weights = OrderedDict()
            for w in weights.keys():
                if 'backbone' in w:
                    weight=weights[w]
                    w=w.replace('backbone.','')
                    backbone_weights[w]=weight

            model.backbone.load_state_dict(backbone_weights)

    return model