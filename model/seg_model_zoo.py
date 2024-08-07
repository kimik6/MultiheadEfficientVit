# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from model.seg import (
    EfficientViTSeg,
    efficientvit_seg_b0,
)
from model.nn.norm import set_norm_eps
from model.utils.network import load_state_dict_from_file

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: dict[str, dict[str, str]] = {
    "bdd": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
    },

}


def create_seg_model(
    name: str, dataset: str, pretrained=True, weight_url: str or None = None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,

    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)

    if model_id in ["l1", "l2"]:
        set_norm_eps(model, 1e-7)

    if pretrained:
        weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model