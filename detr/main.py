import torch
import torch.nn as nn
import torchvision.transforms as transforms
from detr.models.detr_vae import build_vae

def build_ACT_model_and_optimizer(config):
    model = build_vae(config)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config['lr_backbone'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    return model, optimizer