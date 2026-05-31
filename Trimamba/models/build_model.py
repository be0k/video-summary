from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

from .model import TriMamba


MODEL_ALIASES = {"trimamba", "triplesumm"}


def build_model(cfg):
    if cfg.model not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {cfg.model}")

    return TriMamba(
        visual_dim=cfg.visual_dim,
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        get_attn_weights=cfg.get_attn_weights,
        num_encoder_layers=cfg.num_encoder_layers,
        num_bottleneck_layers=cfg.num_bottleneck_layers,
        num_fusion_layers=cfg.num_fusion_layers,
        stride=cfg.stride,
        modalities=cfg.modalities,
        modality_dropout_prob=cfg.modality_dropout_prob,
    )


def build_optimizer(cfg, model):
    if cfg.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    return optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(cfg, optimizer, data_len):
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.num_epochs,
            eta_min=0,
        )
    elif cfg.scheduler == "cosine_hf":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=getattr(cfg, "num_warmup_steps", (cfg.num_epochs * data_len) / 10),
            num_training_steps=cfg.num_epochs * data_len,
        )
    elif cfg.scheduler == "reduce":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.1,
            patience=4,
        )
    elif cfg.scheduler == "restart":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
            last_epoch=-1,
        )
    else:
        scheduler = None

    return scheduler, cfg.scheduler
