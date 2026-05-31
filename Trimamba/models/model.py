import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CrossModalFusionStack,
    DecoderBlock,
    EncoderBlock,
    GistBottleneck,
    apply_mask_to_output,
    downsample_mask,
    make_attn_mask,
    pad_mask,
)
from .layers import ModalityEmbedding


class TriMamba(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        audio_dim: int,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_encoder_layers: int = 2,
        num_bottleneck_layers: int = 2,
        num_fusion_layers: int = 1,
        stride: int = 4,
        modalities: str = "vta",
        get_attn_weights: bool = False,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.stride = stride
        self.modalities = modalities
        self.get_attn_weights = get_attn_weights
        self._pad_multiple = stride ** num_encoder_layers

        self.visual_proj = nn.Linear(visual_dim, input_dim)
        self.text_proj = nn.Linear(text_dim, input_dim)
        self.audio_proj = nn.Linear(audio_dim, input_dim)
        self.visual_ln = nn.LayerNorm(input_dim)
        self.text_ln = nn.LayerNorm(input_dim)
        self.audio_ln = nn.LayerNorm(input_dim)
        self.modality_embedding = ModalityEmbedding(input_dim)

        self.fusion_stack = CrossModalFusionStack(
            dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_fusion_layers,
            modalities=modalities,
            modality_dropout_prob=modality_dropout_prob,
        )
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(input_dim, dropout, stride) for _ in range(num_encoder_layers)]
        )
        self.bottleneck = GistBottleneck(input_dim, num_bottleneck_layers, dropout)
        self.bottleneck_norm = nn.LayerNorm(input_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(input_dim, dropout, stride) for _ in range(num_encoder_layers)]
        )
        self.final_norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _pad_to_multiple(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.size(1) % self._pad_multiple
        if remainder == 0:
            return x
        return F.pad(x, (0, 0, 0, self._pad_multiple - remainder))

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original_steps = visual.size(1)

        visual = self.visual_ln(self.visual_proj(visual))
        text = self.text_ln(self.text_proj(text))
        audio = self.audio_ln(self.audio_proj(audio))

        visual = self._pad_to_multiple(visual)
        text = self._pad_to_multiple(text)
        audio = self._pad_to_multiple(audio)
        padded_steps = visual.size(1)

        attn_mask = pad_mask(make_attn_mask(mask), padded_steps) if mask is not None else None

        visual = apply_mask_to_output(visual, attn_mask)
        text = apply_mask_to_output(text, attn_mask)
        audio = apply_mask_to_output(audio, attn_mask)

        visual = visual + self.modality_embedding(visual, modality_index=0)
        text = text + self.modality_embedding(text, modality_index=1)
        audio = audio + self.modality_embedding(audio, modality_index=2)

        x, weights = self.fusion_stack(visual, text, audio, attn_mask)

        skips = []
        skip_masks = []
        enc_mask = attn_mask
        for block in self.encoder_blocks:
            x, skip = block(x, enc_mask)
            skips.append(skip)
            skip_masks.append(enc_mask)
            enc_mask = downsample_mask(enc_mask, self.stride)

        x = self.bottleneck(x, enc_mask)
        x = apply_mask_to_output(x, enc_mask)

        for index, block in enumerate(self.decoder_blocks):
            skip_index = self.num_encoder_layers - 1 - index
            dec_mask = skip_masks[skip_index]
            x = block(x, skips[skip_index], dec_mask=dec_mask)

        x = self.final_norm(x[:, :original_steps, :])
        return self.head(x).squeeze(-1), weights
    
