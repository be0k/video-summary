import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_attn_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    return ~valid_mask


def downsample_mask(attn_mask: torch.Tensor, stride: int) -> torch.Tensor:
    if attn_mask is None:
        return None
    valid = (~attn_mask).float().unsqueeze(1)
    pooled = F.max_pool1d(valid, kernel_size=stride, stride=stride, padding=0)
    return pooled.squeeze(1) < 0.5


def pad_mask(attn_mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if attn_mask is None or attn_mask.size(1) == target_len:
        return attn_mask
    return F.pad(attn_mask, (0, target_len - attn_mask.size(1)), value=True)


def apply_mask_to_output(x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    if attn_mask is None:
        return x
    return x.masked_fill(attn_mask.unsqueeze(-1), 0.0)


class FFN(nn.Module):
    def __init__(self, dim: int, dropout: float, expand: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        out = self.net(self.norm(x))
        out = apply_mask_to_output(out, attn_mask)
        return x + out


class MambaBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(dim)
        self.ssm_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.ssm_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm(x)
        fwd = self.ssm_fwd(h)
        bwd = self.ssm_bwd(h.flip(1)).flip(1)
        gate = self.gate(torch.cat([fwd, bwd], dim=-1))
        h = gate * fwd + (1.0 - gate) * bwd
        h = apply_mask_to_output(h, attn_mask)
        return x + self.drop(h)


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, modalities: str):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        self.modalities = modalities
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _attend(self, query: torch.Tensor, context: torch.Tensor, batch: int, steps: int, channels: int) -> torch.Tensor:
        bt = batch * steps
        num_context = context.size(1)
        q = self.q(query).view(bt, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(bt, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(bt, num_context, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(bt, 1, channels)
        return self.proj_drop(self.o(out)).view(batch, steps, channels)

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, channels = visual.shape
        bt = batch * steps

        tokens = []
        if "v" in self.modalities:
            tokens.append(("v", visual.view(bt, 1, channels)))
        if "t" in self.modalities:
            tokens.append(("t", text.view(bt, 1, channels)))
        if "a" in self.modalities:
            tokens.append(("a", audio.view(bt, 1, channels)))

        context = torch.cat([token for _, token in tokens], dim=1)
        outputs = {"v": visual, "t": text, "a": audio}
        token_map = {name: token for name, token in tokens}

        for name in self.modalities:
            delta = self._attend(token_map[name], context, batch, steps, channels)
            delta = apply_mask_to_output(delta, attn_mask)
            outputs[name] = outputs[name] + delta

        return outputs["v"], outputs["t"], outputs["a"]


class CrossModalFusionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, modalities: str):
        super().__init__()
        self.cross_attn = CrossModalAttention(dim, num_heads, dropout, modalities)
        self.ffn = FFN(dim, dropout)
        self.modalities = modalities

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, text, audio = self.cross_attn(visual, text, audio, attn_mask)
        if "v" in self.modalities:
            visual = self.ffn(visual, attn_mask)
        if "t" in self.modalities:
            text = self.ffn(text, attn_mask)
        if "a" in self.modalities:
            audio = self.ffn(audio, attn_mask)
        return visual, text, audio


class SoftmaxGatedFusion(nn.Module):
    def __init__(self, dim: int, modalities: str):
        super().__init__()
        self.gates = nn.ModuleDict({name: nn.Linear(dim, 1) for name in modalities})
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        names = list(feats.keys())
        logits = torch.cat([self.gates[name](feats[name]) for name in names], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        fused = sum(weights[..., idx : idx + 1] * feats[name] for idx, name in enumerate(names))
        weight_map = {name: weights[..., idx] for idx, name in enumerate(names)}
        return self.norm(self.proj(fused)), weight_map


class CrossModalFusionStack(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        modalities: str,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.modalities = modalities
        self.modality_dropout_prob = modality_dropout_prob
        self.layers = nn.ModuleList(
            [CrossModalFusionLayer(dim, num_heads, dropout, modalities) for _ in range(num_layers)]
        )
        self.fusion = SoftmaxGatedFusion(dim, modalities)

    def forward(
        self,
        visual: torch.Tensor = None,
        text: torch.Tensor = None,
        audio: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats = {}
        if visual is not None and "v" in self.modalities:
            feats["v"] = visual
        if text is not None and "t" in self.modalities:
            feats["t"] = text
        if audio is not None and "a" in self.modalities:
            feats["a"] = audio
        if not feats:
            raise ValueError("At least one modality is required")

        if self.training and self.modality_dropout_prob > 0:
            available_modalities = list(feats.keys())
            random.shuffle(available_modalities)
            kept_modalities = [
                name for name in available_modalities if random.random() > self.modality_dropout_prob
            ]
            if not kept_modalities:
                kept_modalities = [random.choice(available_modalities)]
            for name in available_modalities:
                if name not in kept_modalities:
                    feats[name] = torch.zeros_like(feats[name])

        def get_feat(name: str) -> torch.Tensor:
            if name in feats:
                return feats[name]
            return torch.zeros_like(next(iter(feats.values())))

        visual, text, audio = get_feat("v"), get_feat("t"), get_feat("a")
        for layer in self.layers:
            visual, text, audio = layer(visual, text, audio, attn_mask)

        updated_feats = {}
        if "v" in self.modalities:
            updated_feats["v"] = visual
        if "t" in self.modalities:
            updated_feats["t"] = text
        if "a" in self.modalities:
            updated_feats["a"] = audio

        fused, weights = self.fusion(updated_feats)
        fused = apply_mask_to_output(fused, attn_mask)
        return fused, weights


class StridedConv1d(nn.Module):
    def __init__(self, dim: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=stride * 2 - 1,
            stride=stride,
            padding=stride - 1,
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = apply_mask_to_output(x, attn_mask)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(self.norm(x))


class TemporalUpsample(nn.Module):
    def __init__(self, dim: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = x.repeat_interleave(self.stride, dim=1)
        if x.size(1) > target_len:
            x = x[:, :target_len]
        elif x.size(1) < target_len:
            pad = x[:, -1:].expand(-1, target_len - x.size(1), -1)
            x = torch.cat([x, pad], dim=1)
        return self.norm(self.act(self.proj(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, stride: int = 2):
        super().__init__()
        self.mamba = MambaBlock(dim, dropout)
        self.ffn = FFN(dim, dropout)
        self.compress = StridedConv1d(dim, stride)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mamba(x, attn_mask)
        x = self.ffn(x, attn_mask)
        skip = x
        x = self.compress(x, attn_mask)
        return x, skip


class GistBottleneck(nn.Module):
    def __init__(self, dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([MambaBlock(dim, dropout), FFN(dim, dropout)]) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        for mamba, ffn in self.layers:
            x = mamba(x, attn_mask)
            x = ffn(x, attn_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, stride: int = 2):
        super().__init__()
        self.upsample = TemporalUpsample(dim, stride)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, dropout)
        self.mamba = MambaBlock(dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, dec_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x, target_len=skip.size(1))
        x = self.norm(apply_mask_to_output(x + skip, dec_mask))
        x = self.ffn(x, dec_mask)
        x = self.mamba(x, dec_mask)
        return x
