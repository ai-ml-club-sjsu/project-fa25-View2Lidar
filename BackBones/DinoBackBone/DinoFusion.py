import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class DepthModelFusionConfig:
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        seq_len: int = 7,              # expected number of camera directions (Argoverse2 ring=7)
        use_cls: bool = True,          # prepend a learnable [CLS] token and use it as fused output
        pos_encoding: str = "learned", # 'learned' or 'sinusoidal'
        pooling: str = "cls",          # 'cls' | 'mean' | 'max' (fallback if use_cls=False)
        in_dim: int = 512,             # input feature dim; will be projected to d_model if different
        device: Optional[str] = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.seq_len = seq_len
        self.use_cls = use_cls
        self.pos_encoding = pos_encoding
        self.pooling = pooling
        self.in_dim = in_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self) -> str:
      return (
          f"\n model output dimension: {self.d_model}"
          f"\n nhead: {self.nhead} "
          f"\n dimension of dim_feedforward: {self.dim_feedforward}"
          f"\n dimension of feedforward: {self.num_layers}"
          f"\n dropout: {self.dropout}"
          f"\n seq_len (Note: by Defualt it will be 7 for the 7 directional images): {self.seq_len}"
          f"\n Device: {self.device}"
          f"\n use_cls: {self.use_cls}"
          f"\n pos_encoding: {self.pos_encoding}"
          f"\n pooling:{self.pooling}"
          f"\n in dim: {self.in_dim} "
      )


class DinoFusion(nn.Module):
    """
    Input:  X (B, S, D_in)  e.g., S=7 camera directions, D_in=512
    Mask:   key_padding_mask (B, S) with True where positions are padding/missing (optional)
    Output: fused (B, d_model), per_view (B, S, d_model)
    """
    def __init__(self, cfg: DepthModelFusionConfig):
        super().__init__()
        self.cfg = cfg

        # project input to d_model if needed
        self.proj = nn.Identity().to(self.cfg.device) if cfg.in_dim == cfg.d_model else nn.Linear(cfg.in_dim, cfg.d_model).to(self.cfg.device)

        # positional encoding
        if cfg.pos_encoding == "learned":
            self.pos_embed = nn.Embedding(cfg.seq_len + (1 if cfg.use_cls else 0), cfg.d_model).to(self.cfg.device)
        elif cfg.pos_encoding == "sinusoidal":
            self.pos_embed = None  # computed on the fly
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")

        # optional CLS token
        if cfg.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)).to(cfg.device)
        else:
            self.register_parameter("cls_token", None)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # <--- important to accept (B, S, D)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm_out = nn.LayerNorm(cfg.d_model)

        self.to(cfg.device)

    def _sinusoidal_pe(self, S: int, D: int, device) -> torch.Tensor:
        """(S, D) sinusoidal PE."""
        pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(1)           # (S,1)
        i = torch.arange(D // 2, dtype=torch.float32, device=self.cfg.device)                     # (D/2,)
        denom = torch.pow(10000.0, (2 * i) / D).to(self.cfg.device)                                          # (D/2,)
        pe = torch.zeros(S, D, device=self.cfg.device)
        pe[:, 0::2] = torch.sin(pos / denom).to(self.cfg.device)
        pe[:, 1::2] = torch.cos(pos / denom).to(self.cfg.device)
        return pe  # (S, D)

    def forward(
        self,
        X: torch.Tensor,                              # (B, S, D_in)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) True=mask/ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert X.ndim == 3, "X must be (B, S, D)"
        B, S, Din = X.shape
        device = self.cfg.device

        # project to model dim
        X = self.proj(X)  # (B, S, d_model)

        # build positions (0..S-1), add CLS if configured
        if self.cfg.use_cls:
            cls = self.cls_token.expand(B, 1, self.cfg.d_model).to(self.cfg.device)  # (B,1,D)
            X = torch.cat([cls, X], dim=1).to(self.cfg.device)                       # (B, S+1, D)
            if key_padding_mask is not None:
                # prepend a non-masked token for CLS
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=self.cfg.device)
                key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1).to(self.cfg.device)  # (B, S+1)
            S_eff = S + 1
            pos_idx = torch.arange(S_eff, device=self.cfg.device)
        else:
            S_eff = S
            pos_idx = torch.arange(S_eff, device=self.cfg.device)

        # add positional encodings
        if self.cfg.pos_encoding == "learned":
            pe = self.pos_embed(pos_idx)[None, :, :].to(self.cfg.device)            # (1, S_eff, D)
            X = X + pe
        else:
            pe = self._sinusoidal_pe(S_eff, self.cfg.d_model, self.cfg.device)[None, :, :]  # (1,S_eff,D)
            X = X + pe

        # encode
        enc = self.encoder(X, src_key_padding_mask=key_padding_mask)  # (B, S_eff, D)
        enc = self.norm_out(enc)

        # split per-view (exclude CLS if used)
        if self.cfg.use_cls:
            fused_token = enc[:, 0, :]                # (B, D)
            per_view = enc[:, 1:, :]                  # (B, S, D)
        else:
            per_view = enc                            # (B, S, D)
            # fallback pooling
            if self.cfg.pooling == "mean":
                if key_padding_mask is not None:
                    # mask-aware mean
                    keep = (~key_padding_mask).float().unsqueeze(-1)  # (B,S,1)
                    fused_token = (per_view * keep).sum(dim=1) / (keep.sum(dim=1).clamp_min(1e-6))
                else:
                    fused_token = per_view.mean(dim=1)
            elif self.cfg.pooling == "max":
                if key_padding_mask is not None:
                    neg_inf = torch.finfo(per_view.dtype).min
                    masked = per_view.masked_fill(key_padding_mask.unsqueeze(-1), neg_inf)
                    fused_token = masked.max(dim=1).values
                else:
                    fused_token = per_view.max(dim=1).values
            else:
                # default to mean if no CLS and unknown pooling
                fused_token = per_view.mean(dim=1)

        return fused_token, per_view  # (B, D), (B, S, D)
