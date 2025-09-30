import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloModelFusionConfig:
    def __init__(
        self,
        d_model: int = 736,            # model width inside the transformer
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        in_dim: int = 736,             # input feature dim from your YOLO embedder
        use_cls: bool = True,          # prepend learnable [CLS] and use it as fused output
        pooling: str = "cls",          # 'cls' | 'mean' | 'max' (used if use_cls=False)
        pos_encoding: str = "learned", # 'learned' | 'sinusoidal'
        max_seq_len: int = 16,         # supports S up to this (7 or 8 typical; keeping slack)
        device: Optional[str] = None,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.in_dim = in_dim
        self.use_cls = use_cls
        self.pooling = pooling
        self.pos_encoding = pos_encoding
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


    def __str__(self) -> str:
      return (
          f"\n model output dimension: {self.d_model}"
          f"\n nhead: {self.nhead} "
          f"\n dimension of dim_feedforward: {self.dim_feedforward}"
          f"\n dimension of feedforward: {self.num_layers}"
          f"\n dropout: {self.dropout}"
          f"\n seq_len (Note: by Defualt it will be 7 for the 7 directional images): {self.max_seq_len}"
          f"\n Device: {self.device}"
          f"\n use_cls: {self.use_cls}"
          f"\n pos_encoding: {self.pos_encoding}"
          f"\n pooling:{self.pooling}"
          f"\n in dim: {self.in_dim} "
      )

class YoloModelFusion(nn.Module):
    """
    Forward:
      X: (B, S, D_in)  -> returns (fused: (B, d_model), per_view: (B, S, d_model))
      key_padding_mask (optional): (B, S) with True where view is missing/ignore
    """
    def __init__(self, cfg: YoloModelFusionConfig):
        super().__init__()
        self.cfg = cfg

        # project to d_model if needed
        self.proj = nn.Identity() if cfg.in_dim == cfg.d_model else nn.Linear(cfg.in_dim, cfg.d_model)

        # positional encodings
        if cfg.pos_encoding == "learned":
            self.pos_embed = nn.Embedding(cfg.max_seq_len + (1 if cfg.use_cls else 0), cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            self.pos_embed = None
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")

        # optional CLS token
        if cfg.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.register_parameter("cls_token", None)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # <--- accept (B, S, D)
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm_out = nn.LayerNorm(cfg.d_model)

        self.to(cfg.device)

    @staticmethod
    def _sinusoidal_pe(S: int, D: int, device) -> torch.Tensor:
        pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(1)  # (S,1)
        i = torch.arange(D // 2, dtype=torch.float32, device=device)            # (D/2,)
        denom = torch.pow(10000.0, (2 * i) / D)                                 # (D/2,)
        pe = torch.zeros(S, D, device=device)
        pe[:, 0::2] = torch.sin(pos / denom)
        pe[:, 1::2] = torch.cos(pos / denom)
        return pe  # (S, D)

    def forward(
        self,
        X: torch.Tensor,                             # (B, S, D_in)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) True=mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert X.ndim == 3, "X must be (B, S, D_in)"
        B, S, _ = X.shape
        device = X.device

        # project input
        X = self.proj(X)  # (B, S, d_model)

        # prepend CLS if used
        if self.cfg.use_cls:
            cls = self.cls_token.expand(B, 1, self.cfg.d_model)  # (B,1,D)
            X = torch.cat([cls, X], dim=1)                       # (B, S+1, D)
            if key_padding_mask is not None:
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
                key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)  # (B, S+1)
            S_eff = S + 1
        else:
            S_eff = S

        # positions
        if S_eff > self.cfg.max_seq_len:
            raise ValueError(f"S_eff={S_eff} exceeds max_seq_len={self.cfg.max_seq_len}. "
                             f"Increase max_seq_len in config.")

        if self.cfg.pos_encoding == "learned":
            pos_idx = torch.arange(S_eff, device=device)
            pe = self.pos_embed(pos_idx)[None, :, :]    # (1, S_eff, D)
            X = X + pe
        else:
            pe = self._sinusoidal_pe(S_eff, self.cfg.d_model, device)[None, :, :]
            X = X + pe

        # encode
        enc = self.encoder(X, src_key_padding_mask=key_padding_mask)  # (B, S_eff, D)
        enc = self.norm_out(enc)

        # outputs
        if self.cfg.use_cls:
            fused = enc[:, 0, :]     # (B, D)
            per_view = enc[:, 1:, :] # (B, S, D)
        else:
            per_view = enc
            if self.cfg.pooling == "mean":
                if key_padding_mask is not None:
                    keep = (~key_padding_mask).float().unsqueeze(-1)      # (B,S,1)
                    fused = (per_view * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1e-6)
                else:
                    fused = per_view.mean(dim=1)
            elif self.cfg.pooling == "max":
                if key_padding_mask is not None:
                    neg_inf = torch.finfo(per_view.dtype).min
                    fused = per_view.masked_fill(key_padding_mask.unsqueeze(-1), neg_inf).max(dim=1).values
                else:
                    fused = per_view.max(dim=1).values
            else:
                # default to mean if pooling unrecognized
                fused = per_view.mean(dim=1)

        return fused, per_view