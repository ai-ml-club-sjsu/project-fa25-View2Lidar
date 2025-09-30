import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalFusionConfig:
    def __init__(
        self,
        d_model: int = 736,           # transformer width
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        depth_in: int = 512,          # depth token dim coming in
        yolo_in: int = 736,           # yolo token dim coming in
        use_cls: bool = True,         # prepend [CLS] and use it as fused output
        pooling: str = "cls",         # fallback pooling if use_cls=False: 'mean'|'max'
        pos_encoding: str = "learned",# 'learned'|'sinusoidal'
        max_seq_len: int = 8,         # max #views per modality (7 typical; keep slack)
        device: Optional[str] = None,
    ) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.depth_in = depth_in
        self.yolo_in = yolo_in
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
          f"\n Yolo in :{self.yolo_in}"
          f"\n Device: {self.device}"
          f"\n use_cls: {self.use_cls}"
          f"\n pos_encoding: {self.pos_encoding}"
          f"\n pooling: {self.pooling}"
          f"\n depth in: {self.depth_in}")

class GlobalFusion(nn.Module):
    """
    Inputs:
      depth_matrix:  (B, S, Dd)   e.g., (B,7,512)
      objects_matrix:(B, S, Do)   e.g., (B,7,736)
      masks (optional):
        depth_mask:  (B, S) True = pad/missing
        yolo_mask:   (B, S) True = pad/missing
    Returns:
      fused:         (B, d_model)
      depth_tokens:  (B, S, d_model)
      yolo_tokens:   (B, S, d_model)
    """
    def __init__(self, cfg: GlobalFusionConfig):
        super().__init__()
        self.cfg = cfg

        # Project to shared width
        self.depth_proj = nn.Linear(cfg.depth_in, cfg.d_model) if cfg.depth_in != cfg.d_model else nn.Identity().to(self.cfg.device)
        self.yolo_proj  = nn.Linear(cfg.yolo_in,  cfg.d_model) if cfg.yolo_in  != cfg.d_model else nn.Identity().to(self.cfg.device)

        # Positional & modality embeddings
        if cfg.pos_encoding == "learned":
            self.pos_embed = nn.Embedding(cfg.max_seq_len * 2 + (1 if cfg.use_cls else 0), cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            self.pos_embed = None
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'")
        # modality/type: 0 = depth, 1 = yolo
        self.type_embed = nn.Embedding(2 + (1 if cfg.use_cls else 0), cfg.d_model).to(self.cfg.device)

        # Optional CLS
        if cfg.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.register_parameter("cls_token", None)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # accept (B, L, D)
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm_out = nn.LayerNorm(cfg.d_model)

        self.to(cfg.device)

    @staticmethod
    def _sinusoidal_pe(self,L: int, D: int, device) -> torch.Tensor:
        pos = torch.arange(L, dtype=torch.float32, device=self.cfg.device).unsqueeze(1)   # (L,1)
        i = torch.arange(D // 2, dtype=torch.float32, device=self.cfg.device)             # (D/2,)
        denom = torch.pow(10000.0, (2 * i) / D).to(self.cfg.device)
        pe = torch.zeros(L, D, device=self.cfg.device)
        pe[:, 0::2] = torch.sin(pos / denom).to(self.cfg.device)
        pe[:, 1::2] = torch.cos(pos / denom).to(self.cfg.device)
        return pe

    def forward(
        self,
        depth_matrix: torch.Tensor,                # (B,S,Dd)
        objects_matrix: torch.Tensor,              # (B,S,Do)
        depth_mask: Optional[torch.Tensor] = None, # (B,S) True=pad
        yolo_mask: Optional[torch.Tensor] = None,  # (B,S) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert depth_matrix.ndim == 3 and objects_matrix.ndim == 3, "Inputs must be (B,S,D)"
        B, Sd, _ = depth_matrix.shape
        Bo, So, _ = objects_matrix.shape
        assert B == Bo and Sd == So, "Depth and YOLO must share (B,S)"

        device = self.cfg.device
        S = Sd

        if self.cfg.use_cls:
            L_eff = 1 + 2 * S
        else:
            L_eff = 2 * S
        if L_eff > (self.cfg.max_seq_len * 2 + (1 if self.cfg.use_cls else 0)):
            raise ValueError(f"Sequence length {L_eff} exceeds configured capacity; increase max_seq_len.")

        # Project
        Dd = self.depth_proj(depth_matrix)    # (B,S,D)
        Do = self.yolo_proj(objects_matrix)   # (B,S,D)

        # Build sequence: [CLS?] + depth(0..S-1) + yolo(0..S-1)
        tokens = []
        type_ids = []
        if self.cfg.use_cls:
            cls = self.cls_token.expand(B, 1, self.cfg.d_model)   # (B,1,D)
            tokens.append(cls);             type_ids.append(torch.full((B,1), 2, device=self.cfg.device, dtype=torch.long))  # 2 for CLS

        tokens.append(Dd);                   type_ids.append(torch.zeros(B, S, device=device, dtype=torch.long))     # 0=depth
        tokens.append(Do);                   type_ids.append(torch.ones (B, S, device=device, dtype=torch.long))     # 1=yolo
        X = torch.cat(tokens, dim=1)        # (B, L_eff, D)
        ttype = torch.cat(type_ids, dim=1)  # (B, L_eff)

        # Positions: depth views 0..S-1, yolo views 0..S-1 (offset by S)
        if self.cfg.use_cls:
            pos_idx = torch.cat([
                torch.zeros(B,1, device=device, dtype=torch.long),                  # CLS at 0
                torch.arange(1, S+1, device=device).view(1,S).expand(B,S),          # depth 1..S
                torch.arange(S+1, 2*S+1, device=device).view(1,S).expand(B,S),      # yolo  S+1..2S
            ], dim=1)
        else:
            pos_idx = torch.cat([
                torch.arange(0, S, device=device).view(1,S).expand(B,S),
                torch.arange(S, 2*S, device=device).view(1,S).expand(B,S),
            ], dim=1)

        # Add embeddings
        if self.cfg.pos_encoding == "learned":
            pe = self.pos_embed(pos_idx)                    # (B, L_eff, D)
        else:
            # sinusoidal shared across batch
            pe = self._sinusoidal_pe(L_eff, self.cfg.d_model, device)[None, :, :].expand(B, L_eff, -1)
        te = self.type_embed(ttype)                         # (B, L_eff, D)
        X = X + pe + te

        # Build mask: True = pad (ignored)
        if depth_mask is None:
            depth_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        if yolo_mask is None:
            yolo_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        if self.cfg.use_cls:
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([cls_pad, depth_mask, yolo_mask], dim=1)  # (B, L_eff)
        else:
            key_padding_mask = torch.cat([depth_mask, yolo_mask], dim=1)           # (B, L_eff)

        # Encode
        enc = self.encoder(X, src_key_padding_mask=key_padding_mask)  # (B,L_eff,D)
        enc = self.norm_out(enc)

        # Slice outputs
        if self.cfg.use_cls:
            fused = enc[:, 0, :]                    # (B,D)
            depth_tokens = enc[:, 1:1+S, :]         # (B,S,D)
            yolo_tokens  = enc[:, 1+S:1+2*S, :]     # (B,S,D)
        else:
            depth_tokens = enc[:, :S, :]
            yolo_tokens  = enc[:, S:, :]
            if self.cfg.pooling == "mean":
                keep = torch.cat([~depth_mask, ~yolo_mask], dim=1).float().unsqueeze(-1)  # (B,L,1)
                fused = (enc * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1e-6)
            elif self.cfg.pooling == "max":
                neg_inf = torch.finfo(enc.dtype).min
                fused = enc.masked_fill(torch.cat([depth_mask, yolo_mask], dim=1).unsqueeze(-1), neg_inf).max(dim=1).values
            else:
                fused = enc.mean(dim=1)

        return fused, depth_tokens, yolo_tokens
