# yolo_backbone_embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple
from ultralytics import YOLO

class YoloBackConfig:
    """
    Configuration for extracting 3 embeddings from YOLO:
      - strategy='spaced': auto-pick 3 Conv-like layers (start/middle/end) evenly spaced
      - strategy='preset': use explicit module names you pass in `preset_names`
    """
    def __init__(
        self,
        model_id: str = "yolov8n.pt",           # path or repo tag for Ultralytics YOLO
        pool: str = "gap",                      # 'gap' or 'flatten'
        l2_normalize: bool = True,
        return_parts: bool = False,
        return_acts: bool = False,
        device: Optional[str] = None,          # "cuda" / "cpu" / None -> auto
        strategy: str = "spaced",              # 'spaced' or 'preset'
        k: int = 3,                            # used by 'spaced' (we’ll clamp to >=3)
        preset_names: Optional[Sequence[str]] = [
        "model.1.conv.conv",
        "model.9.cv2.conv",
        "model.22.cv3.0.1",
    ],  # used by 'preset'
        input_size: Optional[Tuple[int, int]] = (640, 640),  # must be multiple of 32 for YOLO
    ) -> None:
        self.model_id = model_id
        self.pool = pool
        self.l2_normalize = l2_normalize
        self.return_parts = return_parts
        self.return_acts = return_acts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.k = max(3, int(k))
        self.preset_names = list(preset_names) if preset_names else None
        self.input_size = input_size  # if you resize outside, you can set this to None

    def __str__(self) -> str:
        return (
            f"\n YOLO model: {self.model_id}"
            f"\n Strategy: {self.strategy} (k={self.k})"
            f"\n Pool: {self.pool}"
            f"\n L2 normalize: {self.l2_normalize}"
            f"\n return_parts: {self.return_parts}"
            f"\n return_acts: {self.return_acts}"
            f"\n Device: {self.device}"
            f"\n Input size: {self.input_size}"
            f"\n Preset names: {self.preset_names}"
        )


class YoloBackboneEmbedding(nn.Module):
    """
    Forward(X: (B,3,H,W) float in [0,1]) -> (emb, parts?, acts?)
      emb:  (B, D)  concatenation of 3 pooled vectors
      parts: Optional[Dict[name] -> (B, d_i)]
      acts:  Optional[Dict[name] -> (B, C, H, W)]
    """
    def __init__(self, config: YoloBackConfig):
        super().__init__()
        self.config = config
        self._chosen_names: Optional[List[str]] = None  # cache picked module names

        # Load YOLO and move to device
        self.yolo = YOLO(self.config.model_id)
        self.core: nn.Module = self.yolo.model  # DetectionModel (nn.Module)
        self.core.to(self.config.device).eval()

        # Initialize chosen names
        if self.config.strategy == "preset":
            if not self.config.preset_names:
                # Example preset (you may override with your exact names)
                self._chosen_names = [
                    "model.1.conv.conv",   # early downsample conv (32 ch)
                    "model.9.cv2.conv",    # SPPF second conv
                    # A head conv before final 1x1; the exact name can vary by version:
                    # Explore with: print([n for n,_ in self.core.named_modules()])
                    "model.22.cv3.0.1",    # if wrapper Conv, will still return 4D; fallback below if missing
                ]
            else:
                self._chosen_names = list(self.config.preset_names)
        elif self.config.strategy == "spaced":
            self._chosen_names = self._pick_spaced_conv_layers(self.config.k)
        else:
            raise ValueError("strategy must be 'spaced' or 'preset'")

        # Validate names exist; if a name isn't found but the parent wrapper exists,
        # we’ll try to fall back to the wrapper name (so we still get a 4D activation).
        name2mod = dict(self.core.named_modules())
        fixed: List[str] = []
        for n in self._chosen_names:
            if n in name2mod:
                fixed.append(n)
            else:
                # Try dropping a trailing ".conv" to hit the wrapper Conv module
                if n.endswith(".conv") and n.rsplit(".conv", 1)[0] in name2mod:
                    fixed.append(n.rsplit(".conv", 1)[0])
                else:
                    # Keep it for now; forward() will raise with a helpful error if missing
                    fixed.append(n)
        self._chosen_names = fixed

    # --- selection helpers ---
    def _pick_spaced_conv_layers(self, k: int) -> List[str]:
        """Enumerate Conv2d / ConvTranspose2d and pick k evenly spaced names."""
        conv_like = []
        for name, m in self.core.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_like.append(name)

        if not conv_like:
            raise RuntimeError("No Conv-like modules found in YOLO model to tap.")

        if k <= 1:
            idxs = [0]
        else:
            step = (len(conv_like) - 1) / (k - 1)
            idxs = [int(round(i * step)) for i in range(k)]

        # De-dup and clamp
        idxs = sorted({max(0, min(i, len(conv_like) - 1)) for i in idxs})
        # Pad if de-dup shrank
        while len(idxs) < k and len(idxs) < len(conv_like):
            for j in range(len(conv_like) - 1, -1, -1):
                if j not in idxs:
                    idxs.append(j)
                    break
            idxs = sorted(set(idxs))
        return [conv_like[i] for i in idxs[:k]]

    # --- main API ---
    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        X: (B,3,H,W) float in [0,1]. Prefer H,W multiples of 32 (e.g., 640x640 or 224x224).
        """
        return self._yolo_concat_embedding(X)

    @torch.inference_mode()
    def _yolo_concat_embedding(
        self, image_batch_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        assert image_batch_tensor.ndim == 4 and image_batch_tensor.size(1) == 3, \
            "image_batch_tensor must be (B,3,H,W) in [0,1]"
        B = image_batch_tensor.size(0)

        # Optional size check; YOLO prefers multiples of 32
        if self.config.input_size is not None:
            H, W = image_batch_tensor.shape[-2:]
            ih, iw = self.config.input_size
            if (H, W) != (ih, iw):
                # You can also resize outside this class if you prefer
                image_batch_tensor = torch.nn.functional.interpolate(
                    image_batch_tensor, size=(ih, iw), mode="bilinear", align_corners=False
                )

        device = self.config.device
        X = image_batch_tensor.to(device, non_blocking=True)

        # Hook only the chosen modules
        name2mod = dict(self.core.named_modules())
        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(name: str):
            def _hook(_m, _inp, out):
                t = out[0] if isinstance(out, (list, tuple)) else out
                if isinstance(t, torch.Tensor) and t.ndim == 4:  # (B,C,H,W)
                    captured[name] = t.detach()
            return _hook

        handles = []
        try:
            for n in self._chosen_names:
                if n not in name2mod:
                    # Help the user debug quickly:
                    sample = list(name2mod.keys())[:20]
                    raise KeyError(f"Chosen module '{n}' not found. Example names: {sample} ...")
                handles.append(name2mod[n].register_forward_hook(hook_fn(n)))

            _ = self.core(X)  # single forward through YOLO DetectionModel
        finally:
            for h in handles:
                h.remove()

        if not captured:
            raise RuntimeError("Chosen modules produced no 4D activations—verify names and input size (multiple of 32).")

        # Pool & concat
        pooled_list: List[torch.Tensor] = []
        out_parts: Optional[Dict[str, torch.Tensor]] = {} if self.config.return_parts else None

        for n in self._chosen_names:
            t = captured[n]  # (B,C,H,W)
            if self.config.pool == "gap":
                v = F.adaptive_avg_pool2d(t, 1).squeeze(-1).squeeze(-1)  # (B,C)
            elif self.config.pool == "flatten":
                v = t.flatten(1)  # (B, C*H*W)
            elif self.config.pool=='all':
              v = torch.cat([
    F.adaptive_avg_pool2d(t,1).squeeze(-1).squeeze(-1),
    F.adaptive_max_pool2d(t,1).squeeze(-1).squeeze(-1),
], dim=1)

            else:
                raise ValueError("`pool` must be 'gap' or 'flatten'.")

            v_cpu = v.float().cpu()
            pooled_list.append(v_cpu)
            if out_parts is not None:
                out_parts[n] = v_cpu

        emb = torch.cat(pooled_list, dim=1) if pooled_list else torch.empty((B, 0))
        if self.config.l2_normalize and emb.numel() and emb.shape[1] > 0:
            emb = F.normalize(emb, p=2, dim=1, eps=1e-12)

        out_acts = {n: captured[n].cpu() for n in self._chosen_names} if self.config.return_acts else None
        return emb, out_parts, out_acts
