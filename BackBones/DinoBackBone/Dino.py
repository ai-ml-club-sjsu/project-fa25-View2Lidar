import re, time
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthBackConfig:
    def __init__(
        self,
        model_id: Optional[str] = None,
        pool: str = "gap",                  # 'gap' or 'flatten'
        l2_normalize: bool = True,
        return_parts: bool = False,
        return_acts: bool = False,
        strategy: str = "spaced",          # 'spaced' or 'preset'
        k: int = 3,                        # used when strategy='spaced'
        preset_names: Optional[Sequence[str]] = None,  # used when strategy='preset'
    ) -> None:
        self.model_id = model_id or "depth-anything/Depth-Anything-V2-Small-hf"
        self.pool = pool
        self.l2_normalize = l2_normalize
        self.return_parts = return_parts
        self.return_acts = return_acts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.strategy = strategy
        self.k = k
        self.preset_names = list(preset_names) if preset_names else None

    def __str__(self):
        return (
            f"\n Model: {self.model_id}"
            f"\n Strategy: {self.strategy} (k={self.k})"
            f"\n Pool: {self.pool}"
            f"\n L2: {self.l2_normalize}"
            f"\n return_parts: {self.return_parts}"
            f"\n return_acts: {self.return_acts}"
            f"\n Device: {self.device}"
        )

class DepthBackBoneEmbedding(nn.Module):
    def __init__(self, config: DepthBackConfig):
        super().__init__()
        self.config = config
        self.MODEL_ID = config.model_id
        self.device = config.device
        self.model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID).to(config.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)

        # Choose the three taps (start / middle / end)
        if self.config.strategy == "preset":
            # Depth-Anything sensible defaults (rich & stable):
            #   - start: patch embed projection
            #   - middle: neck reassemble (stage 2) projection (192-d)
            #   - end: head conv2 before 1x1
            self._chosen_names = self.config.preset_names or [
                "backbone.embeddings.patch_embeddings.projection",
                "neck.reassemble_stage.layers.2.projection",
                "head.conv2",
            ]
        else:
            # strategy == "spaced": pick evenly spaced conv-like layers
            self._chosen_names = self._pick_spaced_conv_layers(k=max(3, self.config.k))

    def _pick_spaced_conv_layers(self, k: int = 3) -> List[str]:
        """Enumerate Conv2d / ConvTranspose2d in forward order and pick k evenly spaced names."""
        conv_like = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_like.append(name)

        if not conv_like:
            raise RuntimeError("No Conv-like modules found to tap.")

        # start, middle, end (even spacing if >3)
        if k <= 1:
            idxs = [0]
        else:
            step = (len(conv_like) - 1) / (k - 1)
            idxs = [int(round(i * step)) for i in range(k)]

        # ensure unique & sorted
        idxs = sorted(dict.fromkeys(max(0, min(i, len(conv_like)-1)) for i in idxs))
        # If dedup shrank the set, pad from the end
        while len(idxs) < k and len(idxs) < len(conv_like):
            for j in range(len(conv_like)-1, -1, -1):
                if j not in idxs:
                    idxs.append(j)
                    break
            idxs = sorted(dict.fromkeys(idxs))
        chosen = [conv_like[i] for i in idxs[:k]]
        return chosen

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        return self.depth_concat_embedding(X)

    @torch.inference_mode()
    def depth_concat_embedding(
        self, image_batch_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        assert image_batch_tensor.ndim == 4 and image_batch_tensor.size(1) == 3, "X must be (B,3,H,W) in [0,1]"
        B = image_batch_tensor.size(0)

        # ---- preprocess
        pil_list = [to_pil_image(x.detach().cpu().clamp(0, 1)) for x in image_batch_tensor]
        inputs = self.processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ---- hook only the chosen 3 names or more if you wish
        captured: Dict[str, torch.Tensor] = {}
        name2mod = dict(self.model.named_modules())

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
                    raise KeyError(f"Chosen module '{n}' not found. Available: e.g., {list(name2mod)[:5]} ...")
                handles.append(name2mod[n].register_forward_hook(hook_fn(n)))
            _ = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        if not captured:
            raise RuntimeError("Chosen modules produced no 4D activations. Check names or tap locations.")

        # ---- pool and concat
        pooled_list: List[torch.Tensor] = []
        out_parts = {} if self.config.return_parts else None
        for n in self._chosen_names:
            t = captured[n]  # (B,C,H,W)
            if self.config.pool == "gap":
                v = F.adaptive_avg_pool2d(t, 1).squeeze(-1).squeeze(-1)  # (B,C)
            elif self.config.pool == "flatten":
                v = t.flatten(1)  # (B, C*H*W)
            else:
                raise ValueError("pool must be 'gap' or 'flatten'")
            v = v.float().cpu()
            pooled_list.append(v)
            if out_parts is not None:
                out_parts[n] = v

        emb = torch.cat(pooled_list, dim=1) if pooled_list else torch.empty((B, 0))
        if self.config.l2_normalize and emb.numel() and emb.shape[1] > 0:
            emb = F.normalize(emb, p=2, dim=1, eps=1e-12)

        out_acts = {n: captured[n].cpu() for n in self._chosen_names} if self.config.return_acts else None
        return emb, out_parts, out_acts