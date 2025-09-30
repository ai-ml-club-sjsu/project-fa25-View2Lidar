from BackBones.YoloBackBone.YoloBone import *
from BackBones.YoloBackBone.YoloFusion import *
from BackBones.DinoBackBone.DinoFusion import * 
from BackBones.DinoBackBone.Dino import * 
from BackBones.CrossFusion.CrossModalityFususion import *

class ImageToLidarEncoderConfig():
  def __init__(self,
  dino_cfg:DepthBackConfig=None,
  dino_fusion_cfg: DepthModelFusionConfig=None,
  yolo_detection_cfg: YoloBackConfig=None,
  yolo_fusion_cfg:  YoloModelFusionConfig=None,
  global_fusion_cfg:GlobalFusionConfig=None):

    self.dino_cfg= dino_cfg or DepthBackConfig(strategy="preset", return_acts=True, return_parts=True,
     preset_names=[
         "backbone.embeddings.patch_embeddings.projection",
         "neck.reassemble_stage.layers.1.projection",
         "head.conv2", ])
    self.dino_fusion_cfg=dino_fusion_cfg or DepthModelFusionConfig( pos_encoding='sinusoidal')

    self.yolo_detection_cfg= yolo_detection_cfg or  YoloBackConfig(
    model_id="yolov8n.pt",
    strategy="preset",
    pool="all",
    l2_normalize=True,
    return_parts=True,
    return_acts=False,
    input_size=(640, 640),
)
    self.yolo_fusion_cfg= yolo_fusion_cfg or  YoloModelFusionConfig(
    d_model=736, nhead=8, num_layers=6,
    in_dim=736, use_cls=True, pos_encoding="learned",
    pooling="cls", max_seq_len=9


)
    self.global_fusion_cfg= global_fusion_cfg or GlobalFusionConfig()

  def __str__(self):
    return (
        f"###Config Summary###"
        f"\n\nDepthBackConfig : {self.dino_cfg}"
        f"\n\nDepthModelFusionConfig : {self.dino_fusion_cfg}"
        f"\n\nYoloBackConfig : {self.yolo_detection_cfg}"
        f"\n\nYoloModelFusionConfig : {self.yolo_fusion_cfg}"
        f"\n\nGlobalFusionConfig: {self.global_fusion_cfg}"
        )


class ImageToLidarEncoder(nn.Module):
  def __init__(self,cfg:ImageToLidarEncoderConfig):
    super().__init__()
    self.cfg=cfg

    self.depth_embed= DepthBackBoneEmbedding(cfg.dino_cfg)
    self.yolo_embed= YoloBackboneEmbedding(cfg.yolo_detection_cfg)

    self.dino_fusion = DinoFusion(cfg.dino_fusion_cfg)
    self.yolo_fusion= YoloModelFusion(cfg.yolo_fusion_cfg)

    self.head= GlobalFusion(cfg.global_fusion_cfg)

  def reshape(self,X, shape:torch.Size, feature_dim:int)->torch.Tensor:
    return X.reshape(shape[0],shape[1], feature_dim)

  def forward(self,X:torch.Tensor)->torch.Tensor:
    """
    Input:
         X(torch.Tensor) is a tensor of images of shape B, 7, N, N where B is Batch, 7 is the image directions, and NxN is the images size

    Output:
          latent_projection(torch.Tensor) is a tensor of shape B, cfg.global_fusion_cfg.output_dim aka our latent project for the decoder to take as input and produce the lidar map.
    """
    assert len(X.shape) == 5
    print('X', X.shape)


    X_reshaped=X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4] )
    print('X_reshaped', X_reshaped.shape)


    depth_embed_latent=self.depth_embed(X_reshaped)
    depth_embed_latent= self.reshape(depth_embed_latent[0].to(self.cfg.dino_fusion_cfg.device),X.shape,depth_embed_latent[0].shape[1])

    print('depth_embed_latent: ', depth_embed_latent.shape)


    yolo_embed_latent=self.yolo_embed(X_reshaped)
    yolo_embed_latent=self.reshape(yolo_embed_latent[0].to(self.cfg.yolo_fusion_cfg.device), X.shape,yolo_embed_latent[0].shape[1] )
    print('yolo_embed_latent: ',yolo_embed_latent.shape)

    dino_glbl_attention_fused=self.dino_fusion(depth_embed_latent)[1]
    yolo_glbl_attention_fused=self.yolo_fusion(yolo_embed_latent)[1]

    latent_projection= self.head(dino_glbl_attention_fused.clone(),yolo_glbl_attention_fused.clone())[0]
    return latent_projection
