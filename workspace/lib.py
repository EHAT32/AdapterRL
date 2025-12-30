from dataclasses import dataclass
import torch.nn as nn
import torch

class VisionProjector(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Linear(config.img_latent_dim, config.proj_dim, bias=False)
        
    def forward(self, x):
        return self.proj(x)

class TextProjector(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Linear(config.txt_latent_dim, config.proj_dim, bias=False)
        
    def forward(self, x):
        return self.proj(x)
    
if __name__ == '__main__':
    
    @dataclass
    class Config:
        img_latent_dim : int = 2048
        txt_latent_dim : int = 768
        proj_dim : int = 768
        
    cfg = Config()
    
    proj = VisionProjector(cfg)
    B = 4
    T = 128
    d = cfg.img_latent_dim
    img_inputs = torch.randn((B, T, d))
    img_projection = proj(img_inputs)
    print(img_projection.shape)
    txt_inputs = torch.randn((B, 1, cfg.txt_latent_dim))
    txt_proj = TextProjector(cfg)
    txt_projection = txt_proj(txt_inputs)
    print(txt_projection.shape)