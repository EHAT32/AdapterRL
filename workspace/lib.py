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


class ModalityFusor(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_projector = VisionProjector(config)
        self.text_projector = TextProjector(config)
        self.encoder = Encoder(config)
        
    def forward(self, txt_latents, img_latents):
        txt_proj = self.text_projector(txt_latents)
        img_proj = self.image_projector(img_latents)
        output = self.encoder(txt_proj, img_proj)
        return output

class Encoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.block_num)])

class Block(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.txt_norm = nn.LayerNorm(config.proj_dim)
        self.img_norm = nn.LayerNorm(config.proj_dim)
        self.attn = CrossAttention(config)
        self.out_norm = nn.LayerNorm(config.attn_out_dim)
        self.ffn = FFN(config)
        
    def forward(self, text_latents, img_latents):
        out = self.txt_norm(text_latents)
        imgs = self.img_norm(img_latents)
        out = out + self.attn(out, imgs)
        out = self.out_norm(out)
        return out + self.ffn(out)
        
        
class CrossAttention(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert config.proj_dim == config.head_dim * config.head_num
        self.head_num = config.head_num
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.proj_dim, config.proj_dim, bias=False)
        self.key = nn.Linear(config.proj_dim, config.proj_dim, bias=False)
        self.value = nn.Linear(config.proj_dim, config.proj_dim, bias=False)
        self.out = nn.Linear(config.proj_dim, config.proj_dim, bias=False)
        self.dropout = nn.Dropout(config.attn_dropout)
        
    def forward(self, text_latents, img_latents):
        pass
        
    def _split_heads(self, x):
        #B, T, d -> B, H, T, dh
        B, T, d = x.shape
        x = x.view(B, T, self.head_num, self.head_dim)
        return x.transpose(1,2)
    
    def _merge_heads(self, x):
        #B, H, T, dh -> B, T, d
        B, H, T, dh = x.shape
        x = x.transpose(1,2).contiguous().view(B, T, H * dh)
        return x
        
class FFN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)    
    
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