from minestudio.models import VPTPolicy
from nanoGPT import GPT
import tiktoken
import torch.nn as nn
import torch

class NanoGPTFeatureExtractor(nn.Module):
    def __init__(self, device = "cuda", padding_id = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPT.from_pretrained("gpt2").to(device)
        self.device = device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.padding_id = padding_id

    @torch.no_grad()
    def forward(self, texts : list[str], pooling : str = "last", add_eos = False, max_tokens = 1024):
      """extract features from the batch of texts
         extract them according to pooling:
         pooling modes:
          None or 'none' : return all tokens as is
          'mean' : average of all tokens
          'last' : last token"""
      ids = [self.tokenizer.encode(t) for t in texts]
      if add_eos:
        eos = self.tokenizer.eot_token
        ids = [x + [eos] for x in ids]
      ids = [x[-max_tokens:] for x in ids]
      #add padding for same length in batch
      lengths = torch.tensor([len(id) for id in ids], device=self.device, dtype=torch.long)
      B = len(ids)
      T = int(lengths.max().item())
      padded_ids = torch.full((B, T), fill_value=self.padding_id, device=self.device, dtype=torch.long)
      for i, id in enumerate(ids):
        padded_ids[i, :len(id)] = torch.tensor(id, device=self.device, dtype=torch.long)
      _, _, latents = self.model(padded_ids, only_latents=True)
      return self._pooling(lengths, latents, pooling)

    def _pooling(self, lengths, latents, pooling):
      if pooling is None or pooling == 'none':
        return latents, lengths #B, T (padded), D
      B = latents.size(0)
      if pooling == 'last':
        return latents[torch.arange(B, device=latents.device), lengths-1], lengths #B, D
      T = latents.size(1)
      if pooling == 'mean':
        indices = torch.arange(T, device=latents.device).unsqueeze(0) # 1, T (padded)
        mask = indices < lengths.unsqueeze(1) #B, T(padded)
        mask = mask.float().unsqueeze(-1)
        sum = (latents * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return sum / denom, lengths #B, D
      return ValueError(f"pooling can be either none | last | mean but {pooling} was given")

class VPTFeatureExtractor(nn.Module):
    def __init__(self, model_name : str, device : str = "cuda", eval_mode : bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        policy = VPTPolicy.from_pretrained(model_name).to(device)
        self.device = device
        self.net = policy.net.to(device)
        if eval_mode:
            self.net.eval()
     
    @torch.no_grad()
    def init_state(self, batch_size: int):
        st = self.net.initial_state(batch_size)
        return [s.to(self.device) for s in st] if st is not None else None
            
    @torch.no_grad()
    def forward(self, obs : dict, state_in = None, context = None, pooling_mode = None):
        """Extract latent features from the VPT model"""
        if "image" not in obs:
            raise KeyError('Obs must contain "image" key')
        #x dim: (B, T, H, W, C)
        x = obs["image"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() != 5:
            raise ValueError(f"image must be in a 5D shape (B, T, H, W, C) but got {x.shape}")
        
        B, T = x.shape[:2]
        if context is None:
            first = torch.zeros((B, T), dtype=torch.bool, device=self.device)
            context = {"first" : first}
        else:
            context = dict(context)
            context["first"] = context["first"].to(self.device)
        if state_in is None:
            state_in = self.init_state(B)
        else:
            state_in = [s.to(self.device) for s in state_in]
        
        (latents, _), state_out = self.net({"image" : x}, state_in, context)
        latents = self._pooling(latents, pooling_mode)
        return latents, state_out
    
    def _pooling(self, latents, pooling):
      if pooling is None or pooling == 'none':
        return latents,  #B, T, D
      if pooling == 'last':
        return latents[:, -1] #B, D
      if pooling == 'mean':
        return latents.mean(dim=1) #B, D
      return ValueError(f"pooling can be either none | last | mean but {pooling} was given")

def check_gpt():
    texts = ["hello, world!", "i love minecraft so much"]
    extractor = NanoGPTFeatureExtractor(device="cpu")
    latents, lengths = extractor(texts, pooling = "last")
    print(latents.shape)

def check_vpt():
    extractor = VPTFeatureExtractor("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x", device="cuda")
    obs = {"image": torch.zeros(2, 16, 128, 128, 3, device="cuda", dtype=torch.uint8)}
    pi_h, state = extractor(obs, state_in=None, context=None)
    print(pi_h.shape)

  
if __name__ == '__main__':
    check_gpt()
    check_vpt()