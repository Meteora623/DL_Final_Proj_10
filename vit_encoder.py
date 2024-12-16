# vit_encoder.py
import torch
from torch import nn
from einops import rearrange

class ViTEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=8, dim=256, depth=4, heads=4, mlp_ratio=4.0, channels=2):
        super().__init__()
        assert image_size % patch_size == 0, "Image must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        self.patch_to_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, dim))

        encoder_layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout=0.0, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, depth)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        p = self.patch_size
        # 分patch
        patches = rearrange(x, 'b c (h p) (w q) -> b (h w) (c p q)', p=p, q=p)
        # patches: [B, num_patches, patch_dim]
        x = self.patch_to_embed(patches)  # [B, num_patches, dim]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, dim]
        x = x + self.pos_embed[:, :x.size(1), :]

        x = self.transformer(x)  # [B, num_patches+1, dim]
        x = self.norm(x[:, 0])   # 取CLS token的表示

        # 加入L2归一化减少表示坍缩
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x
