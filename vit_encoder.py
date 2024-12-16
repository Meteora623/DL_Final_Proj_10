class ViTEncoder(nn.Module):
    def __init__(self, image_size=65, patch_size=5, dim=256, depth=4, heads=4, mlp_ratio=4):
        super().__init__()
        assert image_size % patch_size == 0, "Image must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 2 * patch_size * patch_size  # channels=2

        self.patch_size = patch_size
        self.dim = dim

        self.patch_to_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches+1, dim))

        encoder_layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W], H=W=65, patch_size=5可整除65
        B, C, H, W = x.shape
        p = self.patch_size
        patches = rearrange(x, 'b c (h p) (w q) -> b (h w) (c p q)', p=p, q=p)
        # patches: [B, num_patches, patch_dim]
        x = self.patch_to_embed(patches)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x
