import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch, embed_dim=512, block_num=4, num_heads=8, mlp_ratio=4.0, dropout=0.1, max_tokens=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_in = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(block_num)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_ch, kernel_size=1)

    def forward(self, x, cond=None):
        B, C, h, w = x.shape
        tokens = self.proj_in(x).flatten(2).transpose(1,2)
        n_tok = tokens.shape[1]
        if n_tok > self.pos_embed.shape[1]:
            raise ValueError(
                f"Bottleneck has {n_tok} tokens but pos_embed only supports "
                f"{self.pos_embed.shape[1]}. Increase max_tokens or downsample more."
            )
        tokens = tokens + self.pos_embed[:, :n_tok, :]
        if cond is not None:
            tokens = tokens + cond.unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        spatial = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        return self.proj_out(spatial), tokens