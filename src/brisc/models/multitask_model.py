import torch
import torch.nn as nn

from blocks import CNNBlock, CNNDecoder, CNNEncoder
from transformer import TransformerBottleneck

class MultiTaskModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, num_seg_ch=1, base_ch=64, embed_dim=512, block_num=4, num_heads=8, dropout=0.1, max_tokens=1024, num_directions=4):
        super().__init__()

        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8
        self.encoder1 = CNNEncoder(in_channels, c1)
        self.encoder2 = CNNEncoder(c1, c2)
        self.encoder3 = CNNEncoder(c2, c3)
        self.encoder4 = CNNEncoder(c3, c4)

        self.bottleneck_conv = CNNBlock(c4, c4)    # c4 or embed_dim

        self.bottleneck = TransformerBottleneck(in_ch=c4, embed_dim=embed_dim, block_num=block_num, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout, max_tokens=max_tokens)

        self.bottleneck_end = CNNBlock(c4, c4)

        self.direction_embed = nn.Embedding(num_directions, embed_dim)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self.decoder4 = CNNDecoder(c4, c4, c3)
        self.decoder3 = CNNDecoder(c3, c3, c2)
        self.decoder2 = CNNDecoder(c2, c2, c1)
        self.decoder1 = CNNDecoder(c1, c1, c1)
        # # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
        #     CNNBlock(c1, c1),
        # )

        self.seg_head = nn.Conv2d(c1, num_seg_ch, kernel_size=1)

    def forward(self, x, direction=None):
        s1, x = self.encoder1(x)
        s2, x = self.encoder2(x)
        s3, x = self.encoder3(x)
        s4, x = self.encoder4(x)

        x = self.bottleneck_conv(x)

        dir_emb = self.direction_embed(direction) if direction is not None else None
        bottleneck_map, tokens = self.bottleneck(x, cond=dir_emb)
        bottleneck_end = self.bottleneck_end(bottleneck_map)

        cls_feat = tokens.mean(dim=1)
        cls_logits = self.cls_head(cls_feat)

        d = self.decoder4(bottleneck_end, s4)
        d = self.decoder3(d, s3)
        d = self.decoder2(d, s2)
        d = self.decoder1(d, s1)

        seg_logits = self.seg_head(d)

        return cls_logits, seg_logits
