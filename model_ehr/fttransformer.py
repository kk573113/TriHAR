import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from heads import HeadBlock, BaseModel

# =========================================================================
# Helper Modules (Self-Attention, FeedForward etc.)
# =========================================================================

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v) # Fixed einsum indices
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# =========================================================================
# Numerical Embedder (Feature Tokenizer)
# =========================================================================

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# =========================================================================
# Main FTTransformer Class (3-Stage Output Adapted)
# =========================================================================

class FTTransformer(BaseModel): # [수정] BaseModel 상속
    def __init__(
        self,
        *,
        categories,         
        num_continuous,     
        dim,                
        depth,              
        heads,              
        dim_head = 16,
        num_classes = 2,    
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.depth = depth

        # 1. CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 2. Embeddings
        if self.num_categories > 0:
            self.categ_embeds = nn.ModuleList([
                nn.Embedding(num_classes, dim) for num_classes in categories
            ])

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, num_continuous)

        # 3. Transformer Layers (List로 관리하여 Stage 제어)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout)),
            ]))

        # 4. Multi-Stage Heads
        # FT-Transformer는 [CLS] 토큰(dim 차원)을 사용하여 분류함
        self.h1 = HeadBlock(dim, 32, num_classes)
        self.h2 = HeadBlock(dim, 32, num_classes)
        self.h3 = HeadBlock(dim, 32, num_classes)
        
        # Layer Groups for Freezing
        stage_step = depth // 3
        self.layers_groups = [
            self.layers[:stage_step],
            self.layers[stage_step:stage_step*2],
            self.layers[stage_step*2:]
        ]

    def forward(self, x):
        # Input Split
        x_cont = x[:, :self.num_continuous]
        x_categ = x[:, self.num_continuous:].long()

        tokens = []

        # 1. Continuous Features -> Tokens
        if self.num_continuous > 0:
            x_cont_emb = self.numerical_embedder(x_cont)
            tokens.append(x_cont_emb)

        # 2. Categorical Features -> Tokens
        if self.num_categories > 0:
            categ_tokens = []
            for i, embed_layer in enumerate(self.categ_embeds):
                categ_tokens.append(embed_layer(x_categ[:, i]))
            x_categ_emb = torch.stack(categ_tokens, dim=1)
            tokens.append(x_categ_emb)

        # 3. Concatenate Features & Append CLS
        x = torch.cat(tokens, dim=1)
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 4. 3-Stage Processing Loop
        stage_step = self.depth // 3
        
        z1, z2, z3 = None, None, None
        f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3 = None, None, None, None, None, None

        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            
            # Checkpoints for Heads
            if i == stage_step - 1 or i == stage_step * 2 - 1 or i == self.depth - 1:
                # Extract [CLS] token for classification
                cls_output = x[:, 0] # (Batch, Dim)
                
                if i == stage_step - 1:
                    f_seq1, f_pool1, z1 = self.h1(cls_output)
                elif i == stage_step * 2 - 1:
                    f_seq2, f_pool2, z2 = self.h2(cls_output)
                elif i == self.depth - 1:
                    f_seq3, f_pool3, z3 = self.h3(cls_output)

        # Return format consistent with other models
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]