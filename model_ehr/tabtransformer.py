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
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
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
# Main TabTransformer Class (3-Stage Output Adapted)
# =========================================================================

class TabTransformer(BaseModel): # [수정] BaseModel 상속
    def __init__(
        self,
        *,
        categories,         
        num_continuous,     
        dim,                
        depth,              
        heads,              
        dim_head = 16,
        num_classes = 2,    # [수정] 3-Class Classification
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.
    ):
        super().__init__()
        
        self.categories = categories
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.depth = depth
        
        # 1. Embeddings
        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)
        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std=0.02)

        self.embeds = nn.ModuleList([
            nn.Embedding(num_classes, dim - shared_embed_dim) for num_classes in categories
        ])

        # 2. Continuous Norm
        if self.num_continuous > 0:
            self.norm = nn.LayerNorm(num_continuous)

        # 3. Transformer Layers (List로 관리하여 Stage 제어)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout)),
            ]))

        # 4. Multi-Stage Heads
        # Feature Dimension = (Categorical Embeddings Flattened) + Continuous Features
        input_size = (dim * self.num_categories) + num_continuous
        
        self.h1 = HeadBlock(input_size, 32, num_classes)
        self.h2 = HeadBlock(input_size, 32, num_classes)
        self.h3 = HeadBlock(input_size, 32, num_classes)
        
        # Layer Groups for freezing
        # Stage별로 Layer를 묶어서 관리 (간소화)
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

        # 1. Categorical Embedding
        if self.num_categories > 0:
            categ_embeds = []
            for i, embed_layer in enumerate(self.embeds):
                categ_embeds.append(embed_layer(x_categ[:, i]))
            x_trans = torch.stack(categ_embeds, dim=1)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b=x_trans.shape[0])
                x_trans = torch.cat((x_trans, shared_categ_embed), dim=-1)
        else:
            x_trans = None

        # 2. Continuous Norm
        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)
        else:
            normed_cont = torch.empty(x.shape[0], 0, device=x.device)

        # 3. 3-Stage Processing
        stage_step = self.depth // 3
        
        # Outputs
        z1, z2, z3 = None, None, None
        f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3 = None, None, None, None, None, None

        # Transformer Loop
        if x_trans is not None:
            for i, (attn, ff) in enumerate(self.layers):
                x_trans = attn(x_trans) + x_trans
                x_trans = ff(x_trans) + x_trans
                
                # Checkpoints for Heads
                if i == stage_step - 1 or i == stage_step * 2 - 1 or i == self.depth - 1:
                    # Flatten Categorical Features
                    flat_categ = rearrange(x_trans, 'b ... -> b (...)')
                    
                    # Concatenate with Continuous Features
                    # (Note: Continuous features are static context here, as per TabTransformer design)
                    combined_feat = torch.cat((flat_categ, normed_cont), dim=-1)
                    
                    if i == stage_step - 1:
                        f_seq1, f_pool1, z1 = self.h1(combined_feat)
                    elif i == stage_step * 2 - 1:
                        f_seq2, f_pool2, z2 = self.h2(combined_feat)
                    elif i == self.depth - 1:
                        f_seq3, f_pool3, z3 = self.h3(combined_feat)
        else:
            # 범주형 변수가 없을 경우 (Continuous Only) -> 바로 MLP 통과와 동일
            # 이 경우 모든 Stage가 동일한 입력을 받음
            combined_feat = normed_cont
            f_seq1, f_pool1, z1 = self.h1(combined_feat)
            f_seq2, f_pool2, z2 = self.h2(combined_feat)
            f_seq3, f_pool3, z3 = self.h3(combined_feat)

        # Return format consistent with other models
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]