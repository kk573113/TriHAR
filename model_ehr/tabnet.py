import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from heads import HeadBlock, BaseModel

# =========================================================================
# Helper Layers (Sparsemax, GBN, GLU, Transformers) - 그대로 유지
# =========================================================================

class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_values = torch.arange(start=1, end=number_of_logits + 1, device=input.device, dtype=input.dtype).view(1, -1)
        range_values = range_values.expand_as(zs)

        bound = 1 + range_values * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumulative_sum_zs
        k = torch.max(is_gt * range_values, dim=dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        self.output = torch.max(torch.zeros_like(input), input - taus)
        output = self.output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        super(GLU, self).__init__()
        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.bn = GBN(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x[:, :self.output_dim] * torch.sigmoid(x[:, self.output_dim:])
        return out

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size=128, momentum=0.02):
        super(FeatureTransformer, self).__init__()
        self.shared = nn.ModuleList()
        if shared_layers:
            self.shared = shared_layers
        else:
            self.shared = nn.ModuleList()
            for _ in range(2):
                self.shared.append(GLU(input_dim, output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum))

        self.independent = nn.ModuleList()
        for _ in range(n_glu_independent):
            self.independent.append(GLU(output_dim, output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum))
        
        self.scale = torch.sqrt(torch.tensor([0.5], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        
        for glu in self.independent:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.selector = Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x

# =========================================================================
# [수정 2] Main TabNet Class (Multi-Stage Output Adapted)
# =========================================================================

class TabNet(BaseModel): # [수정] BaseModel 상속
    def __init__(
        self,
        input_dim,
        num_classes=2,  # [수정] Output Dim -> num_classes
        n_d=8,
        n_a=8,
        n_steps=3,      # [중요] 3-Stage 출력을 위해 n_steps=3 권장
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Embedding Layers
        if self.cat_idxs:
            self.embeddings = nn.ModuleList([
                nn.Embedding(n_classes, cat_emb_dim) for n_classes in self.cat_dims
            ])
            self.post_embed_dim = self.input_dim + len(self.cat_idxs) * (self.cat_emb_dim - 1)
        else:
            self.post_embed_dim = self.input_dim

        self.initial_bn = nn.BatchNorm1d(self.post_embed_dim, momentum=0.01)

        # TabNet Core Layers
        self.shared_layers = nn.ModuleList()
        for i in range(n_shared):
            if i == 0:
                self.shared_layers.append(GLU(self.post_embed_dim, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))
            else:
                self.shared_layers.append(GLU(n_d + n_a, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatureTransformer(
                self.post_embed_dim, n_d + n_a, self.shared_layers, n_independent,
                virtual_batch_size=virtual_batch_size, momentum=momentum
            )
            self.feat_transformers.append(transformer)
            attentive = AttentiveTransformer(
                n_a, self.post_embed_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
            )
            self.att_transformers.append(attentive)

        # [수정] 3-Stage Heads
        # TabNet의 Feature Dimension은 n_d임
        self.h1 = HeadBlock(n_d, 32, num_classes)
        self.h2 = HeadBlock(n_d, 32, num_classes)
        self.h3 = HeadBlock(n_d, 32, num_classes)
        
        # Layer Groups for freezing (Simplified)
        self.layers_groups = [self.shared_layers, self.feat_transformers, self.att_transformers]

    def forward(self, x):
        # 1. Embedding Handling
        if self.cat_idxs:
            all_cols = range(x.shape[1])
            cont_idxs = [i for i in all_cols if i not in self.cat_idxs]
            
            x_cont = x[:, cont_idxs]
            x_cat_raw = x[:, self.cat_idxs].long()
            
            embeddings = [emb(x_cat_raw[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(embeddings, dim=1)
            x = torch.cat([x_cont, x_cat_emb], dim=1)

        # 2. TabNet Processing
        x = self.initial_bn(x)

        priors = torch.ones(x.shape).to(x.device)
        M_loss = 0
        att = self.feat_transformers[0](x)
        
        # Accumulation buffer
        res = torch.zeros(x.shape[0], self.n_d).to(x.device)
        
        # Outputs containers
        z1, z2, z3 = None, None, None
        f_seq1, f_pool1 = None, None
        f_seq2, f_pool2 = None, None
        f_seq3, f_pool3 = None, None

        # TabNet Steps Loop
        for step in range(self.n_steps):
            M = self.att_transformers[step](priors, att[:, self.n_d:])
            
            # Update priors
            priors = priors * (self.gamma - M)
            
            # Masked Features
            masked_x = M * x
            att = self.feat_transformers[step](masked_x)
            
            # Accumulate Decision (Residual Addition)
            current_decision = torch.relu(att[:, :self.n_d])
            res = torch.add(res, current_decision)
            
            # [수정] 각 Step의 누적 결과(res)를 Head에 통과시켜 출력
            # TabNet은 n_steps=3일 때 3개의 출력을 생성
            
            if step == 0:
                f_seq1, f_pool1, z1 = self.h1(res)
            elif step == 1:
                f_seq2, f_pool2, z2 = self.h2(res)
            elif step == 2:
                f_seq3, f_pool3, z3 = self.h3(res)

        # Return: [Logits..., Features...]
        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]