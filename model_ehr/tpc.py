import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.modules.batchnorm import _BatchNorm
from heads import HeadBlock, BaseModel

# ==========================================
# Helper Classes (BatchNorm, EmptyModule)
# ==========================================
class MyBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)
        if not self.training: self.eval_momentum = 0
        exponential_average_factor = 0.0 if self.momentum is None else (self.momentum if self.training else self.eval_momentum)
        
        if self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: exponential_average_factor = self.momentum if self.training else self.eval_momentum

        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                            training=True, momentum=exponential_average_factor, eps=self.eps)

class MyBatchNorm1d(MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3: raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))

class EmptyModule(nn.Module):
    def forward(self, X): return X

# ==========================================
# Main TPC Model (Simplified & Multi-Head)
# ==========================================
class TempPointConv(BaseModel):
    def __init__(self, config, F=None, D=None, no_flat_features=None, num_classes=2):
        super(TempPointConv, self).__init__()
        
        # Configurations
        self.n_layers = config.n_layers
        self.model_type = config.model_type
        self.diagnosis_size = config.diagnosis_size
        self.main_dropout_rate = config.main_dropout_rate
        self.temp_dropout_rate = config.temp_dropout_rate
        self.kernel_size = config.kernel_size
        self.temp_kernels = config.temp_kernels
        self.point_sizes = config.point_sizes
        self.batchnorm = config.batchnorm
        self.F = F
        self.D = D
        self.no_flat_features = no_flat_features
        self.no_diag = config.no_diag
        self.no_mask = config.no_mask
        self.no_skip_connections = config.no_skip_connections
        self.momentum = 0.01 if self.batchnorm == 'low_momentum' else 0.1

        # Layers
        self.main_dropout = nn.Dropout(p=self.main_dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)
        self.relu = nn.ReLU()
        self.empty_module = EmptyModule()
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

        # BatchNorm Class Selection
        if self.batchnorm in ['mybatchnorm', 'pointonly', 'temponly', 'low_momentum']:
            self.batchnormclass = MyBatchNorm1d
        else:
            self.batchnormclass = nn.BatchNorm1d

        # Diagnosis Encoder (사용 안 할 경우 무시)
        self.diagnosis_encoder = nn.Linear(in_features=self.D, out_features=self.diagnosis_size)
        if self.batchnorm in ['mybatchnorm', 'pointonly', 'low_momentum', 'default']:
            self.bn_diagnosis_encoder = self.batchnormclass(num_features=self.diagnosis_size, momentum=self.momentum)
        else:
            self.bn_diagnosis_encoder = self.empty_module

        # Init Layers
        self.init_tpc()

        # [NEW] Multi-Stage Heads
        # TPC의 각 Stage 출력 차원은 point_size에 의해 결정됨
        head_dim = self.point_sizes[0]
        self.h1 = HeadBlock(head_dim, 32, num_classes)
        self.h2 = HeadBlock(head_dim, 32, num_classes)
        self.h3 = HeadBlock(head_dim, 32, num_classes)
        
        # Layer Groups for Freezing (Simplification)
        # self.layers_groups = ... (복잡하므로 생략하거나 필요시 추가)

    def init_tpc(self):
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1
            temp_k = self.temp_kernels[i]
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, temp_k=temp_k, point_size=point_size, dilation=dilation, stride=1)
        self.create_temp_pointwise_layers()

    def update_layer_info(self, layer=None, temp_k=None, point_size=None, dilation=None, stride=None):
        self.layers.append({})
        if point_size is not None: self.layers[layer]['point_size'] = point_size
        if temp_k is not None:
            padding = [(self.kernel_size - 1) * dilation, 0]
            self.layers[layer]['temp_kernels'] = temp_k
            self.layers[layer]['dilation'] = dilation
            self.layers[layer]['padding'] = padding
            self.layers[layer]['stride'] = stride

    def create_temp_pointwise_layers(self):
        self.layer_modules = nn.ModuleDict()
        self.Y = 0; self.Z = 0; self.Zt = 0

        for i in range(self.n_layers):
            temp_in_channels = (self.F + self.Zt) * (1 + self.Y) if i > 0 else 2 * self.F
            temp_out_channels = (self.F + self.Zt) * self.layers[i]['temp_kernels']
            linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z * self.F + self.no_flat_features
            linear_output_dim = self.layers[i]['point_size']
            
            if self.no_mask:
                if i == 0: temp_in_channels = self.F
                linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + self.F + self.no_flat_features

            temp = nn.Conv1d(in_channels=temp_in_channels, out_channels=temp_out_channels,
                             kernel_size=self.kernel_size, stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'], groups=self.F + self.Zt)
            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            if self.batchnorm in ['default', 'mybatchnorm', 'low_momentum']:
                bn_temp = self.batchnormclass(num_features=temp_out_channels, momentum=self.momentum)
                bn_point = self.batchnormclass(num_features=linear_output_dim, momentum=self.momentum)
            else:
                bn_temp = bn_point = self.empty_module

            self.layer_modules[str(i)] = nn.ModuleDict({'temp': temp, 'bn_temp': bn_temp, 'point': point, 'bn_point': bn_point})
            self.Y = self.layers[i]['temp_kernels']
            self.Z = linear_output_dim
            self.Zt += self.Z

    def temp_pointwise(self, B, T, X, repeat_flat, X_orig, temp, bn_temp, point, bn_point, temp_kernels, point_size, padding, prev_temp, prev_point, point_skip):
        Z = prev_point.shape[1] if prev_point is not None else 0
        X_padded = pad(X, padding, 'constant', 0)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))

        X_concat = cat(self.remove_none((prev_temp, prev_point, X_orig, repeat_flat)), dim=1)
        point_output = self.main_dropout(bn_point(point(X_concat)))

        point_skip = cat((point_skip, prev_point.view(B, T, Z).permute(0, 2, 1)), dim=1) if prev_point is not None else point_skip
        temp_skip = cat((point_skip.unsqueeze(2), X_temp.view(B, point_skip.shape[1], temp_kernels, T)), dim=2)

        X_point_rep = point_output.view(B, T, point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + temp_kernels), 1)
        X_combined = self.relu(cat((temp_skip, X_point_rep), dim=1))
        next_X = X_combined.view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, point_skip.shape[1] * temp_kernels)
        return temp_output, point_output, next_X, point_skip

    def forward(self, X, diagnoses=None, flat=None):
        # 1. Input Parsing (Flexible for Tabular)
        if X.dim() == 2:
            X = X.unsqueeze(-1)
        B, C, T = X.shape
        if C == 2 * self.F + 2 or C == self.F + 2:
             X_separated = torch.split(X[:, 1:-1, :], self.F, dim=1)
        else:
            if self.no_mask: X_separated = (X,)
            else: X_separated = torch.split(X, self.F, dim=1)

        B, _, T = X_separated[0].shape
        
        # X_orig
        if C == 2 * self.F + 2 or C == self.F + 2:
            if self.no_mask: X_orig = cat((X_separated[0], X[:, 0, :].unsqueeze(1), X[:, -1, :].unsqueeze(1)), dim=1).permute(0, 2, 1).contiguous().view(B * T, self.F + 2)
            else: X_orig = X.permute(0, 2, 1).contiguous().view(B * T, 2 * self.F + 2)
        else:
            X_orig = X.permute(0, 2, 1).contiguous().view(B * T, C)

        # Flat features
        if flat is None: flat = torch.zeros(B, self.no_flat_features, device=X.device)
        repeat_flat = flat.repeat_interleave(T, dim=0)

        # Initial States
        if self.no_mask: next_X = X_separated[0]
        else: next_X = torch.stack(X_separated, dim=2).reshape(B, 2 * self.F, T)
        point_skip = X_separated[0]
        temp_output = None
        point_output = None

        repeat_args = {'repeat_flat': repeat_flat, 'X_orig': X_orig, 'B': B, 'T': T}
        
        # 2. Layer Loop & Multi-Stage Output
        stage_step = self.n_layers // 3
        z1, z2, z3 = None, None, None
        f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3 = None, None, None, None, None, None

        for i in range(self.n_layers):
            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            
            temp_output, point_output, next_X, point_skip = self.temp_pointwise(
                X=next_X, point_skip=point_skip, prev_temp=temp_output, prev_point=point_output,
                temp_kernels=self.layers[i]['temp_kernels'], padding=self.layers[i]['padding'],
                point_size=self.layers[i]['point_size'], **kwargs
            )
            
            # Extract Intermediate Outputs
            # point_output: (B*T, point_size) -> (B, point_size, T)
            current_feat = point_output.view(B, T, -1).permute(0, 2, 1)

            if i == stage_step - 1:
                f_seq1, f_pool1, z1 = self.h1(current_feat)
            elif i == stage_step * 2 - 1:
                f_seq2, f_pool2, z2 = self.h2(current_feat)
            elif i == self.n_layers - 1:
                f_seq3, f_pool3, z3 = self.h3(current_feat)

        return [z1, z2, z3, f_seq1, f_pool1, f_seq2, f_pool2, f_seq3, f_pool3]