import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """
        input: Tensor of shape (B, ..., dim)
        return: Tensor of same shape with sparsemax applied along dim
        """
        original_size = input.size()

        # reshape input to 2D: (N, D)
        input = input.transpose(self.dim, -1)
        input = input.reshape(-1, input.size(-1))  # (N, D)

        # Step 1: sort input in descending order
        zs = torch.sort(input, dim=1, descending=True)[0]
        range = torch.arange(1, zs.size(1)+1, device=input.device).float().unsqueeze(0)

        # Step 2: compute k(z)
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim=1)
        is_gt = bound > cumulative_sum_zs
        k = is_gt.sum(dim=1).unsqueeze(1)  # (N, 1)

        # Step 3: compute tau
        zs_sparse = torch.gather(zs, 1, k - 1)
        tau = (cumulative_sum_zs.gather(1, k - 1) - 1) / k.float()

        # Step 4: compute sparsemax output
        output = torch.clamp(input - tau, min=0)

        # reshape back
        output = output.reshape(*original_size[:-1], output.size(-1))
        output = output.transpose(self.dim, -1)

        return output
    
class CrossAttentionBasedDistance(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, use_sparsemax=True):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

        self.use_sparsemax = use_sparsemax
        if use_sparsemax:
            self.attn_fn = Sparsemax(dim=1)
        else:
            self.attn_fn = nn.Softmax(dim=1)

    def forward(self, eeg_embed, motor_embed):
        """
        eeg_embed: (N, D) - acts as query
        motor_embed: (N, D) - acts as key/value
        """
        Q = self.q_proj(eeg_embed)     # (N, H)
        K = self.k_proj(motor_embed)   # (N, H)
        V = self.v_proj(motor_embed)   # (N, H)

        # Compute attention scores
        attn_scores = (Q * K).sum(dim=1, keepdim=True) / (K.size(1) ** 0.5)  # (N, 1)

        # Apply attention function
        attn_weights = self.attn_fn(attn_scores)  # (N, 1)

        # Weighted value
        attended = attn_weights * V  # (N, H)
        attended = self.out_proj(attended)  # (N, D)

        # Compute L2/MSE distance between attended and original EEG
        distance = F.mse_loss(attended, eeg_embed, reduction='none').mean(dim=1)  # (N,)
        return distance

class DeepConvFeatureExtractor(nn.Module):
    
    def __init__(self, 
                 kernel_width=20,
                 pool_width=3,
                 num_eeg_channels=59, 
                 time_steps=400, 
                 max_norm=2, 
                 dropout_p=0.5, 
                 nFiltLaterLayer=[25, 50, 100, 200]):
        super(DeepConvFeatureExtractor, self).__init__()
      
        self.nFiltLaterLayer = nFiltLaterLayer
        
        layers = []
        in_channels = 1  # Assuming input is grayscale (1 channel)
        
        # Create layers dynamically based on nFiltLaterLayer
        for i, out_channels in enumerate(nFiltLaterLayer):
            # Add padding layer for the kernel
            layers.append(nn.ZeroPad2d((kernel_width // 2 - 1, kernel_width // 2, 0, 0)))
            
            # Add the first convolutional layer
            if i == 0:
                layers.append(Conv2dWithConstraint(in_channels, out_channels, (1, kernel_width), max_norm=max_norm))
                layers.append(Conv2dWithConstraint(out_channels, out_channels, (num_eeg_channels, 1), bias=False, max_norm=max_norm))
            else:
                # Add subsequent convolutional layers
                layers.append(Conv2dWithConstraint(nFiltLaterLayer[i-1], out_channels, (1, kernel_width), bias=False, max_norm=max_norm))
            
            # Add BatchNorm, ELU activation, and MaxPool
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU())
            layers.append(nn.MaxPool2d((1, pool_width)))
            
            # Add dropout layer
            layers.append(nn.Dropout(p=dropout_p))
            
            # Update the number of input channels for the next layer
            in_channels = out_channels
        
        # Create a sequential model with the layers
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, 1, 59, 400)
        return self.cnn(x)  # Output shape: (B, 200, 1, T')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]

class EEGEncoder(nn.Module):
    def __init__(self, num_eeg_channels=59, time_steps=400, embed_dim=128,
                 kernel_width=20, pool_width=3, nFiltLaterLayer = [25, 50, 100, 200], dropout_p=0.5, max_norm=2,
                 transformer_layers=2, nhead=4):
        super().__init__()
        self.embed_dim = embed_dim

        self.cnn = DeepConvFeatureExtractor(
            num_eeg_channels=num_eeg_channels, kernel_width=kernel_width,
            pool_width=pool_width, nFiltLaterLayer= nFiltLaterLayer,
            time_steps=time_steps, max_norm=max_norm, dropout_p=dropout_p
        )
        self.output = Conv2dWithConstraint(nFiltLaterLayer[-1], embed_dim, (1, time_steps // (pool_width ** len(nFiltLaterLayer))), bias=False, max_norm=max_norm)

    def forward(self, x):  # (B, 1, 59, 400)
        x = self.cnn(x)  #(B, 200, 1, 400//81)
        x = self.output(x)  # (B, embed_dim, 1, 1)
        x = x.view(x.size(0), -1)
        return x

class PredDecoder(nn.Module):
    def __init__(self, embed_dim=128, output_dim=6, hidden_dims=[128, 64]):
        """
        A simple MLP decoder that predicts the final motor state (one frame) from the embedding.
        """
        super().__init__()
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))  # Final output: one frame of motor data
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):  # (B, embed_dim)
        return self.decoder(x)  # (B, output_dim)

class EEG_pred(nn.Module):
    def __init__(self, num_eeg_channels=59,
                 time_steps=400,
                 embed_dim=128,
                 output_dim=6,
                 kernel_width=20,
                 pool_width=3,
                 nFiltLaterLayer = [25, 50, 100, 200],
                 dropout_p=0.5,
                 n_domains=100,
                 max_norm=2,
                 hidden_dims=[128, 64]):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = EEGEncoder(
            num_eeg_channels=num_eeg_channels,
            time_steps=time_steps,
            embed_dim=embed_dim,
            kernel_width=kernel_width,
            pool_width=pool_width,
            nFiltLaterLayer = nFiltLaterLayer,
            dropout_p=dropout_p,
            max_norm=max_norm
        )
        self.output_layer = PredDecoder(
            embed_dim=embed_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
        
        self.loss = nn.MSELoss()

    def forward(self, batch):  # 输入 x: (B, 1, 59, 400)
        x = batch['EEG']  # (B, 1, 59, 400)
        y = batch['Y']  # (B, 6)
        x = self.encoder(x)  # (B, embed_dim)
        
        x = self.output_layer(x)  # (B, 6)
        
        mse_loss = self.loss(x, y)
        if self.training:
            rmse = torch.sqrt(mse_loss + 1e-8)  # 防止 sqrt(0) 的数值不稳定
        
            logs = {
                'loss_prediction': mse_loss.item(),
                'loss_rmse': rmse.item()
            }
            return mse_loss, logs
        else:
            # During inference, return the predicted motor output
            return mse_loss.item(), x  # (B, 6)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

class EEG_pred_multihead(nn.Module):
    def __init__(self, num_eeg_channels=59,
                 time_steps=400,
                 embed_dim=128,
                 output_dim=6,
                 kernel_width=20,
                 pool_width=3,
                 nFiltLaterLayer = [25, 50, 100, 200],
                 dropout_p=0.5,
                 max_norm=2,
                 n_domains=100, 
                 hidden_dims=[128, 64],
                 target_domains = [98, 99]):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_domains = n_domains
        self.target_domains = target_domains
        self.output_dim = output_dim
        self.encoder = EEGEncoder(
            num_eeg_channels=num_eeg_channels,
            time_steps=time_steps,
            embed_dim=embed_dim,
            kernel_width=kernel_width,
            pool_width=pool_width,
            nFiltLaterLayer = nFiltLaterLayer,
            dropout_p=dropout_p,
            max_norm=max_norm
        )
        self.output_layer = PredDecoder(
            embed_dim=embed_dim,
            output_dim=output_dim * self.n_domains,
            hidden_dims=hidden_dims
        )

        self.domain_linear = nn.Linear(embed_dim, self.n_domains)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, batch, vis_embd=False, return_entropy=False):  # 输入 x: (B, 1, 59, 400)
        x = batch['EEG']
        x.requires_grad_()
        x.retain_grad() 
        batch['EEG'] = x  # 保证外部可以读取 batch['EEG'].grad

        Y = batch['Y']
        domain = batch['domain']
        x = self.encoder(x)

        if vis_embd:
            return x

        domain_weights = self.domain_linear(x)
        domain_weights[torch.arange(x.size(0)), domain] = -float('inf')
        domain_weights[:, self.target_domains] = -float('inf')
        domain_weights = torch.softmax(domain_weights, dim=-1)
        domain_weights = domain_weights.unsqueeze(-1)

        x = self.output_layer(x)
        logits_interdomain = x.view(x.size(0), self.n_domains, self.output_dim)
        logits_cross = torch.sum(logits_interdomain * domain_weights, dim=1)
        loss_cross = self.loss_fn(logits_cross, Y)

        if return_entropy:
            # squeeze 最后一维，如果存在
            if domain_weights.dim() == 3:
                domain_weights = domain_weights.squeeze(-1)  # shape: [B, N]

            # 避免 log(0)
            eps = 1e-10
            entropy = -torch.sum(domain_weights * torch.log(domain_weights + eps), dim=1)  # shape: [B]

            # 计算最大熵（均匀分布下的熵）用于归一化
            num_domains = domain_weights.shape[1]
            max_entropy = torch.log(torch.tensor(num_domains, dtype=domain_weights.dtype, device=domain_weights.device))

            normalized_entropy = entropy / max_entropy  # shape: [B], 值在 [0, 1] 之间

            return logits_cross.to(torch.float32), normalized_entropy
        if not self.training:
            return loss_cross.item(), logits_cross.to(torch.float32)
        # 获取当前 domain head 输出 (Intra-domain prediction)
        logits_intra = logits_interdomain[torch.arange(x.size(0)), domain]  # (B, n_classes)
        # 计算监督损失 (L_pred)
        loss_intra = self.loss_fn(logits_intra, Y)
        loss_rmse = torch.sqrt(loss_intra + 1e-8)  # 防止 sqrt(0) 的数值不稳定
        loss_total = loss_intra + 0.5 * loss_cross
        if self.training:
            
            
            logs = {
                'loss_mse': loss_intra.item(),
                'loss_rmse': loss_rmse.item(),
                'loss_cross': loss_cross.item(),
                'loss_total': loss_total.item()
            }
            return loss_total, logs
        
        
        else:
            # During inference, return the predicted motor output
            return loss_cross.item(), logits_cross.to(torch.float32)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

class MotorEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128, time_steps=400, hidden_dims=[64, 128],
                 transformer_layers=2, nhead=4):
        super().__init__()
        layers = []
        in_channels = input_dim
        for h in hidden_dims:
            layers.append(nn.Conv1d(in_channels, h, kernel_size=39, stride=4, padding=19))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_channels = h
        self.cnn = nn.Sequential(*layers)
        reduced_len = time_steps // (4 ** len(hidden_dims))
        self.project = nn.Linear(in_channels, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # (B, 400, 6)
        x = x.permute(0, 2, 1)  # (B, 6, 400)
        x = self.cnn(x)  # (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.project(x)  # (B, L, D)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B, L, D)
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return x

class MotorDecoder(nn.Module):
    def __init__(self, embed_dim=128, output_dim=6, time_steps=400, hidden_dims=[128, 64], 
                 kernel_size=39, stride=4, padding=19):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.time_steps = time_steps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 反推 encoder 输出的 reduced_len
        reduced_len = time_steps
        for _ in hidden_dims:
            reduced_len = self._conv1d_output_length(
                reduced_len, kernel_size, stride, padding
            )
        self.reduced_len = reduced_len

        # 初始化展开维度
        self.init_channels = hidden_dims[0]
        self.expand = nn.Linear(embed_dim, self.init_channels * self.reduced_len)

        # 构造 decoder 网络
        layers = []
        in_channels = self.init_channels
        current_len = self.reduced_len

        for h in hidden_dims[1:]:
            output_padding = self._get_output_padding(current_len, stride, padding, kernel_size)
            layers.append(nn.ConvTranspose1d(in_channels, h, kernel_size, stride, padding, output_padding))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            in_channels = h
            current_len = current_len * stride  # 理想恢复长度

        # 最后一层输出 motor 信号
        output_padding = self._get_output_padding(current_len, stride, padding, kernel_size, final_layer=True)
        layers.append(nn.ConvTranspose1d(in_channels, output_dim, kernel_size, stride, padding, output_padding))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):  # (B, embed_dim)
        x = self.expand(x)  # (B, C*L)
        x = x.view(x.size(0), self.init_channels, self.reduced_len)  # (B, C, L)
        x = self.decoder(x)  # (B, 6, time_steps)
        x = x.permute(0, 2, 1)  # (B, time_steps, 6)
        assert x.shape[1] == self.time_steps, f"Expected {self.time_steps}, got {x.shape[1]}"
        return x

    @staticmethod
    def _conv1d_output_length(L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    @staticmethod
    def _conv_transpose1d_output_length(L_in, kernel_size, stride, padding, output_padding=0, dilation=1):
        return (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    def _get_output_padding(self, L_in, stride, padding, kernel_size, final_layer=False):
        # 计算实际输出长度
        out_no_pad = self._conv_transpose1d_output_length(
            L_in, kernel_size, stride, padding, output_padding=0
        )
        desired = self.time_steps if final_layer else L_in * stride
        output_padding = desired - out_no_pad
        if output_padding < 0 or output_padding > stride:
            raise ValueError(f"Cannot match desired output length with current config (got {output_padding})")
        return output_padding
"""
class EEGMotorContrastiveModel(nn.Module):
    def __init__(self,
                 num_eeg_channels=59,
                 time_steps=400,
                 embed_dim=128,
                 motor_input_dim=6,
                 kernel_width=20,
                 pool_width=3,
                 nFiltLaterLayer = [25,50, 100, 200],
                 dropout_p=0.5,
                 max_norm=2,
                 target_domains = [i for i in range(80,99)],
                 n_domains=100,
                 contrastive_margin=1.0,
                 reconstruction_weight=0.5,
                 sparsemax=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_margin = contrastive_margin
        self.reconstruction_weight = reconstruction_weight
        self.target_domains = target_domains

        
        self.eeg_encoder = EEGEncoder(
            num_eeg_channels=num_eeg_channels,
            time_steps=time_steps,
            embed_dim=embed_dim,
            kernel_width=kernel_width,
            pool_width=pool_width,
            nFiltLaterLayer = nFiltLaterLayer,
            dropout_p=dropout_p,
            max_norm=max_norm
        )

        self.motor_encoder = MotorEncoder(
            time_steps=time_steps,
            input_dim=motor_input_dim,
            embed_dim=embed_dim
        )

        self.motor_decoder = MotorDecoder(
            embed_dim=embed_dim,
            time_steps=time_steps,
            output_dim=motor_input_dim
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))  # 可学习的 temperature
        self.logit_bias = nn.Parameter(torch.tensor(0.0))    # 可学习的 bias
        self.cross_attention_distance = CrossAttentionBasedDistance(embed_dim=embed_dim, use_sparsemax=sparsemax)


        self.mse_loss = nn.MSELoss()

    def forward(self, batch, ratio=20, contrastive_weight=1, reconstruction_weight = 1, prediction_weight=0, alignment_weight=100, loss_cross_weight=2, vis_embd = False):  # 输入 x: (B, 1, 59, 400)
        eeg = batch['EEG']  # (B, 1, 59, 400)
        motor = batch['Y']  # (B, 400, 6)
        domain = batch['domain']  # (B, 1)
        

        eeg_embed = self.eeg_encoder(eeg)     # (B, embed_dim)
        if vis_embd:
            return eeg_embed
        # Reconstruction loss
        
        
        if self.training:
            

            # === 创建掩码：哪些样本是 source domain ===
            # domain: (B, 1), target_domains: List or Tensor
            is_target = torch.isin(domain, torch.tensor(self.target_domains, device=domain.device))  # (B,)
            is_source = ~is_target  # (B,)
            # Contrastive loss (L2 distance)
            motor_embed = self.motor_encoder(motor)  # (B, embed_dim)
            # 只计算 source domain 的 reconstruction loss
            
            # if is_source.any() and is_target.any() and alignment_weight != 0:
            #     alignment_loss = coral_loss(eeg_embed[is_source], eeg_embed[is_target]) * eeg.shape[1]  # (B, embed_dim)
            # else:
            #     alignment_loss = torch.tensor(0.0, device=eeg.device)
            # if is_target.any():
            #     siglip_loss_target = sigmoid_contrastive_loss(eeg_embed[is_target], motor_embed[is_target], self.temperature, self.logit_bias, ratio=ratio, save_vis=False)
            #     # siglip_loss_target = sigmoid_contrastive_loss_with_distance(eeg_embed[is_target], 
            #     #                                                             motor_embed[is_target], 
            #     #                                                             cross_attention_distance_module=self.cross_attention_distance,
            #     #                                                             temperature=self.temperature,
            #     #                                                             logit_bias=self.logit_bias,
            #     #                                                             ratio=ratio,
            #     #                                                             save_vis=False,
            #     #                                                             bidirectional=False,
            #     #                                                             monitor=False
            #     #                                                             )
            # else:
            #     siglip_loss_target = torch.tensor(0.0, device=motor.device)
            alignment_loss = torch.tensor(0.0, device=eeg.device)
            siglip_loss_target = torch.tensor(0.0, device=motor.device)

            if is_source.any():
                if contrastive_weight != 0:
                    contrastive_loss = relative_contrastive_loss(
                        eeg_embed[is_source], 
                        motor_embed[is_source], 
                        self.cross_attention_distance,
                        # ratio = ratio,
                        temperature=self.temperature
                    )
                    contrastive_loss = contrastive_loss / 100
                
                    # contrastive_loss = sigmoid_contrastive_loss(eeg_embed[is_source], motor_embed[is_source], self.temperature, self.logit_bias, ratio=ratio, save_vis=False)

                    # contrastive_loss = sigmoid_contrastive_loss_with_distance(eeg_embed[is_source],
                    #                                                         motor_embed[is_source],
                    #                                                         cross_attention_distance_module=self.cross_attention_distance,
                    #                                                         temperature=self.temperature,
                    #                                                         logit_bias=self.logit_bias,
                    #                                                         ratio=ratio,
                    #                                                         save_vis=False,
                    #                                                         bidirectional=False,
                    #                                                         monitor=False
                    #                                                     )
                else:
                    contrastive_loss = torch.tensor(0.0, device=motor.device)

               
                motor_pred = self.motor_decoder(eeg_embed)  # (B, 400, 6)
                if reconstruction_weight != 0:
                    # 计算重建损失 (L_rec)
                    reconstruction_loss = self.mse_loss(motor_pred[is_source], motor[is_source])  # (B, 400, 6)
                    # reconstruction_loss = reconstruction_loss + last_frame_loss
                else:
                    reconstruction_loss = torch.tensor(0.0, device=motor.device)
                if prediction_weight != 0:
                    last_frame_loss = self.mse_loss(motor_pred[is_source, -1, :], motor[is_source, -1, :])  # (B, 6)
                else:
                    last_frame_loss = torch.tensor(0.0, device=motor.device)

            
            total_loss = contrastive_weight * contrastive_loss + \
                        reconstruction_weight * reconstruction_loss + \
                        alignment_weight * alignment_loss + \
                        prediction_weight + last_frame_loss

            logs = {
                'loss_total': total_loss.item(),
                'loss_contrastive': contrastive_loss.item(),
                'siglip_loss_target': siglip_loss_target.item(),
                'loss_reconstruction': reconstruction_loss.item(),
                'last_frame_loss': last_frame_loss.item(),
                'loss_coral': alignment_loss.item()
            }

            return total_loss, logs
        else:
            motor_pred = self.motor_decoder(eeg_embed)  # (B, 400, 6)
            # During inference, return the predicted motor output
            return 0, motor_pred[:, -1,:].to(torch.float32)  # (B, 6)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
"""

class EEGMotorContrastiveModel(nn.Module):
    def __init__(self,
                 num_eeg_channels=59,
                 time_steps=400,
                 embed_dim=128,
                 motor_input_dim=6,
                 kernel_width=20,
                 pool_width=3,
                 nFiltLaterLayer = [25,50, 100, 200],
                 dropout_p=0.5,
                 max_norm=2,
                 target_domains = [i for i in range(80,99)],
                 n_domains=100,
                 contrastive_margin=1.0,
                 reconstruction_weight=0.5,
                 sparsemax=True,
                 distance_type="cross_attention"   # 新增参数: "cross_attention" or "cosine"
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_margin = contrastive_margin
        self.reconstruction_weight = reconstruction_weight
        self.target_domains = target_domains
        self.distance_type = distance_type   # 保存距离函数选择

        self.eeg_encoder = EEGEncoder(
            num_eeg_channels=num_eeg_channels,
            time_steps=time_steps,
            embed_dim=embed_dim,
            kernel_width=kernel_width,
            pool_width=pool_width,
            nFiltLaterLayer = nFiltLaterLayer,
            dropout_p=dropout_p,
            max_norm=max_norm
        )

        self.motor_encoder = MotorEncoder(
            time_steps=time_steps,
            input_dim=motor_input_dim,
            embed_dim=embed_dim
        )

        self.motor_decoder = MotorDecoder(
            embed_dim=embed_dim,
            time_steps=time_steps,
            output_dim=motor_input_dim
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))  # 可学习的 temperature
        self.logit_bias = nn.Parameter(torch.tensor(0.0))    # 可学习的 bias

        # 默认用 cross-attention distance
        self.cross_attention_distance = CrossAttentionBasedDistance(embed_dim=embed_dim, use_sparsemax=sparsemax)
        self.mse_loss = nn.MSELoss()

    def _compute_distance(self, eeg_embed, motor_embed):
        """
        内部统一接口，根据 distance_type 选择不同的距离函数
        """
        if self.distance_type == "cross_attention":
            return self.cross_attention_distance(eeg_embed, motor_embed)
        elif self.distance_type == "cosine":
            # 余弦相似度 -> 转换为"距离"
            sim = F.cosine_similarity(eeg_embed, motor_embed, dim=-1)  # (N,)
            return 1 - sim  # 越小越相似
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")

    def forward(self, batch, ratio=20, contrastive_weight=1, reconstruction_weight = 1, prediction_weight=0, alignment_weight=100, loss_cross_weight=2, vis_embd = False):  
        eeg = batch['EEG']  # (B, 1, 59, 400)
        motor = batch['Y']  # (B, 400, 6)
        domain = batch['domain']  # (B, 1)
        
        eeg_embed = self.eeg_encoder(eeg)     # (B, embed_dim)
        if vis_embd:
            return eeg_embed

        if self.training:
            is_target = torch.isin(domain, torch.tensor(self.target_domains, device=domain.device))  # (B,)
            is_source = ~is_target  # (B,)
            motor_embed = self.motor_encoder(motor)  # (B, embed_dim)

            alignment_loss = torch.tensor(0.0, device=eeg.device)
            siglip_loss_target = torch.tensor(0.0, device=motor.device)

            if is_source.any():
                if contrastive_weight != 0:
                    if self.distance_type == "cross_attention":
                        contrastive_loss = relative_contrastive_loss(
                            eeg_embed[is_source], 
                            motor_embed[is_source], 
                            self.cross_attention_distance,
                            temperature=self.temperature
                        )
                    elif self.distance_type == "cosine":
                        # cosine 下直接用 pairwise 距离矩阵
                        B = eeg_embed[is_source].size(0)
                        eeg_expand = eeg_embed[is_source].unsqueeze(1).expand(B, B, -1)
                        motor_expand = motor_embed[is_source].unsqueeze(0).expand(B, B, -1)
                        distances = 1 - F.cosine_similarity(eeg_expand, motor_expand, dim=-1)  # (B,B)
                        similarity = -distances / self.temperature
                        # 用 relative_contrastive_loss 的内部逻辑需要 callable，这里直接重用
                        def cosine_distance_fn(eeg_f, motor_f):
                            return 1 - F.cosine_similarity(eeg_f, motor_f, dim=-1)
                        contrastive_loss = relative_contrastive_loss(
                            eeg_embed[is_source],
                            motor_embed[is_source],
                            cosine_distance_fn,
                            temperature=self.temperature
                        )
                    else:
                        raise ValueError(f"Unknown distance_type: {self.distance_type}")

                    contrastive_loss = contrastive_loss / 100
                else:
                    contrastive_loss = torch.tensor(0.0, device=motor.device)

                motor_pred = self.motor_decoder(eeg_embed)  # (B, 400, 6)
                if reconstruction_weight != 0:
                    reconstruction_loss = self.mse_loss(motor_pred[is_source], motor[is_source])  
                else:
                    reconstruction_loss = torch.tensor(0.0, device=motor.device)

                if prediction_weight != 0:
                    last_frame_loss = self.mse_loss(motor_pred[is_source, -1, :], motor[is_source, -1, :])  
                else:
                    last_frame_loss = torch.tensor(0.0, device=motor.device)
            
            total_loss = contrastive_weight * contrastive_loss + \
                        reconstruction_weight * reconstruction_loss + \
                        alignment_weight * alignment_loss + \
                        prediction_weight + last_frame_loss

            logs = {
                'loss_total': total_loss.item(),
                'loss_contrastive': contrastive_loss.item(),
                'siglip_loss_target': siglip_loss_target.item(),
                'loss_reconstruction': reconstruction_loss.item(),
                'last_frame_loss': last_frame_loss.item(),
                'loss_coral': alignment_loss.item()
            }

            return total_loss, logs
        else:
            motor_pred = self.motor_decoder(eeg_embed)  # (B, 400, 6)
            return 0, motor_pred[:, -1,:].to(torch.float32)  # (B, 6)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

class EEGMotorContrastiveModel_multihead(nn.Module):
    def __init__(self,
                 num_eeg_channels=59,
                 time_steps=400,
                 embed_dim=128,
                 motor_input_dim=6,
                 kernel_width=20,
                 pool_width=2,
                 nFiltLaterLayer = [20, 40, 80, 160, 240, 320],
                 dropout_p=0.5,
                 max_norm=2,
                 contrastive_margin=1.0,
                 n_domains=100, 
                 target_domains = [i for i in range(80,99)]):
        super().__init__()
        self.embed_dim = embed_dim
        self.contrastive_margin = contrastive_margin
        # self.reconstruction_weight = reconstruction_weight
        # self.prediction_weight = prediction_weight
        # self.alignment_weight = alignment_weight
        self.n_domains = n_domains
        self.target_domains = target_domains
        self.output_dim = motor_input_dim

        # self.domain_encoder = DomainEncoder(n_domains=n_domains, num_eeg_channels=num_eeg_channels)

        self.eeg_encoder = EEGEncoder(
            num_eeg_channels=num_eeg_channels,
            time_steps=time_steps,
            embed_dim=embed_dim,
            kernel_width=kernel_width,
            pool_width=pool_width,
            nFiltLaterLayer = nFiltLaterLayer,
            dropout_p=dropout_p,
            max_norm=max_norm
        )

        self.motor_encoder = MotorEncoder(
            time_steps=time_steps,
            input_dim=motor_input_dim,
            embed_dim=embed_dim
        )

        self.motor_decoder = MotorDecoder(
            embed_dim=embed_dim,
            time_steps=time_steps,
            output_dim=motor_input_dim
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))  # 可学习的 temperature
        self.logit_bias = nn.Parameter(torch.tensor(0.0))    # 可学习的 bias
        self.domain_linear = nn.Linear(embed_dim, self.n_domains)

        self.output_layer = PredDecoder(
            embed_dim=embed_dim,
            output_dim=self.output_dim * self.n_domains,
            hidden_dims=[128, 64]
        )
        
        
        self.loss_fn = nn.MSELoss()

    def forward(self, batch, ratio=20, contrastive_weight=1, reconstruction_weight = 0.5, prediction_weight=0.5, alignment_weight=10, loss_cross_weight=2):  # 输入 x: (B, 1, 59, 400)
        eeg = batch['EEG']  # (B, 1, 59, 400)
        motor = batch['Y']  # (B, 400, 6)
        domain = batch['domain']  # (B,)

        # domain_embed = self.domain_encoder(domain)  # (B, 59)
        # domain_embed = domain_embed.unsqueeze(-1)   # (B, 59, 1)
        # eeg = eeg.squeeze(1) + domain_embed         # 原始 EEG: (B, 59, 400)
        # eeg = eeg.unsqueeze(1)                      # 恢复形状 (B, 1, 59, 400)

        eeg_embed = self.eeg_encoder(eeg)     # (B, embed_dim)
        # Reconstruction loss

        domain_weights = self.domain_linear(eeg_embed) # (B, n_domains)
        domain_weights[torch.arange(eeg_embed.size(0)), domain] = -float('inf')  # (B, 100)
        domain_weights[:, self.target_domains] = -float('inf')# (B, 100)
        domain_weights = torch.softmax(domain_weights, dim=-1)  # 归一化，使其总和为 1
        domain_weights = domain_weights.unsqueeze(-1) #(B, n_domains, 1)

        x = self.output_layer(eeg_embed) ## (B, n_domains * 6)
        logits_interdomain = x.view(x.size(0), self.n_domains, self.output_dim)  # (B, n_domains, 6)
        logits_cross = torch.sum(logits_interdomain * domain_weights, dim=1)  # (B, n_classes)
        
        # 评估模式
        if not self.training:
            return 0, logits_cross.to(torch.float32)
        
        
        if self.training:
            

            # === 创建掩码：哪些样本是 source domain ===
            # domain: (B, 1), target_domains: List or Tensor
            is_target = torch.isin(domain, torch.tensor(self.target_domains, device=domain.device))  # (B,)
            is_source = ~is_target  # (B,)
            # Contrastive loss (L2 distance)
            motor_embed = self.motor_encoder(motor)  # (B, embed_dim)
            # 只计算 source domain 的 reconstruction loss
            
            if is_source.any() and is_target.any():
                alignment_loss = coral_loss(eeg_embed[is_source], eeg_embed[is_target]) * x.shape[1]  # (B, embed_dim)
            else:
                alignment_loss = torch.tensor(0.0, device=eeg.device)
            if is_target.any():
                siglip_loss_target = sigmoid_contrastive_loss(eeg_embed[is_target], motor_embed[is_target], self.temperature, self.logit_bias, ratio=ratio, save_vis=False)
            else:
                siglip_loss_target = torch.tensor(0.0, device=motor.device)
            if is_source.any():

                contrastive_loss = sigmoid_contrastive_loss(eeg_embed[is_source], motor_embed[is_source], self.temperature, self.logit_bias, ratio=ratio, save_vis=False)  # (B, embed_dim)
                # contrastive_loss = siglip_loss #contrastive_mse_loss + 
                
                motor_pred = self.motor_decoder(eeg_embed)  # (B, 400, 6)
                reconstruction_loss = self.loss_fn(motor_pred[is_source], motor[is_source])  # (B, 400, 6)
                if prediction_weight != 0:
                    logits_intra = logits_interdomain[torch.arange(x.size(0)), domain]  # (B, n_classes)
                    # 计算监督损失 (L_pred)
                    loss_intra = self.loss_fn(logits_intra[is_source], motor[is_source, -1, :])
                    
                    loss_cross = self.loss_fn(logits_cross[is_source], motor[is_source, -1, :])  # 让 cross-domain 预测接近 intra-domain

                    prediction_loss = loss_intra + loss_cross_weight * loss_cross
                else:
                    prediction_loss = torch.tensor(0.0, device=motor.device)
                    loss_intra = torch.tensor(0.0, device=motor.device)
                    loss_cross = torch.tensor(0.0, device=motor.device)
            else:
                print("No source domain samples found in the batch.")
                contrastive_loss = torch.tensor(0.0, device=motor.device)
                reconstruction_loss = torch.tensor(0.0, device=motor.device)
                prediction_loss = torch.tensor(0.0, device=motor.device)

            
            total_loss = contrastive_weight * contrastive_loss + \
                        reconstruction_weight * reconstruction_loss + \
                        prediction_weight * prediction_loss + \
                        alignment_weight * alignment_loss

            logs = {
                'loss_total': total_loss.item(),
                'loss_contrastive': contrastive_loss.item(),
                'siglip_loss_target': siglip_loss_target.item(),
                'loss_reconstruction': reconstruction_loss.item(),
                'loss_prediction': prediction_loss.item(),
                'loss_intra': loss_intra.item(),
                'loss_cross': loss_cross.item(),
                'loss_coral': alignment_loss.item()
            }

            return total_loss, logs
        else:
            
            return 0, logits_cross.to(torch.float32)  # (B, 6)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

class DomainEncoder(nn.Module):
    def __init__(self, n_domains, num_eeg_channels):
        super().__init__()
        self.embedding = nn.Embedding(n_domains, num_eeg_channels)

    def forward(self, domain):  # domain: (B,)
        return self.embedding(domain)  # 输出: (B, num_eeg_channels)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self._apply_weight_constraint()
        return super().forward(x)

    def _apply_weight_constraint(self):
        with torch.no_grad():
            norm = self.weight.norm(2, dim=(1, 2, 3), keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            self.weight *= (desired / (1e-8 + norm))

def coral_loss(source, target):
    """
    CORAL loss between source and target.
    Inputs:
        source: (N_s, D)
        target: (N_t, D)
    Returns:
        scalar loss value
    """
    d = source.size(1)
    
    # source covariance
    source_c = source - source.mean(dim=0, keepdim=True)
    source_cov = (source_c.T @ source_c) / (source.size(0) - 1)
    
    # target covariance
    target_c = target - target.mean(dim=0, keepdim=True)
    target_cov = (target_c.T @ target_c) / (target.size(0) - 1)

    # Frobenius norm
    loss = torch.mean((source_cov - target_cov) ** 2)
    return loss / (4 * d * d)

import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid_contrastive_loss(eeg_embed, motor_embed, temperature=0.1, logit_bias=0.0, ratio=20, save_path='similarity_heatmap_target.png', save_vis=False):
    """
    SigLIP-style contrastive loss using sigmoid + BCE.
    Positive pairs are aligned on diagonal.
    Negatives are subsampled to maintain a 1:ratio positive:negative balance.
    Also includes simple checks for embedding collapse.
    """
    # Normalize embeddings
    eeg_embed = F.normalize(eeg_embed, dim=-1)
    motor_embed = F.normalize(motor_embed, dim=-1)

    # ---------- Collapse Check ----------
    # def check_collapse(embedding, name=""):
    #     std = embedding.std(dim=0)
    #     mean_std = std.mean().item()
    #     if mean_std < 1e-3:
    #         print(f"[Collapse Warning] {name} embedding might be collapsed! Mean STD: {mean_std:.6f}")

    # check_collapse(eeg_embed, "EEG")
    # check_collapse(motor_embed, "Motor")

    batch_size = eeg_embed.size(0)

    # Compute similarity matrix
    logits = torch.matmul(eeg_embed, motor_embed.T)
    logits = logits / temperature + logit_bias
    if save_vis:
        # ---------- Save Heatmap ----------
        similarity_matrix = logits.detach().cpu().numpy()
        plt.figure(figsize=(6, 5))
        sns.heatmap(similarity_matrix, cmap='coolwarm', square=True, cbar=True)
        plt.title("EEG-Motor Similarity (Logits)")
        plt.xlabel("Motor Embedding Index")
        plt.ylabel("EEG Embedding Index")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # do not show

    # Collapse check on similarity
    # diag_mean = logits.diagonal().mean().item()
    # off_diag = logits.masked_select(~torch.eye(batch_size, dtype=torch.bool, device=logits.device))
    # off_diag_mean = off_diag.mean().item()
    # if abs(diag_mean - off_diag_mean) < 0.01 and ratio > 10:
    #     print(f"[Collapse Warning] Logits may be collapsed! Diagonal mean: {diag_mean:.4f}, Off-diagonal mean: {off_diag_mean:.4f}")

    # Targets
    targets = torch.eye(batch_size, device=eeg_embed.device)

    # Step 1: Positive samples
    pos_indices = torch.arange(batch_size, device=eeg_embed.device)
    pos_logits = logits[pos_indices, pos_indices]
    pos_targets = torch.ones_like(pos_logits)

    # Step 2: Negative samples
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=eeg_embed.device)
    neg_logits_all = logits[mask]
    neg_targets_all = torch.zeros_like(neg_logits_all)

    num_pos = batch_size
    num_neg = min(ratio * num_pos, neg_logits_all.size(0))

    perm = torch.randperm(neg_logits_all.size(0), device=eeg_embed.device)
    neg_logits = neg_logits_all[perm[:num_neg]]
    neg_targets = neg_targets_all[perm[:num_neg]]

    # Step 3: Concatenate
    logits_final = torch.cat([pos_logits, neg_logits], dim=0)
    targets_final = torch.cat([pos_targets, neg_targets], dim=0)

    # Step 4: BCE Loss
    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits_final, targets_final)

    return loss

def relative_contrastive_loss(eeg_embed, motor_embed, cross_attention_distance_module, temperature=1.0):
    """
    Adapted listwise-style relative contrastive loss based on pairwise similarity,
    compatible with the same input format as the top-k contrastive loss.
    
    Args:
        eeg_embed: (B, embed_dim)
        motor_embed: (B, embed_dim)
        cross_attention_distance_module: callable, input (B*B, embed_dim), output (B*B,)
        tau: temperature for softmax scaling
    Returns:
        loss: scalar tensor
    """
    B = eeg_embed.size(0)

    # Step 1: Compute pairwise similarity (similar to the first loss)
    eeg_expand = eeg_embed.unsqueeze(1).expand(B, B, -1)       # (B, B, D)
    motor_expand = motor_embed.unsqueeze(0).expand(B, B, -1)   # (B, B, D)

    eeg_flat = eeg_expand.reshape(B * B, -1)
    motor_flat = motor_expand.reshape(B * B, -1)

    distances_flat = cross_attention_distance_module(eeg_flat, motor_flat)
    distances = distances_flat.view(B, B)  # (B, B)
    similarity = -distances / temperature          # (B, B)

    loss = torch.zeros(B, device=eeg_embed.device)

    for i in range(B):
        sims = similarity[i]  # shape (B,)
        
        # Sort indices by similarity (descending), so most positive comes last
        sorted_sim, sorted_inds = torch.sort(sims, descending=True)  # (B,)
        
        # For each position j, compute log-softmax probability:
        # log(p_j) = s_j - log(sum_{k=j}^{B-1} e^{s_k})
        exp_sorted_sim = torch.exp(sorted_sim)
        cumsum_exp = torch.cumsum(exp_sorted_sim.flip(0), dim=0).flip(0)  # reverse cumulative sum

        eps = 1e-8  # or 1e-6
        log_probs = sorted_sim[1:] - torch.log(cumsum_exp[1:] + eps)
        loss_i = -torch.sum(log_probs)
        loss[i] = loss_i

    return loss.mean()

def sampled_relative_contrastive_loss(eeg_embed, motor_embed, cross_attention_distance_module, temperature=1.0, ratio=20):
    """
    Listwise-style contrastive loss with negative sample subsampling.

    Args:
        eeg_embed: (B, D)
        motor_embed: (B, D)
        cross_attention_distance_module: function, computes distance for (B*B, D) → (B*B,)
        temperature: float, temperature scaling
        ratio: int, number of negatives to sample per positive

    Returns:
        Scalar contrastive loss
    """
    B = eeg_embed.size(0)
    device = eeg_embed.device
    loss = torch.zeros(B, device=device)

    for i in range(B):
        anchor = eeg_embed[i].unsqueeze(0)  # (1, D)
        positive = motor_embed[i].unsqueeze(0)  # (1, D)

        # Build candidate negatives (exclude index i)
        neg_indices = torch.tensor([j for j in range(B) if j != i], device=device)
        perm = torch.randperm(len(neg_indices), device=device)
        num_neg = min(ratio, len(neg_indices))
        sampled_neg_indices = neg_indices[perm[:num_neg]]

        # Combine positive and negatives
        cand_motor = torch.cat([positive, motor_embed[sampled_neg_indices]], dim=0)  # (1+N, D)
        cand_eeg = anchor.expand_as(cand_motor)  # (1+N, D)

        # Compute pairwise distances
        pairwise_input = torch.cat([cand_eeg, cand_motor], dim=1)  # concat features (optional)
        distances = cross_attention_distance_module(cand_eeg, cand_motor)  # shape (1+N,)
        sim = -distances / temperature  # similarity

        # listwise loss: log-softmax (positive is at index 0)
        exp_sim = torch.exp(sim)
        denom = torch.cumsum(exp_sim.flip(0), dim=0).flip(0)
        log_probs = sim[1:] - torch.log(denom[1:])
        loss_i = -torch.sum(log_probs)
        loss[i] = loss_i

    return loss.mean()

def sigmoid_contrastive_loss_with_distance(
    eeg_embed,
    motor_embed,
    cross_attention_distance_module,
    temperature=0.1,
    logit_bias=0.0,
    ratio=20,
    save_path='similarity_heatmap_target.png',
    save_vis=False,
    bidirectional=False,
    monitor=False
):
    """
    Sigmoid contrastive loss with learned distance function and rank tracking.
    Supports optional bidirectional loss (EEG->Motor and Motor->EEG).
    """

    # Step 0: Normalize
    eeg_embed = F.normalize(eeg_embed, dim=-1)
    motor_embed = F.normalize(motor_embed, dim=-1)
    B = eeg_embed.size(0)
    device = eeg_embed.device

    def compute_similarity(e1, e2, direction='eeg_to_motor'):
        # Expand all pairwise (B, B, D)
        e1_expand = e1.unsqueeze(1).expand(B, B, -1)
        e2_expand = e2.unsqueeze(0).expand(B, B, -1)
        e1_flat = e1_expand.reshape(B * B, -1)
        e2_flat = e2_expand.reshape(B * B, -1)
        # Compute distances and convert to similarity
        dist_flat = cross_attention_distance_module(e1_flat, e2_flat)
        similarity = -dist_flat.view(B, B) / temperature + logit_bias  # (B, B)

        # Optional visualization
        if save_vis and direction == 'eeg_to_motor':
            sim_np = similarity.detach().cpu().numpy()
            plt.figure(figsize=(6, 5))
            sns.heatmap(sim_np, cmap='coolwarm', square=True, cbar=True)
            plt.title("EEG-Motor Similarity (Learned Distance)")
            plt.xlabel("Motor Embedding Index")
            plt.ylabel("EEG Embedding Index")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return similarity

    def compute_siglip_loss(similarity):
        # Positive pairs: diagonal
        pos_indices = torch.arange(B, device=device)
        pos_logits = similarity[pos_indices, pos_indices]
        pos_targets = torch.ones_like(pos_logits)

        # Negative pairs: off-diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_logits_all = similarity[mask]
        neg_targets_all = torch.zeros_like(neg_logits_all)

        num_pos = B
        num_neg = min(ratio * num_pos, neg_logits_all.size(0))
        perm = torch.randperm(neg_logits_all.size(0), device=device)
        neg_logits = neg_logits_all[perm[:num_neg]]
        neg_targets = neg_targets_all[perm[:num_neg]]

        # Concatenate
        logits_final = torch.cat([pos_logits, neg_logits], dim=0)
        targets_final = torch.cat([pos_targets, neg_targets], dim=0)

        # BCE Loss
        bce = nn.BCEWithLogitsLoss()
        loss = bce(logits_final, targets_final)
        return loss

    def monitor_rank(similarity, direction='EEG→Motor'):
        with torch.no_grad():
            sorted_indices = similarity.argsort(dim=1, descending=True)
            correct = torch.arange(B, device=similarity.device)
            correct_ranks = (sorted_indices == correct.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
            top1_acc = (correct_ranks == 0).float().mean().item()
            avg_rank = correct_ranks.float().mean().item()
            print(f"[Rank Stat] {direction}  top-1 acc: {top1_acc:.3f}, avg rank: {avg_rank:.2f}")

    # ---------- Forward direction ----------
    sim_fwd = compute_similarity(eeg_embed, motor_embed, direction='eeg_to_motor')
    loss_fwd = compute_siglip_loss(sim_fwd)
    if monitor:
        monitor_rank(sim_fwd, direction="EEG→Motor")

    if not bidirectional:
        return loss_fwd

    # ---------- Backward direction ----------
    sim_bwd = compute_similarity(motor_embed, eeg_embed, direction='motor_to_eeg')
    loss_bwd = compute_siglip_loss(sim_bwd)
    if monitor:
        monitor_rank(sim_bwd, direction="Motor→EEG")

    # Final loss
    loss = 0.5 * (loss_fwd + loss_bwd)
    return loss


