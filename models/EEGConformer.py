from braindecode.models import EEGConformer
import torch.nn as nn
import torch
import inspect
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss


from utils.utils import get_metrics

class MyEEGConformer(EEGConformer):
    def __init__(self,
                 # 原始模型参数
                 n_outputs,        # 对应你的config.eegnet.num_chan_kin
                 n_chans,          # 对应你的config.eegnet.num_chan_eeg
                 n_times,          # 对应你的config.eegnet.eeg.time_step
                 final_fc_length=280,
                 add_log_softmax=False,
                 loss_name='mse',
                 **kwargs):
        
        # 调用父类初始化
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            final_fc_length=final_fc_length,
            add_log_softmax=add_log_softmax,
            **kwargs
        )
        self.loss_name = loss_name
        if self.loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_name == 'freq_mse':
            self.loss_fn = FreDFLoss()
        elif self.loss_name == 'reward_mse':
            self.loss_fn = MSEWithEncouragingLoss()
        elif self.loss_name == 'freq_reward_mse':
            self.loss_fn = FreDFLossWithEncouraging()
        
        
    
    def forward(self, batch, metrics=None):
        # x = self.allButLastLayers(x)
        x = batch['EEG'] #(B, 1, 59, 400)
        Y = batch['Y']
        # 先执行原始前向传播
        x = super().forward(x)
        loss = self.loss_fn(x, Y)
        if not self.training:
            return loss.item(), x.to(torch.float32)
        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(x.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return loss, log
        
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
    
