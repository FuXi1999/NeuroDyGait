from torch import nn
from model.neural_transformer import NeuralTransformer, NTConfig
import torch
from torch.functional import F
import inspect
from collections import OrderedDict
from utils import get_metrics
from model.transformer import Block


class AlignmentModule(nn.Module):
    def __init__(self, n_classes = 7, num_position = 128, dim=256):
        super(AlignmentModule, self).__init__()
        self.base_vectors = nn.Parameter(torch.randn(n_classes, dim))
        self.conv = nn.Conv2d(1, 3, kernel_size=(num_position, 1), stride=(1, 1), padding=(0, 0), bias=False)
    
    def forward(self, input, Y):
        # input [batchsize, n, 256]
        # Y [batchsize]
        # input = torch.mean(input, dim=1, keepdim=True) # [batch_size, 1, 256]
        input = torch.unsqueeze(input, dim=1) # [batch_size, 1, 128, 256]
        input = self.conv(input) # [batch_size, 1, 1, 256]
        input = torch.squeeze(input, dim=2) # [batch_size, 1, 256]
        Y = torch.unsqueeze(Y, dim=-1) # [batch_size, 1]
        base_vectors = F.embedding(Y, self.base_vectors) # [batch_size, 1, 256]
        sim = torch.mean((base_vectors - input) ** 2, dim=-1) # [batch_size, 1]
        sim = torch.mean(sim)

        return input, sim


# model for fine-tuning
class FT(nn.Module):
    def __init__(self, EEG_config, EOG_config, ECG_config, EMG_config, pretrained_ckpt_path=None
                 , n_classes=7, regression=False, emb_dropout = 0.5, loss_ratio=[0.5, 0.5, 0.5, 0.5, 0.5], **kwargs):
        super().__init__()
        print('teacher loss ratio: (ignore if training student)')
        print(loss_ratio)
        self.EEG_encoder = NeuralTransformer(EEG_config) if EEG_config is not None else None
        self.EOG_encoder = NeuralTransformer(EOG_config) if EOG_config is not None else None
        self.ECG_encoder = NeuralTransformer(ECG_config) if ECG_config is not None else None
        self.EMG_encoder = NeuralTransformer(EMG_config) if EMG_config is not None else None

        self.use_EEG = True if EEG_config is not None else False
        self.use_EOG = True if EOG_config is not None else False
        self.use_ECG = True if ECG_config is not None else False
        self.use_EMG = True if EMG_config is not None else False

        self.num_modalities = sum([self.use_EEG, self.use_EOG, self.use_ECG, self.use_EMG])
        self.loss_ratio = loss_ratio

        if pretrained_ckpt_path is not None:
            print('loading weight from pretrained_ckpt')
            pretrained_ckpt = torch.load(pretrained_ckpt_path)['model']
            EEG_dict = OrderedDict()
            EOG_dict = OrderedDict()
            ECG_dict = OrderedDict()
            EMG_dict = OrderedDict()
            for key in list(pretrained_ckpt.keys()):
                if key.startswith('EEG_encoder.'):
                    EEG_dict[key[len('EEG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('EOG_encoder.'):
                    EOG_dict[key[len('EOG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('ECG_encoder.'):
                    ECG_dict[key[len('ECG_encoder.'):]] = pretrained_ckpt[key]
                elif key.startswith('EMG_encoder.'):
                    ECG_dict[key[len('EMG_encoder.'):]] = pretrained_ckpt[key]
            if EEG_config is not None:
                self.EEG_encoder.load_state_dict(EEG_dict, strict=False)
            if EOG_config is not None:
                self.EOG_encoder.load_state_dict(EOG_dict, strict=False)
            if ECG_config is not None:
                self.ECG_encoder.load_state_dict(ECG_dict, strict=False)
            if EMG_config is not None:
                self.EMG_encoder.load_state_dict(EMG_dict, strict=False)

        self.dim = 128
        # Embedding for EEG, EOG, ECG to the same feature length
        self.EEG_embedding = nn.Sequential(
            nn.LayerNorm(EEG_config.n_embd),
            nn.Linear(EEG_config.n_embd, self.dim),
            nn.LayerNorm(self.dim)
        ) if EEG_config is not None else None
        self.EOG_embedding = nn.Sequential(
            nn.LayerNorm(EOG_config.n_embd),
            nn.Linear(EOG_config.n_embd, self.dim),
            nn.LayerNorm(self.dim)   
        ) if EOG_config is not None else None
        self.ECG_embedding = nn.Sequential(
            nn.LayerNorm(ECG_config.n_embd),
            nn.Linear(ECG_config.n_embd, self.dim),
            nn.LayerNorm(self.dim)
        ) if ECG_config is not None else None
        self.EMG_embedding = nn.Sequential(
            nn.LayerNorm(EMG_config.n_embd),
            nn.Linear(EMG_config.n_embd, self.dim),
            nn.LayerNorm(self.dim)
        ) if EMG_config is not None else None

        self.dropout = nn.Dropout(emb_dropout)

        self.num_position = 128
        self.EEG_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_position)
        ) if EEG_config is not None else None
        self.EOG_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_position)
        ) if EOG_config is not None else None
        self.ECG_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_position)
        ) if ECG_config is not None else None
        self.EMG_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_position)
        ) if EMG_config is not None else None

        self.embd_dim = 128
        self.EEG_Linear = nn.Linear(EEG_config.n_embd, self.embd_dim) if EEG_config is not None else None
        self.EOG_Linear = nn.Linear(EOG_config.n_embd, self.embd_dim) if EOG_config is not None else None
        self.ECG_Linear = nn.Linear(ECG_config.n_embd, self.embd_dim) if ECG_config is not None else None
        self.EMG_Linear = nn.Linear(EMG_config.n_embd, self.embd_dim) if EMG_config is not None else None

        self.alignment_module = AlignmentModule(n_classes, self.num_position, self.embd_dim)
 
        transformer_args = dict(n_layer=12, n_head=8, n_embd=self.embd_dim * self.num_modalities, block_size=1024, patch_size=200, 
                            bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=8)
        transformer_conf = NTConfig(**transformer_args)
        self.X_transformer = Block(transformer_conf)

        self.ms_heads = nn.ModuleList([
                            nn.Linear(self.embd_dim, n_classes) if EEG_config is not None else None,
                            nn.Linear(self.embd_dim, n_classes) if EOG_config is not None else None,
                            nn.Linear(self.embd_dim, n_classes) if ECG_config is not None else None,
                            nn.Linear(self.embd_dim, n_classes) if EMG_config is not None else None
                        ])
        self.lm_head = nn.Linear(self.embd_dim * self.num_modalities, n_classes)

        if regression:
            self.loss_fn = nn.MSELoss()
        elif n_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()


        self.lm_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def cal_accuracy(self, logits, targets):
        _, preds = torch.max(logits, dim=-1)
        accuracy = torch.sum(preds == targets).float() / targets.size(0)
        return accuracy.item()
    
    def reorganization(self, feature, position):
        feature = torch.matmul(feature.transpose(1, 2).contiguous(), position)
        feature = feature.transpose(1, 2).contiguous()
        return feature

    def forward(self, batch, metrics=None, mask=None, return_all_tokens=True):
        EEG_X = batch['EEG_X']; EOG_X = batch['EOG_X']; ECG_X = batch['ECG_X']; EMG_X = batch['EMG_X']
        Y = batch['Y']
        EEG_input_chans = batch['EEG_input_chans']; EEG_input_time = batch['EEG_input_time']
        EOG_input_chans = batch['EOG_input_chans']; EOG_input_time = batch['EOG_input_time']
        ECG_input_chans = batch['ECG_input_chans']; ECG_input_time = batch['ECG_input_time']
        EMG_input_chans = batch['EMG_input_chans']; EMG_input_time = batch['EMG_input_time']

        EEG_X_inputs = self.EEG_encoder(EEG_X, EEG_input_chans, EEG_input_time, mask, return_all_tokens) if EEG_X is not None else None
        EOG_X_inputs = self.EOG_encoder(EOG_X, EOG_input_chans, EOG_input_time, mask, return_all_tokens) if EOG_X is not None else None
        ECG_X_inputs = self.ECG_encoder(ECG_X, ECG_input_chans, ECG_input_time, mask, return_all_tokens) if ECG_X is not None else None
        EMG_X_inputs = self.EMG_encoder(EMG_X, EMG_input_chans, EMG_input_time, mask, return_all_tokens) if EMG_X is not None else None
        
        EEG_X = self.dropout(self.EEG_embedding(EEG_X_inputs)) if EEG_X is not None else None #(B, seq_len_EEG, dim)
        EOG_X = self.dropout(self.EOG_embedding(EOG_X_inputs)) if EOG_X is not None else None #(B, seq_len_EOG, dim)
        ECG_X = self.dropout(self.ECG_embedding(ECG_X_inputs)) if ECG_X is not None else None #(B, seq_len_ECG, dim)
        EMG_X = self.dropout(self.EMG_embedding(EMG_X_inputs)) if EMG_X is not None else None #(B, seq_len_EMG, dim)

        EEG_X = self.EEG_head(EEG_X) if EEG_X is not None else None #(B, seq_len_EEG, num_position)
        EOG_X = self.EOG_head(EOG_X) if EOG_X is not None else None #(B, seq_len_EOG, num_position)
        ECG_X = self.ECG_head(ECG_X) if ECG_X is not None else None #(B, seq_len_ECG, num_position)
        EMG_X = self.ECG_head(EMG_X) if EMG_X is not None else None #(B, seq_len_ECG, num_position)

        EEG_X = torch.softmax(EEG_X, dim=-1) if EEG_X is not None else None #(B, seq_len_EEG, num_position)
        EOG_X = torch.softmax(EOG_X, dim=-1) if EOG_X is not None else None #(B, seq_len_EOG, num_position)
        ECG_X = torch.softmax(ECG_X, dim=-1) if ECG_X is not None else None #(B, seq_len_ECG, num_position)
        EMG_X = torch.softmax(EMG_X, dim=-1) if EMG_X is not None else None #(B, seq_len_EMG, num_position)

        EEG_X = self.reorganization(EEG_X_inputs, EEG_X) if EEG_X is not None else None #(B, num_position, EEG_config.n_embd)
        EOG_X = self.reorganization(EOG_X_inputs, EOG_X) if EOG_X is not None else None #(B, num_position, EOG_config.n_embd)
        ECG_X = self.reorganization(ECG_X_inputs, ECG_X) if ECG_X is not None else None #(B, num_position, ECG_config.n_embd)
        EMG_X = self.reorganization(ECG_X_inputs, EMG_X) if EMG_X is not None else None #(B, num_position, EMG_config.n_embd)

        EEG_X = self.EEG_Linear(EEG_X) if EEG_X is not None else None #(B, num_position, embd_dim)
        EOG_X = self.EOG_Linear(EOG_X) if EOG_X is not None else None #(B, num_position, embd_dim)
        ECG_X = self.ECG_Linear(ECG_X) if ECG_X is not None else None #(B, num_position, embd_dim)
        EMG_X = self.EMG_Linear(EMG_X) if EMG_X is not None else None #(B, num_position, embd_dim)

        feature = 0
        specific_loss = 0
        for (modality, ratio, head) in [X for X in [(EEG_X, self.loss_ratio[1], self.ms_heads[0]), \
                                                    (EOG_X, self.loss_ratio[2], self.ms_heads[1]), \
                                                    (ECG_X, self.loss_ratio[3], self.ms_heads[2]), \
                                                    (EMG_X, self.loss_ratio[4], self.ms_heads[3])] if X[0] is not None]:
            feature += modality

            if 'f1_weighted' in metrics or 'r2' in metrics:

                specific_loss += ratio * self.loss_fn(head(torch.mean(modality, dim=1)), Y)
            else:
                specific_loss += ratio * self.loss_fn(head(torch.mean(modality, dim=1)), Y.float().unsqueeze(-1))
            
        feature /= self.num_modalities


        if EEG_X is not None:
            B = EEG_X.size(0)
            device = EEG_X.device   
        if EOG_X is not None:
            B = EOG_X.size(0)
            device = EOG_X.device
        if ECG_X is not None:
            B = ECG_X.size(0)
            device = ECG_X.device
        if EMG_X is not None:
            B = EMG_X.size(0)
            device = EMG_X.device
            

        if self.use_EEG and EEG_X is None:
            EEG_X = torch.zeros(B, self.num_position, self.embd_dim).to(device)
        if self.use_EOG and EOG_X is None:
            EOG_X = torch.zeros(B, self.num_position, self.embd_dim).to(device)
        if self.use_ECG and ECG_X is None:
            ECG_X = torch.zeros(B, self.num_position, self.embd_dim).to(device)
        if self.use_EMG and EMG_X is None:
            EMG_X = torch.zeros(B, self.num_position, self.embd_dim).to(device)

        X = torch.cat([x for x in [EEG_X, EOG_X, ECG_X, EMG_X] if x is not None], dim=-1) #(B, num_position, embd_dim * self.num_modalities)
        X = self.X_transformer(X) #(B, num_position, embd_dim * self.num_modalities)
        X = torch.mean(X, dim=1) #(B, embd_dim * self.num_modalities)
        self.feature = X #(B, embd_dim * self.num_modalities)

        
        logits = self.lm_head(X)

        loss = self.loss_fn(logits, Y)

        total_loss = (1 - self.loss_ratio[0]) * loss + self.loss_ratio[0] * specific_loss


        alignment_loss = torch.tensor(0.0)

        if not self.training:
            return loss.item(), logits.to(torch.float32)

        with torch.amp.autocast('cuda', enabled=False):
            if 'r2' in metrics:
                results = get_metrics(logits.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)
            elif 'f1_weighted' not in metrics:
                # binary classification
                results = get_metrics(torch.sigmoid(logits.to(torch.float32).detach()).cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=True)
            else:
                # multi-class classification
                results = get_metrics(logits.to(torch.float32).detach().cpu().numpy(), Y.cpu().numpy(), metrics, is_binary=False)

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/total_loss'] = loss.item()
        log[f'{split}/loss'] = alignment_loss.item()
        log[f'{split}/alignment_loss'] = alignment_loss.item()
        for key, value in results.items():
            log[f'{split}/{key}'] = value

        return total_loss, log
    
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
