import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
import clip
import math
from einops import rearrange, repeat
from functools import partial

from models.transformer.tools import *


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        # x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, height=200, width=50):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.permute(1, 2, 0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.shape[0], :x.shape[1], None, :]
        return self.dropout(x)


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer2D(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer2D, self).__init__()
        # latent_dim = latent_dim * 2
        # num_heads = num_heads * 2
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        # self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.position2d_enc = PositionalEncoding2D(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        if opt.attnj:
            self.attnj = True
            seqTransEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')
            self.seqTransEncoder2 = nn.TransformerEncoder(seqTransEncoderLayer2,
                                                        num_layers=num_layers)
        else:
            self.attnj = False

        if 'attnt' in opt and opt.attnt:
            self.attnt = True
            seqTransEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')
            self.seqTransEncoder3 = nn.TransformerEncoder(seqTransEncoderLayer3,
                                                        num_layers=num_layers)
        else:
            self.attnt = False

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens2d + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = opt.num_tokens2d
        self.pad_id = opt.num_tokens2d + 1

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens2d, latent_dim=self.latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''

        cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        x = self.token_emb(motion_ids)
        bs, T, J, dim = x.shape
        # print(x.shape)
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x.reshape(bs, T*J, dim).permute(1, 0, 2))

        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)

        x = torch.cat([cond.repeat(J, 1, 1)[None, ...], x.reshape(T, J, bs, x.shape[-1])], dim=0)
        x = self.position2d_enc(x)
        xseq = x.reshape((T+1)*J, bs, x.shape[-1])
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        padding_mask = rearrange(padding_mask, 'b t j -> b (t j)')
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)

        if not self.attnj:
            output = output.reshape(T+1, J, bs, -1)[1:].reshape(T*J, bs, -1)
        else:
            # # attention on joints of one time: J x B*(T+1) x Da
            x2 = output.reshape(T+1, J, bs, -1)
            x2 = self.position2d_enc(x2)
            x2 = rearrange(x2, 't j b d -> j (b t) d')
            padding_mask2 = padding_mask.reshape(bs, T+1, J)
            padding_mask2[:, 0] = True  # text condition do not perform attention on joints
            padding_mask2 = rearrange(padding_mask2, 'b t j -> (b t) j')
            nonpad_ids = torch.where(~padding_mask2[:, 0])[0]
            output2 = self.seqTransEncoder2(x2[:, nonpad_ids])
            x2[:, nonpad_ids] = output2
            output = rearrange(x2.reshape(J, bs, T+1, -1), 'j b t d -> t j b d')[1:].reshape(T*J, bs, -1)

        if self.attnt:
            # # attention on all times of one joint: (T+1) x B*J x D
            padding_mask3 = padding_mask.reshape(bs, T+1, J)
            padding_mask3 = rearrange(padding_mask3, 'b t j -> (b j) t')
            x3 = output.reshape(T, J, bs, -1)
            xseq3 = torch.cat([cond.repeat(J, 1, 1)[None, ...], x3], dim=0)
            xseq3 = self.position2d_enc(xseq3)
            xseq3 = rearrange(xseq3, 't j b d -> t (b j) d')
            output3 = self.seqTransEncoder3(xseq3, src_key_padding_mask=padding_mask3)
            output = output + rearrange(output3.reshape(T+1, bs, J, -1)[1:], 't b j d -> (t j) b d')

        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids_j, y, m_lens):
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        # ids_j = ids_j[..., None]
        device = ids_j.device
        bs, xtokens, ytokens = ids_j.shape
        non_pad_mask = torch.arange(xtokens, device=device).expand(bs, xtokens) < m_lens.unsqueeze(1)
        non_pad_mask = non_pad_mask[..., None].repeat(1, 1, ytokens)
        ids_j = torch.where(non_pad_mask, ids_j, self.pad_id)

        # # Positions that are PADDED are ALL FALSE
        # non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        # ids = torch.where(non_pad_mask, ids, self.pad_id)

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        '''
        Prepare mask
        '''
        ntokens = xtokens * ytokens
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)

        # ========== temporal mask
        num_token_masked = (xtokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, xtokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        # Positions to be MASKED must also be NON-PADDED
        mask = mask & non_pad_mask[..., 0]
        # Note this is our training target, not input
        labels = torch.where(mask[..., None].repeat(1, 1, ytokens), ids_j, self.mask_id)
        x_ids_j = ids_j.clone()
        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
        x_ids_j = torch.where(mask_rid[..., None].repeat(1, 1, ytokens), rand_id, x_ids_j)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids_j = torch.where(mask_mid[..., None].repeat(1, 1, ytokens), self.mask_id, x_ids_j)
        mask_time = mask
        mask_time = mask_time[..., None].repeat(1, 1, ytokens)       # keep temperal mask still masked
        # print((x_ids_j==512).sum(), mask_time.sum(), mask.sum(), (labels!=512).sum(), mask_rid.sum())

        # ========== spatial mask
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        # Positions to be MASKED must also be NON-PADDED
        mask = mask & non_pad_mask.reshape(bs, -1)
        mask = mask & ~mask_time.reshape(bs, -1)
        # Note this is our training target, not input
        labels = torch.where(mask, x_ids_j.reshape(bs, -1), labels.reshape(bs, -1))
        x_ids_j = x_ids_j.reshape(bs, -1)
        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)

        rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
        x_ids_j = torch.where(mask_rid, rand_id, x_ids_j)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        # mask_mid = mask
        x_ids_j = torch.where(mask_mid, self.mask_id, x_ids_j)

        logits = self.trans_forward(x_ids_j.reshape(bs, xtokens, ytokens), cond_vector, ~non_pad_mask, force_mask)

        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 n_j=1,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):
        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )
        non_pad_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < m_lens.unsqueeze(1)
        padding_mask = ~ (non_pad_mask[..., None].repeat(1, 1, n_j))

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.).reshape(batch_size, seq_len*n_j)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens * n_j).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            # logits = logits.reshape(batch_size, seq_len, n_j, -1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(is_mask, pred_ids, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)
            scores = scores.reshape(batch_size, seq_len*n_j)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids


    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        if edit_mask == None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self,
                  conds,
                  conds_og,
                  tokens,
                  m_lens,
                  cond_scale: int,
                  force_mask=False,
                  ):

        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
                if conds_og is not None:
                    cond_vector_og = self.encode_text(conds_og)
                else:
                    cond_vector_og = None
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
            if conds_og is not None:
                cond_vector_og = self.enc_action(conds_og).to(device)
            else:
                cond_vector_og = None
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              cond_vector_neg=cond_vector_og,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
        og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

        return og_tokens_scores


class ResidualTransformer2D(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, **kargs):
        super(ResidualTransformer2D, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        # assert shared_codebook == True, "Only support shared codebook right now!"

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        # self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        # self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        self.position2d_enc = PositionalEncoding2D(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        if opt.attnj:
            self.attnj = True
            seqTransEncoderLayer2 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')
            self.seqTransEncoder2 = nn.TransformerEncoder(seqTransEncoderLayer2,
                                                        num_layers=num_layers)
        else:
            self.attnj = False

        if opt.attnt:
            self.attnt = True
            seqTransEncoderLayer3 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')
            self.seqTransEncoder3 = nn.TransformerEncoder(seqTransEncoderLayer3,
                                                        num_layers=num_layers)
        else:
            self.attnt = False

        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens2d + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = opt.num_tokens2d
        self.pad_id = opt.num_tokens2d + 1

        # self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                # self.output_proj_bias = 0
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, _num_tokens)

        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                output_proj_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    # def

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text


    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            # if not self.registered:
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
                # self.registered = True

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output

    def trans_forward(self, motion_codes, qids, cond, padding_mask, T, J, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes.permute(1, 0, 2))
        N, B, D = x.shape

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)

        x = x.reshape(T, J, B, D)
        xseq = torch.cat([cond.repeat(J, 1, 1)[None, ...], q_emb.repeat(J, 1, 1)[None, ...], x], dim=0)  # (T+2, J, B, latent_dim)
        xseq = self.position2d_enc(xseq)
        xseq = xseq.reshape((T+2)*J, B, D)
        padding_mask = padding_mask.reshape(B, T, J)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, T+2, J)
        padding_mask = rearrange(padding_mask, 'b t j -> b (t j)')
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)  # (T+2 * J, B, D)

        output = output.reshape(T+2, J, B, -1)[2:].reshape(T*J, B, D)
        if self.attnj:
            # # attention on joints of one time: J x B*(T+1) x D
            x2 = output.reshape(T, J, B, D)
            x2 = self.position2d_enc(x2)
            x2 = rearrange(x2, 't j b d -> j (b t) d')
            padding_mask2 = padding_mask.reshape(B, T+2, J)[:, 2:]
            padding_mask2 = rearrange(padding_mask2, 'b t j -> (b t) j')
            nonpad_ids = torch.where(~padding_mask2[:, 0])[0]
            output2 = torch.zeros_like(x2)
            output2[:, nonpad_ids] = self.seqTransEncoder2(x2[:, nonpad_ids])
            output = output + rearrange(output2.reshape(T, B, J, -1), 't b j d -> (t j) b d')
        if self.attnt:
            # # attention on all times of one joint: (T+2) x B*J x D
            x3 = output.reshape(T, J, B, D)
            xseq3 = torch.cat([cond.repeat(J, 1, 1)[None, ...], q_emb.repeat(J, 1, 1)[None, ...], x3], dim=0)
            xseq3 = self.position2d_enc(xseq3)
            xseq3 = rearrange(xseq3, 't j b d -> t (b j) d')
            padding_mask3 = padding_mask.reshape(B, T+2, J)
            padding_mask3 = rearrange(padding_mask3, 'b t j -> (b j) t')
            output3 = self.seqTransEncoder3(xseq3, src_key_padding_mask=padding_mask3)
            output = output + rearrange(output3.reshape(T+2, B, J, -1)[2:], 't b j d -> (t j) b d')

        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                T, J,
                                cond_scale=3,
                                force_mask=False):
        bs = motion_codes.shape[0]
        # if cond_scale == 1:
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, T, J, force_mask=True)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, T, J)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, T, J, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids-1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices, y, m_lens):
        '''
        :param all_indices: (b, n, q)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        self.process_embed_proj_weight()

        bs, xtokens, ytokens, num_quant_layers = all_indices.shape
        ntokens = xtokens * ytokens
        all_indices = all_indices.reshape(bs, ntokens, num_quant_layers)
        device = all_indices.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = torch.arange(xtokens, device=device).expand(bs, xtokens) < m_lens.unsqueeze(1)
        non_pad_mask = non_pad_mask[..., None].repeat(1, 1, ytokens)
        non_pad_mask = non_pad_mask.reshape(bs, xtokens*ytokens)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # print(self.token_embed_weight.shape, all_indices.shape)
        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        # print(token_embed.shape, gather_indices.shape)
        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)

        cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, xtokens, ytokens, force_mask)
        logits = self.output_project(logits, active_q_layers-1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 conds,
                 m_lens,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1, # If it's -1, use all.
                 ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        B, T, J = motion_ids.shape
        motion_ids = motion_ids.reshape(B, T*J)
        seq_len = T
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        non_pad_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < m_lens.unsqueeze(1)
        padding_mask = ~ (non_pad_mask[..., None].repeat(1, 1, J))
        padding_mask = padding_mask.reshape(B, T*J)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, T, J, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        if all_indices.max() == self.mask_id:
            print('!!! Warning: id exceeds mask_id !!!')
            all_indices = all_indices.clip(0, self.mask_id-1)
            # print(all_indices.max())
        # all_indices = all_indices.masked_fill()
        return all_indices.reshape(B, T, J, num_quant_layers)

    @torch.no_grad()
    @eval_decorator
    def edit(self,
            motion_ids,
            conds,
            m_lens,
            temperature=1,
            topk_filter_thres=0.9,
            cond_scale=2
            ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices