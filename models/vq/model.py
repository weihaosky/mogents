from einops import rearrange

import torch
import torch.nn as nn

from models.vq.encdec import Encoder, Decoder, Encoder2d, Decoder2d
from models.vq.residual_vq import ResidualVQ


class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.code_dim1d = args.code_dim1d
        self.num_code1d = args.nb_code1d
        # self.quant = args.quantizer
        self.encoder1d = Encoder(input_width, self.code_dim1d, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder1d = Decoder(input_width, self.code_dim1d, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': self.num_code1d,
            'code_dim': self.code_dim1d,
            'args': args,
        }
        self.quantizer1d = ResidualVQ(**rvqvae_config)

        if args.dataset_name == 'humanml3d':
            self.J0 = 22
        elif args.dataset_name == 'kit':
            self.J0 = 21
        elif args.dataset_name == 'motionx':
            self.J0 = 22
        self.J = 6
        self.JD0 = 12
        # self.JD = 48
        self.encode_dim2d = self.JD0
        self.num_code2d = args.nb_code2d ###########
        self.code_dim2d = args.code_dim2d ###########
        self.encoder2d = Encoder2d(self.encode_dim2d, self.code_dim2d, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.linear_merge = nn.Linear(self.code_dim2d * self.J, self.code_dim2d * self.J)
        self.decoder2d = Decoder2d(self.encode_dim2d, self.code_dim2d, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.linear_out = nn.Linear(input_width*2, input_width)
        joints_rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': self.num_code2d,
            'code_dim': self.code_dim2d,
            'args': args,
        }
        self.quantizer2d = ResidualVQ(**joints_rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x1d, x2d):
        N, T, _ = x1d.shape
        x_in = self.preprocess(x1d)
        x_encoder1d = self.encoder1d(x_in)
        code_idx1d, all_codes1d = self.quantizer1d.quantize(x_encoder1d, return_latent=True)

        B, T0, _, _ = x2d.shape
        if self.J0 == 22:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 21:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 1, 2))  # for downsample X 2
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape
        code_idx2d, all_codes2d = self.quantizer2d.quantize(rearrange(x2d_encode, 'b d t j -> (b j) d t'), return_latent=True)
        n_qt, _, _, _ = all_codes2d.shape
        code_idx2d = code_idx2d.reshape(B, self.J, T, n_qt).permute(0, 2, 1, 3)

        return code_idx1d, None, code_idx2d, None

    def forward(self, x1d, x2d):
        x_in = self.preprocess(x1d)
        # Encode
        x_encoder1d = self.encoder1d(x_in)
        ## quantization
        x_quantized1d, code_idx1d, commit_loss1d, perplexity1d = self.quantizer1d(x_encoder1d, sample_codebook_temp=0.5)
        ## decoder
        x_out1d = self.decoder1d(x_quantized1d)

        B, T0, _, _ = x2d.shape
        if self.J0 == 22:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 21:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 1, 2))  # for downsample X 2
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape

        x2d_quantized, code_idx2d, commit_loss2d, perplexity2d = self.quantizer2d(rearrange(x2d_encode, 'b d t j -> (b j) d t'), sample_codebook_temp=0.5)
        x2d_quantized = x2d_quantized.reshape(B, self.J*D, T)
        x2d_quantized = self.linear_merge(x2d_quantized.permute(0, 2, 1).reshape(B*T, self.J*D)).reshape(B, T, self.J*D).permute(0, 2, 1)
        x2d_out = self.decoder2d(rearrange(x2d_quantized.reshape(B, self.J, D, T), 'b j d t -> b d t j'))

        T0 = x2d_out.shape[1]
        if self.J0 == 22:
            x2d_out = x2d_out.reshape(B, T0, self.J0+2, self.JD0)[:, :, 1:-1, :].reshape(B * T0, self.J0, self.JD0)
            jmotion = torch.zeros([B, T0, 263], device=x2d.device)
        elif self.J0 == 21:
            x2d_out = x2d_out.reshape(B, T0, self.J0+3, self.JD0)[:, :, 1:-2, :].reshape(B * T0, self.J0, self.JD0)
            jmotion = torch.zeros([B, T0, 251], device=x2d.device)
        jmotion[:, :, :4] = x2d_out[:, 0, :4].reshape(B, T0, -1)
        jmotion[:, :, -4:] = x2d_out[:, 0, 4:8].reshape(B, T0, -1)
        jmotion[:, :, 4: 4 + (self.J0 - 1) * 3] = x2d_out[:, 1:, :3].reshape(B, T0, -1)
        jmotion[:, :, 4 + (self.J0 - 1) * 3: 4 + (self.J0 - 1) * 9] = x2d_out[:, 1:, 3:9].reshape(B, T0, -1)
        jmotion[:, :, 4 + (self.J0 - 1) * 9: 4 + (self.J0 - 1) * 9 + self.J0 * 3] = x2d_out[:, :, 9:12].reshape(B, T0, -1)

        x_out2d = self.linear_out(torch.concat([x_out1d, jmotion], dim=2))

        return x_out1d, commit_loss1d, perplexity1d, jmotion, commit_loss2d, perplexity2d, x_out2d

    def forward_decoder(self, x1d=None, x2d=None):
        x_out1d = None
        if x1d is not None:
            x_d = self.quantizer1d.get_codes_from_indices(x1d)
            x1d = x_d.sum(dim=0).permute(0, 2, 1)
            x_out1d = self.decoder1d(x1d)

        x2d_out = None
        if x2d is not None:
            B, T, _, _ = x2d.shape
            x_d_2d = self.quantizer2d.get_codes_from_indices(rearrange(x2d, 'b t j n -> (b j) t n'))
            D = x_d_2d.shape[-1]
            x2d = rearrange(x_d_2d.sum(dim=0).reshape(B, self.J, T, D), 'b j t d -> b (j d) t')
            x2d_quantized = self.linear_merge(x2d.permute(0, 2, 1).reshape(B*T, self.J*D)).reshape(B, T, self.J*D).permute(0, 2, 1)
            x2d_out = self.decoder2d(rearrange(x2d_quantized.reshape(B, self.J, D, T), 'b j d t -> b d t j'))
            T0 = x2d_out.shape[1]
            if self.J0 == 22:
                x2d_out = x2d_out.reshape(B, T0, self.J0+2, self.JD0)[:, :, 1:-1, :].reshape(B * T0, self.J0, self.JD0)
                jmotion = torch.zeros([B, T0, 263], device=x2d.device)
            elif self.J0 == 21:
                x2d_out = x2d_out.reshape(B, T0, self.J0+3, self.JD0)[:, :, 1:-2, :].reshape(B * T0, self.J0, self.JD0)
                jmotion = torch.zeros([B, T0, 251], device=x2d.device)
            jmotion[:, :, :4] = x2d_out[:, 0, :4].reshape(B, T0, -1)
            jmotion[:, :, -4:] = x2d_out[:, 0, 4:8].reshape(B, T0, -1)
            jmotion[:, :, 4: 4 + (self.J0 - 1) * 3] = x2d_out[:, 1:, :3].reshape(B, T0, -1)
            jmotion[:, :, 4 + (self.J0 - 1) * 3: 4 + (self.J0 - 1) * 9] = x2d_out[:, 1:, 3:9].reshape(B, T0, -1)
            jmotion[:, :, 4 + (self.J0 - 1) * 9: 4 + (self.J0 - 1) * 9 + self.J0 * 3] = x2d_out[:, :, 9:12].reshape(B, T0, -1)
            x_out2d = self.linear_out(torch.concat([x_out1d, jmotion], dim=2))

        return x_out1d, x_out2d


class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)