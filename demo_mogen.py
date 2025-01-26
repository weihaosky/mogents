import os
from os.path import join as pjoin
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.transformer.transformer_aux import MaskTransformer, ResidualTransformer
from models.transformer.transformer_ts import MaskTransformer2D, ResidualTransformer2D
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from visualization.joints2bvh import Joint2BVHConvertor

clip_version = 'ViT-B/32'


def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!', ckpt['ep'], ckpt['value'])
    return vq_model, vq_opt


def load_trans_aux(model_opt, which_model):
    transformer = MaskTransformer(code_dim=vq_model.code_dim1d,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = transformer.load_state_dict(ckpt['t2m_transformer_aux'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {model_opt.name} from epoch {ckpt["ep"]}!', 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else '')
    return transformer


def load_trans_ts(model_opt, which_model):
    transformer = MaskTransformer2D(code_dim=vq_model.code_dim2d,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = transformer.load_state_dict(ckpt['t2m_transformer_ts'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {model_opt.name} from epoch {ckpt["ep"]}!', 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else '')
    return transformer


def load_res_aux(res_opt, which_model):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens1d = vq_model.num_code1d
    res_transformer = ResidualTransformer(code_dim=vq_model.code_dim1d,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer_aux'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


def load_res_ts(res_opt, which_model):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens2d = vq_model.num_code2d
    res_transformer = ResidualTransformer2D(code_dim=vq_model.code_dim2d,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', which_model),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer_ts'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    msg = f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!', 'Value: ', ckpt['best_value'] if 'best_value' in ckpt.keys() else ''
    print(msg)
    return res_transformer


def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin('checkpoints', opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.mtrans_name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./outputs', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens1d = vq_model.num_code1d
    model_opt.num_tokens2d = vq_model.num_code2d

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.rtrans_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_opt.num_tokens2d = vq_model.num_code2d
    assert res_opt.vq_name == model_opt.vq_name

    res_transformer_aux = load_res_aux(res_opt, 'net_best_fid.tar')
    res_transformer_ts = load_res_ts(res_opt, 'net_best_fid.tar')

    #################################
    ######Loading M-Transformer######
    #################################
    mask_transformer_aux = load_trans_aux(model_opt, 'net_best_fid.tar')
    mask_transformer_ts = load_trans_ts(model_opt, 'net_best_fid.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    mask_transformer_aux.eval()
    mask_transformer_ts.eval()
    vq_model.eval()
    res_transformer_aux.eval()
    res_transformer_ts.eval()
    length_estimator.eval()

    mask_transformer_aux.to(opt.device)
    mask_transformer_ts.to(opt.device)
    vq_model.to(opt.device)
    res_transformer_aux.to(opt.device)
    res_transformer_ts.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if line.startswith('#'): continue
                infos = line.split('=')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = mask_transformer_ts.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    J = 6

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids_aux = mask_transformer_aux.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample)
            mids_ts = mask_transformer_ts.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample, n_j=J)
            mids_aux = res_transformer_aux.generate(mids_aux, captions, token_lens, temperature=1, cond_scale=5)
            mids_ts = res_transformer_ts.generate(mids_ts, captions, token_lens, temperature=1, cond_scale=5)
            _, pred_motions = vq_model.forward_decoder(mids_aux, mids_ts)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = inv_transform(pred_motions)

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, caption[:100])
            joint_path = pjoin(joints_dir, caption[:100])

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)
            np.save(pjoin(animation_path, "sample%d_repeat%d_len%d_data.npy"%(k, r, m_length[k])), joint_data)