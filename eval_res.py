import os
from os.path import join as pjoin

import torch

from models.transformer.transformer_aux import MaskTransformer, ResidualTransformer
from models.transformer.transformer_ts import MaskTransformer2D, ResidualTransformer2D
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain
from utils.plot_script import plot_3d_motion
import utils.eval_t2m_ddp as eval_t2m
from utils.fixseed import fixseed

import numpy as np


def plot_t2m(data, save_dir, captions, m_lengths):
    data = eval_dataset.inv_transform(data)
    kinematic_chain = kit_kinematic_chain if opt.dataset_name == 'kit' else t2m_kinematic_chain

    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.nb_joints).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        try:
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
        except Exception as e:
            print('Exception:', e)
    return data


def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
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
    print(msg, file=f, flush=True)
    return res_transformer


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.mtrans_name)
    if not os.path.exists(root_dir): exit()
    model_dir = pjoin(root_dir, 'model')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens1d = vq_model.num_code1d
    model_opt.num_tokens2d = vq_model.num_code2d

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.rtrans_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_opt.num_tokens2d = vq_model.num_code2d

    assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/humanml3d/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, eval_dataset = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    if not opt.traverse_res:
        file_res = opt.which_ckpt
        out_dir = pjoin(root_dir, 'eval')
        os.makedirs(out_dir, exist_ok=True)
        out_path = pjoin(out_dir, "%s.log"%opt.ext)
        f = open(pjoin(out_path), 'a')
        print('===========================================', file=f, flush=True)
        res_transformer_aux = load_res_aux(res_opt, file_res)
        res_transformer_ts = load_res_ts(res_opt, file_res)
    else:
        model_dir = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')
        file = opt.which_ckpt
        out_dir = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'eval')
        os.makedirs(out_dir, exist_ok=True)
        out_path = pjoin(out_dir, "%s.log"%opt.ext)
        f = open(pjoin(out_path), 'a')
        print('===========================================', file=f, flush=True)
        mask_transformer_aux = load_trans_aux(model_opt, file)
        mask_transformer_ts = load_trans_ts(model_opt, file)

    print(opt.mtrans_name, file=f, flush=True)
    print(opt.rtrans_name, file=f, flush=True)
    print(opt.which_ckpt, file=f, flush=True)

    # model_dir = pjoin(opt.)
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        if not opt.traverse_res:
            mask_transformer_aux = load_trans_aux(model_opt, file)
            mask_transformer_ts = load_trans_ts(model_opt, file)
        else:
            res_transformer_aux = load_res_aux(res_opt, file)
            res_transformer_ts = load_res_ts(res_opt, file)
        mask_transformer_aux.eval()
        mask_transformer_ts.eval()
        vq_model.eval()
        res_transformer_aux.eval()
        res_transformer_ts.eval()

        mask_transformer_aux.to(opt.device)
        mask_transformer_ts.to(opt.device)
        vq_model.to(opt.device)
        res_transformer_aux.to(opt.device)
        res_transformer_ts.to(opt.device)

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []

        repeat_time = 20
        for i in range(repeat_time):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, best_mm = \
                    eval_t2m.evaluation_mask_res_transformer_test(eval_val_loader, vq_model, 
                            res_transformer_aux, res_transformer_ts, mask_transformer_aux, mask_transformer_ts,
                            i, eval_wrapper=eval_wrapper, out_dir=out_dir, plot_func=plot_t2m,
                            time_steps=opt.time_steps, cond_scale=opt.cond_scale, temperature=opt.temperature, 
                            topkr=opt.topkr, force_mask=opt.force_mask, cal_mm=True)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
