import os
import tqdm
import numpy as np
import torch
from utils.metrics import *
import torch.nn.functional as F
from utils.motion_process import recover_from_ric


SAVE_BEST = 10
BEST_FID_LIST = []


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, plot_func, save=True, draw=True):
    net.eval()

    motion_gt_list = []
    motion_pred_list = []

    R_precision_gt = 0
    R_precision_pred = 0
    nb_sample = 0
    matching_score_gt = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion_gt, m_length, token, joints_gt = batch

        motion_gt = motion_gt.cuda()
        joints_gt = joints_gt.to(motion_gt.device)
        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)
        bs, seq = motion_gt.shape[0], motion_gt.shape[1]

        _, _, _, _, _, _, motion_pred = net(motion_gt, joints_gt)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_pred, m_length)
        motion_pred_list.append(em_pred)
        motion_gt_list.append(em_gt)

        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
    diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
    R_precision_gt = R_precision_gt / nb_sample
    matching_score_gt = matching_score_gt / nb_sample

    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    R_precision_pred = R_precision_pred / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    msg = f'==> Eval. Ep {ep}: FID. {fid:.4f}, ' \
            f'Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, ' \
            f'R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, ' \
            f'matching_score_gt. {matching_score_gt}, matching_score_pred. {matching_score_pred}'
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity_pred, ep)
        writer.add_scalar('./Test/top1', R_precision_pred[0], ep)
        writer.add_scalar('./Test/top2', R_precision_pred[1], ep)
        writer.add_scalar('./Test/top3', R_precision_pred[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    # ============= Save the best-10 ================
    if len(BEST_FID_LIST) < SAVE_BEST:
        BEST_FID_LIST.append((ep, fid))
        torch.save({'vq_model': net.state_dict(), 'ep': ep, 'value': fid}, os.path.join(out_dir, f'net_best_fid_ep{ep:04d}.tar'))
    else:
        BEST_FID_LIST.sort(key=lambda tup: tup[1])
        if fid < BEST_FID_LIST[-1][1]:
            ep_rm, fid_rm = BEST_FID_LIST.pop(-1)
            os.system('rm ' + os.path.join(out_dir, f'net_best_fid_ep{ep_rm:04d}.tar'))
            BEST_FID_LIST.append((ep, fid))
            torch.save({'vq_model': net.state_dict(), 'ep': ep, 'value': fid}, os.path.join(out_dir, f'net_best_fid_ep{ep:04d}.tar'))

    if fid < best_fid:
        if draw: print("==> ==> FID Improved from %.5f to %.5f !!!" % (best_fid, fid))
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep, 'value': fid}, os.path.join(out_dir, 'net_best_fid.tar'))
    else:
        if draw: print("==> ==> FID remains %.5f !!!" % (best_fid))

    if R_precision_pred[0] > best_top1:
        msg = "==> ==> Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision_pred[0])
        if draw: print(msg)
        best_top1 = R_precision_pred[0]

    if R_precision_pred[1] > best_top2:
        msg = "==> ==> Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision_pred[1])
        if draw: print(msg)
        best_top2 = R_precision_pred[1]

    if R_precision_pred[2] > best_top3:
        msg = "==> ==> Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision_pred[2])
        if draw: print(msg)
        best_top3 = R_precision_pred[2]

    if matching_score_pred < best_matching:
        msg = f"==> ==> matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred

    save_anim = True
    if save_anim:
        num_plot = 3
        eval_dir = out_dir.replace('/model', '/animation_eval')
        data = torch.cat([motion_gt[:num_plot], motion_pred[:num_plot]], dim=0).detach().cpu().numpy()
        save_dir = os.path.join(eval_dir, 'E%04d' % (ep))
        os.makedirs(save_dir, exist_ok=True)
        plot_func(data, save_dir)

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_gt_list = []
    motion_pred_list = []
    R_precision_gt = 0
    R_precision_pred = 0
    nb_sample = 0
    matching_score_gt = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion_gt, m_length, token, joints_gt = batch

        motion_gt = motion_gt.cuda()
        joints_gt = joints_gt.to(motion_gt.device)
        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)
        bs, seq = motion_gt.shape[0], motion_gt.shape[1]

        _, _, _, _, _, _, motion_pred = net(motion_gt, joints_gt)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_pred, m_length)
        motion_pred_list.append(em_pred)

        bgt = val_loader.dataset.inv_transform(motion_gt.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(motion_pred.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_gt_list.append(em_gt)

        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_gt = R_precision_gt / nb_sample
    R_precision_pred = R_precision_pred / nb_sample

    matching_score_gt = matching_score_gt / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f'==> Eval. Re {repeat_id}: FID. {fid:.4f}, ' \
            f'Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, ' \
            f'R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, ' \
            f'matching_score_gt. {matching_score_gt}, matching_score_pred. {matching_score_pred}, mpjpe: {mpjpe}'
    print(msg)

    return fid, diversity_pred, R_precision_pred, matching_score_pred, mpjpe


@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans_aux, trans_ts, vq_model, writer, ep, 
                           best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
                           eval_wrapper, plot_func, save_ckpt=False, save_anim=False):

    def save(file_name, ep, best_value=None, with_ep=False):
        t2m_trans_aux_state_dict = trans_aux.state_dict() if not isinstance(trans_aux,
            torch.nn.parallel.DistributedDataParallel) else trans_aux.module.state_dict()
        clip_weights1 = [e for e in t2m_trans_aux_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights1:
            del t2m_trans_aux_state_dict[e]
        t2m_trans_ts_state_dict = trans_ts.state_dict() if not isinstance(trans_ts, 
            torch.nn.parallel.DistributedDataParallel) else trans_ts.module.state_dict()
        clip_weights2 = [e for e in t2m_trans_ts_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights2:
            del t2m_trans_ts_state_dict[e]
        state = {
            't2m_transformer_aux': t2m_trans_aux_state_dict,
            't2m_transformer_ts': t2m_trans_ts_state_dict,
            'ep': ep,
            'best_value': best_value,
        }
        if with_ep:
            torch.save(state, file_name.replace('.tar', f'_ep{ep:04d}.tar'))
        else:
            torch.save(state, file_name)

    trans_aux.eval()
    trans_ts.eval()
    vq_model.eval()
    device = trans_ts.device
    rank = torch.distributed.get_rank()

    motion_gt_list = []
    motion_pred_list = []
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    nb_sample = 0
    for batch in tqdm.tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, motion_gt, m_length, token, joints_gt = batch
        m_length = m_length.to(device)

        bs, seq = motion_gt.shape[:2]
        motion_gt = motion_gt.to(device).float()

        if isinstance(trans_ts, torch.nn.parallel.DistributedDataParallel):
            mids_aux = trans_aux.module.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
            mids_ts = trans_ts.module.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1, n_j=6)
        else:
            mids_aux = trans_aux.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
            mids_ts = trans_ts.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1, n_j=6)

        mids_aux.unsqueeze_(-1)
        mids_ts.unsqueeze_(-1)
        _, pred_motions = vq_model.forward_decoder(mids_aux, mids_ts)

        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)
        motion_gt_list.append(em_gt)
        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions, m_length)
        motion_pred_list.append(em_pred)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt = torch.cat(motion_gt_list, dim=0)
    motion_pred = torch.cat(motion_pred_list, dim=0)

    all_motion_gt = [torch.zeros_like(motion_gt) for _ in range(torch.distributed.get_world_size())] # world_size is the total number of gpu processes you are running.
    torch.distributed.all_gather(all_motion_gt, motion_gt)
    all_motion_pred = [torch.zeros_like(motion_pred) for _ in range(torch.distributed.get_world_size())] # world_size is the total number of gpu processes you are running.
    torch.distributed.all_gather(all_motion_pred, motion_pred)

    if rank == 0:
        motion_gt_np = torch.cat(all_motion_gt, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(all_motion_pred, dim=0).cpu().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)

        diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
        diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        R_precision_gt = R_precision_gt / nb_sample
        R_precision_pred = R_precision_pred / nb_sample

        matching_score_gt = matching_score_gt / nb_sample
        matching_score_pred = matching_score_pred / nb_sample

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f'==> Eval. Ep {ep}: FID. {fid:.4f}, ' \
            f'Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, ' \
            f'R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, ' \
            f'matching_score_gt. {matching_score_gt}, matching_score_pred. {matching_score_pred}'
        print(msg)

        if writer is not None:
            writer.add_scalar('./Test/FID', fid, ep)
            writer.add_scalar('./Test/Diversity', diversity_pred, ep)
            writer.add_scalar('./Test/top1', R_precision_pred[0], ep)
            writer.add_scalar('./Test/top2', R_precision_pred[1], ep)
            writer.add_scalar('./Test/top3', R_precision_pred[2], ep)
            writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

        if len(BEST_FID_LIST) < SAVE_BEST:
            BEST_FID_LIST.append((ep, fid))
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid, with_ep=True)
        else:
            BEST_FID_LIST.sort(key=lambda tup: tup[1])
            if fid < BEST_FID_LIST[-1][1]:
                ep_rm, _ = BEST_FID_LIST.pop(-1)
                os.system('rm ' + os.path.join(out_dir, 'model', f'net_best_fid_ep{ep_rm:04d}.tar'))
                BEST_FID_LIST.append((ep, fid))
                save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid, with_ep=True)
            
        if fid < best_fid:
            msg = f"==> ==> FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            print(msg)
            best_fid, best_ep = fid, ep
            if save_ckpt:
                save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid)
        else:
            print("==> ==> FID remains %.5f !!!" % (best_fid))

        if matching_score_pred < best_matching:
            msg = f"==> ==> matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            print(msg)
            best_matching = matching_score_pred

        if abs(diversity_gt - diversity_pred) < abs(diversity_gt - best_div):
            msg = f"==> ==> Diversity Improved from {best_div:.5f} to {diversity_pred:.5f} !!!"
            print(msg)
            best_div = diversity_pred

        if R_precision_pred[0] > best_top1:
            msg = f"==> ==> Top1 Improved from {best_top1:.4f} to {R_precision_pred[0]:.4f} !!!"
            print(msg)
            best_top1 = R_precision_pred[0]

        if R_precision_pred[1] > best_top2:
            msg = f"==> ==> Top2 Improved from {best_top2:.4f} to {R_precision_pred[1]:.4f} !!!"
            print(msg)
            best_top2 = R_precision_pred[1]

        if R_precision_pred[2] > best_top3:
            msg = f"==> ==> Top3 Improved from {best_top3:.4f} to {R_precision_pred[2]:.4f} !!!"
            print(msg)
            best_top3 = R_precision_pred[2]

        if save_anim:
            rand_idx = torch.randint(bs, (3,))
            data = pred_motions[rand_idx].detach().cpu().numpy()
            captions = [clip_text[k] for k in rand_idx]
            lengths = m_length[rand_idx].cpu().numpy()
            save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
            os.makedirs(save_dir, exist_ok=True)
            plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans_aux, trans_ts, vq_model, writer, ep, 
                           best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
                           eval_wrapper, plot_func, save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep, best_value=None, with_ep=False):
        res_trans_aux_state_dict = trans_aux.state_dict() if not isinstance(trans_aux, 
            torch.nn.parallel.DistributedDataParallel) else trans_aux.module.state_dict()
        clip_weights1 = [e for e in res_trans_aux_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights1:
            del res_trans_aux_state_dict[e]
        res_trans_ts_state_dict = trans_ts.state_dict() if not isinstance(trans_ts, 
            torch.nn.parallel.DistributedDataParallel) else trans_ts.module.state_dict()
        clip_weights2 = [e for e in res_trans_ts_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights2:
            del res_trans_ts_state_dict[e]
        state = {
            'res_transformer_aux': res_trans_aux_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'res_transformer_ts': res_trans_ts_state_dict,
            'ep': ep,
            'best_value': best_value,
        }
        if with_ep:
            torch.save(state, file_name.replace('.tar', f'_ep{ep:04d}.tar'))
        else:
            torch.save(state, file_name)

    trans_aux.eval()
    trans_ts.eval()
    vq_model.eval()
    device = next(trans_aux.parameters()).device
    rank = torch.distributed.get_rank()

    motion_gt_list = []
    motion_pred_list = []
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, motion_gt, m_length, token, joints_gt = batch
        m_length = m_length.to(device).long()
        motion_gt = motion_gt.to(device).float()
        joints_gt = joints_gt.to(device).float()

        bs, seq = motion_gt.shape[:2]

        code_indices_aux, _, code_indices_ts, _ = vq_model.encode(motion_gt, joints_gt)
        # (b, seqlen)
        if ep == 0:
            pred_ids_aux = code_indices_aux[..., 0:1]
            pred_ids_ts = code_indices_ts[..., 0:1]
        else:
            if isinstance(trans_aux, torch.nn.parallel.DistributedDataParallel):
                pred_ids_aux = trans_aux.module.generate(code_indices_aux[..., 0], clip_text, m_length//4,
                                        temperature=temperature, cond_scale=cond_scale)
                pred_ids_ts = trans_ts.module.generate(code_indices_ts[..., 0], clip_text, m_length//4,
                                        temperature=temperature, cond_scale=cond_scale)
            else:
                pred_ids_aux = trans_aux.generate(code_indices_aux[..., 0], clip_text, m_length//4,
                                        temperature=temperature, cond_scale=cond_scale)
                pred_ids_ts = trans_ts.generate(code_indices_ts[..., 0], clip_text, m_length//4,
                                        temperature=temperature, cond_scale=cond_scale)

        _, pred_motions = vq_model.forward_decoder(pred_ids_aux, pred_ids_ts)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        motion_gt = motion_gt.to(pred_motions.device).float()
        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)

        motion_gt_list.append(em_gt)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt = torch.cat(motion_gt_list, dim=0)
    motion_pred = torch.cat(motion_pred_list, dim=0)

    all_motion_gt = [torch.zeros_like(motion_gt) for _ in range(torch.distributed.get_world_size())] # world_size is the total number of gpu processes you are running.
    torch.distributed.all_gather(all_motion_gt, motion_gt)
    all_motion_pred = [torch.zeros_like(motion_pred) for _ in range(torch.distributed.get_world_size())] # world_size is the total number of gpu processes you are running.
    torch.distributed.all_gather(all_motion_pred, motion_pred)

    if rank == 0:
        motion_annotation_np = torch.cat(all_motion_gt, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(all_motion_pred, dim=0).cpu().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)

        diversity_gt = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        R_precision_gt = R_precision_gt / nb_sample
        R_precision_pred = R_precision_pred / nb_sample

        matching_score_gt = matching_score_gt / nb_sample
        matching_score_pred = matching_score_pred / nb_sample

        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f'==> Eval. Ep {ep}: FID. {fid:.4f}, ' \
            f'Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, ' \
            f'R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, ' \
            f'matching_score_gt. {matching_score_gt}, matching_score_pred. {matching_score_pred}'
        print(msg)

        # if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity_pred, ep)
        writer.add_scalar('./Test/top1', R_precision_pred[0], ep)
        writer.add_scalar('./Test/top2', R_precision_pred[1], ep)
        writer.add_scalar('./Test/top3', R_precision_pred[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

        if len(BEST_FID_LIST) < SAVE_BEST:
            BEST_FID_LIST.append((ep, fid))
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid, with_ep=True)
        else:
            BEST_FID_LIST.sort(key=lambda tup: tup[1])
            if fid < BEST_FID_LIST[-1][1]:
                ep_rm, fid_rm = BEST_FID_LIST.pop(-1)
                os.system('rm ' + os.path.join(out_dir, 'model', f'net_best_fid_ep{ep_rm:04d}.tar'))
                BEST_FID_LIST.append((ep, fid))
                save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid, with_ep=True)

        if fid < best_fid:
            msg = f"==> ==> FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            print(msg)
            best_fid, best_ep = fid, ep
            if save_ckpt:
                save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep, fid)
        else:
            print("==> ==> FID remains %.5f !!!" % (best_fid))

        if matching_score_pred < best_matching:
            msg = f"==> ==> matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            print(msg)
            best_matching = matching_score_pred

        if abs(diversity_gt - diversity_pred) < abs(diversity_gt - best_div):
            msg = f"==> ==> Diversity Improved from {best_div:.5f} to {diversity_pred:.5f} !!!"
            print(msg)
            best_div = diversity_pred

        if R_precision_pred[0] > best_top1:
            msg = f"==> ==> Top1 Improved from {best_top1:.4f} to {R_precision_pred[0]:.4f} !!!"
            print(msg)
            best_top1 = R_precision_pred[0]

        if R_precision_pred[1] > best_top2:
            msg = f"==> ==> Top2 Improved from {best_top2:.4f} to {R_precision_pred[1]:.4f} !!!"
            print(msg)
            best_top2 = R_precision_pred[1]

        if R_precision_pred[2] > best_top3:
            msg = f"==> ==> Top3 Improved from {best_top3:.4f} to {R_precision_pred[2]:.4f} !!!"
            print(msg)
            best_top3 = R_precision_pred[2]

        if save_anim:
            rand_idx = torch.randint(bs, (3,))
            data = pred_motions[rand_idx].detach().cpu().numpy()
            captions = [clip_text[k] for k in rand_idx]
            lengths = m_length[rand_idx].cpu().numpy()
            save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
            os.makedirs(save_dir, exist_ok=True)
            try:
                plot_func(data, save_dir, captions, lengths)
            except Exception as e:
                print("!!! Exception:", e)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans_aux, trans_ts, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans_aux.eval()
    trans_ts.eval()
    vq_model.eval()

    motion_gt_list = []
    motion_pred_list = []
    motion_multimodality = []
    multimodality = 0
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, motion_gt, m_length, token, joints_gt = batch
        m_length = m_length.cuda()
        B, T, _ = motion_gt.shape
        T = m_length.max()
        J0 = joints_gt.shape[2]
        J = 6

        bs, seq = motion_gt.shape[:2]

        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                mids_aux = trans_aux.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)
                mids_ts = trans_ts.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask, n_j=J)
                mids_aux.unsqueeze_(-1)
                mids_ts.unsqueeze_(-1)
                _, pred_motions = vq_model.forward_decoder(mids_aux, mids_ts)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids_aux = trans_aux.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)
            mids_ts = trans_ts.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask, n_j=J)
            mids_aux.unsqueeze_(-1)
            mids_ts.unsqueeze_(-1)
            _, pred_motions = vq_model.forward_decoder(mids_aux, mids_ts)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        motion_gt = motion_gt.cuda().float()

        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)
        motion_gt_list.append(em_gt)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_gt = R_precision_gt / nb_sample
    R_precision_pred = R_precision_pred / nb_sample

    matching_score_gt = matching_score_gt / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"==> Eval. Repeat {repeat_id}: FID. {fid:.4f}, " \
          f"Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, " \
          f"R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, " \
          f"matching_score_gt. {matching_score_gt:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_pred, R_precision_pred, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_res_transformer_test(val_loader, vq_model, rtrans_aux, rtrans_ts, mtrans_aux, mtrans_ts, 
        repeat_id, eval_wrapper, time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, 
        out_dir=None, plot_func=None, cal_mm=True, res_cond_scale=5):
    mtrans_aux.eval()
    mtrans_ts.eval()
    vq_model.eval()
    rtrans_aux.eval()
    rtrans_ts.eval()

    motion_gt_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0
    multimodality = 0
    J = 6

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, motion_gt, m_length, token, joints_gt = batch
        m_length = m_length.cuda()

        bs, seq = motion_gt.shape[:2]

        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                mids_aux = mtrans_aux.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)
                mids_ts = mtrans_ts.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask, n_j=J)
                pred_ids_aux = rtrans_aux.generate(mids_aux, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                pred_ids_ts = rtrans_ts.generate(mids_ts, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                _, pred_motions = vq_model.forward_decoder(pred_ids_aux, pred_ids_ts)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids_aux = mtrans_aux.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)
            mids_ts = mtrans_ts.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask, n_j=J)
            pred_ids_aux = rtrans_aux.generate(mids_aux, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            pred_ids_ts = rtrans_ts.generate(mids_ts, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            _, pred_motions = vq_model.forward_decoder(pred_ids_aux, pred_ids_ts)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        motion_gt = motion_gt.cuda().float()

        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_gt, m_length)
        motion_gt_list.append(em_gt)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

        save_anim = False
        if save_anim and plot_func is not None:
            rand_idx = torch.arange(32)
            data = pred_motions.detach().cpu().numpy()
            captions = [clip_text[k] for k in rand_idx]
            lengths = m_length[rand_idx].cpu().numpy()
            save_dir = os.path.join(out_dir.replace('/eval', '/eval_anim'), 'E%04d' % i)
            os.makedirs(save_dir, exist_ok=True)
            data_inv = plot_func(data[rand_idx], save_dir, captions, lengths)
            np.save(os.path.join(save_dir, 'motion.npy'), data_inv)
            data = motion_gt.detach().cpu().numpy()
            save_dir = os.path.join(out_dir.replace('/eval', '/eval_anim'), 'E%04d_gt' % i)
            os.makedirs(save_dir, exist_ok=True)
            data_inv = plot_func(data[rand_idx], save_dir, captions, lengths)
            np.save(os.path.join(save_dir, 'motion_gt.npy'), data_inv)
            with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
                for i, t in enumerate(clip_text):
                    f.write('%s,%d\n' % (t, lengths[i]))

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_gt = R_precision_gt / nb_sample
    R_precision_pred = R_precision_pred / nb_sample

    matching_score_gt = matching_score_gt / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"==> Eval. Repeat {repeat_id}: FID. {fid:.4f}, " \
          f"Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, " \
          f"R_precision_gt. {R_precision_gt}, R_precision_pred. {R_precision_pred}, " \
          f"matching_score_gt. {matching_score_gt:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity_pred, R_precision_pred, matching_score_pred, multimodality