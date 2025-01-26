import os
from os.path import join as pjoin
import torch
import numpy as np

from torch import distributed

from options.train_option import TrainT2MOptions
from data.t2m_dataset import Text2MotionDataset, Text2MotionDatasetEval
from models.transformer.transformer_aux import ResidualTransformer
from models.transformer.transformer_ts import ResidualTransformer2D
from models.transformer.transformer_trainer_ddp import ResidualTransformerTrainer
from models.vq.model import RVQVAE
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import humanml3d_kinematic_chain, kit_kinematic_chain
from utils.word_vectorizer import WordVectorizer


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
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
    print(f'Loading VQ Model {opt.vq_name}', ckpt['ep'], ckpt['value'])
    vq_model.to(opt.device)
    return vq_model, vq_opt


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    # fixseed(opt.seed)

    # opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    # torch.autograd.set_detect_anomaly(True)

    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed.init_process_group("nccl")
        device_id = rank
        device = torch.device(device_id)
    except KeyError:
        print('Exception: DDP Init Error !!!')
        world_size = 1
        rank = 0
        local_rank = 0
        distributed.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
    opt.device = device

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        # os.makedirs(opt.meta_dir, exist_ok=True)
        os.makedirs(opt.eval_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 'humanml3d':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = humanml3d_kinematic_chain
        dataset_opt_path = './checkpoints/humanml3d/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': #TODO
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_quantizers = vq_opt.num_quantizers
    opt.num_tokens1d = vq_model.num_code1d
    opt.num_tokens2d = vq_model.num_code2d

    res_transformer_aux = ResidualTransformer(code_dim=vq_opt.code_dim1d,
                                          cond_mode='text',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)

    res_transformer_ts = ResidualTransformer2D(code_dim=vq_model.code_dim2d,
                                          cond_mode='text',
                                          latent_dim=opt.latent_dim,
                                          ff_size=opt.ff_size,
                                          num_layers=opt.n_layers,
                                          num_heads=opt.n_heads,
                                          dropout=opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=opt.cond_drop_prob,
                                          # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=opt.share_weight,
                                          clip_version=clip_version,
                                          opt=opt)


    all_params = 0
    pc_transformer_aux = sum(param.numel() for param in res_transformer_aux.parameters_wo_clip())
    all_params += pc_transformer_aux
    pc_transformer_ts = sum(param.numel() for param in res_transformer_ts.parameters_wo_clip())
    all_params += pc_transformer_ts
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    res_transformer_aux = torch.nn.parallel.DistributedDataParallel(res_transformer_aux.to(device), device_ids=[rank])
    res_transformer_ts = torch.nn.parallel.DistributedDataParallel(res_transformer_ts.to(device), device_ids=[rank])

    print('===> device:', opt.device, 'rank:', rank, 'world_size:', world_size)

    eval_opt = get_opt(dataset_opt_path, device)
    mean = np.load(pjoin(eval_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(eval_opt.meta_dir, 'std.npy'))
    w_vectorizer = WordVectorizer('./checkpoints/glove', 'our_vab')
    split_file = pjoin(eval_opt.data_root, 'val.txt')
    eval_dataset = Text2MotionDatasetEval(eval_opt, mean, std, split_file, w_vectorizer)

    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = ResidualTransformerTrainer(opt, res_transformer_aux, res_transformer_ts, vq_model)

    trainer.train(train_dataset, val_dataset, eval_dataset, batch_size=opt.batch_size, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)

    distributed.destroy_process_group()

    exit()