import torch
import torch.optim as optim
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict, defaultdict
from os.path import join as pjoin

from utils.utils import *
from utils.eval_t2m_ddp import evaluation_mask_transformer, evaluation_res_transformer
from data.t2m_dataset import collate_fn
from models.transformer.tools import *


def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, mask_transformer_aux, mask_transformer_ts, vq_model):
        self.opt = args
        self.mask_transformer_aux = mask_transformer_aux
        self.mask_transformer_ts = mask_transformer_ts
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.rank, self.world_size = distributed.get_rank(), distributed.get_world_size()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir) if self.rank == 0 else None

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_mask_transformer_aux.param_groups:
            param_group["lr"] = current_lr
        for param_group in self.opt_mask_transformer_ts.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, joints, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        joints = joints.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx1d, _, code_idx2d, _ = self.vq_model.encode(motion, joints)
        m_lens_4 = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        _loss1, _pred_ids1, _acc1 = self.mask_transformer_aux(code_idx1d[..., 0], conds, m_lens_4)
        _loss2, _pred_ids2, _acc2 = self.mask_transformer_ts(code_idx2d[..., 0], conds, m_lens_4)

        return _loss1, _acc1, _loss2, _acc2

    def update(self, batch_data):
        loss1, acc1, loss2, acc2 = self.forward(batch_data)
        self.opt_mask_transformer_aux.zero_grad()
        loss1.backward()
        self.opt_mask_transformer_aux.step()
        self.scheduler_aux.step()

        self.opt_mask_transformer_ts.zero_grad()
        loss2.backward()
        self.opt_mask_transformer_ts.step()
        self.scheduler_ts.step()

        return loss1.item(), acc1, loss2.item(), acc2

    def save(self, file_name, ep, total_it):
        mask_trans_aux_state_dict = self.mask_transformer_aux.state_dict() if not isinstance(self.mask_transformer_aux, 
            torch.nn.parallel.DistributedDataParallel) else self.mask_transformer_aux.module.state_dict()
        clip_weights1 = [e for e in mask_trans_aux_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights1:
            del mask_trans_aux_state_dict[e]
        mask_trans_ts_state_dict = self.mask_transformer_ts.state_dict() if not isinstance(self.mask_transformer_ts, 
            torch.nn.parallel.DistributedDataParallel) else self.mask_transformer_ts.module.state_dict()
        clip_weights2 = [e for e in mask_trans_ts_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights2:
            del mask_trans_ts_state_dict[e]
        state = {
            'mask_transformer_aux': mask_trans_aux_state_dict,
            'mask_transformer_ts': mask_trans_ts_state_dict,
            'opt_mask_transformer_aux': self.opt_mask_transformer_aux.state_dict(),
            'scheduler_aux':self.scheduler_aux.state_dict(),
            'opt_mask_transformer_ts': self.opt_mask_transformer_ts.state_dict(),
            'scheduler_ts':self.scheduler_ts.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.mask_transformer_aux.module.load_state_dict(checkpoint['mask_transformer_aux'], strict=False)
        missing_keys, unexpected_keys = self.mask_transformer_ts.module.load_state_dict(checkpoint['mask_transformer_ts'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_mask_transformer_aux.load_state_dict(checkpoint['opt_mask_transformer_aux']) # Optimizer
            self.scheduler_aux.load_state_dict(checkpoint['scheduler_aux']) # Scheduler
            self.opt_mask_transformer_ts.load_state_dict(checkpoint['opt_mask_transformer_ts']) # Optimizer
            self.scheduler_ts.load_state_dict(checkpoint['scheduler_ts']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataset, val_dataset, eval_dataset, batch_size, eval_wrapper, plot_eval):
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
        # eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
        eval_sampler = DistributedSampler(
                eval_dataset, shuffle=True)
        eval_val_loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=eval_sampler,
                                collate_fn=collate_fn)
        self.vq_model.to(self.device)

        self.opt_mask_transformer_aux = optim.AdamW([
                                                    {'params': self.mask_transformer_aux.parameters()},
                                                ], betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler_aux = optim.lr_scheduler.MultiStepLR(self.opt_mask_transformer_aux,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)
        self.opt_mask_transformer_ts = optim.AdamW([
                                                    {'params': self.mask_transformer_ts.parameters()},
                                                ], betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler_ts = optim.lr_scheduler.MultiStepLR(self.opt_mask_transformer_ts,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.mask_transformer_aux, self.mask_transformer_ts, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=False, save_anim=True
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.mask_transformer_aux.train()
            self.mask_transformer_ts.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                # train_sampler.set_epoch(epoch)
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                _, _, loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_mask_transformer_ts.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.rank == 0: self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

            if self.rank == 0: self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                self.opt.save_root, eval_val_loader, self.mask_transformer_aux, self.mask_transformer_ts, self.vq_model, self.logger, epoch,
                best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )

            if self.rank == 0:
                print('==> Validation:')
                self.vq_model.eval()
                self.mask_transformer_aux.eval()
                self.mask_transformer_ts.eval()

                val_loss = []
                val_acc = []
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        _, _, loss, acc = self.forward(batch_data)
                        val_loss.append(loss.item())
                        val_acc.append(acc)

                print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

                self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
                self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

                if np.mean(val_acc) > best_acc:
                    print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = np.mean(val_acc)


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer_aux, res_transformer_ts, vq_model):
        self.opt = args
        self.res_transformer_aux = res_transformer_aux
        self.res_transformer_ts = res_transformer_ts
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        try:
            self.rank, self.world_size = distributed.get_rank(), distributed.get_world_size()
        except Exception as e:
            print('===> DDP not initilized, use single GPU')
            self.rank = 0

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir) if self.rank == 0 else None
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer_aux.param_groups:
            param_group["lr"] = current_lr
        for param_group in self.opt_res_transformer_ts.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, joints, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        joints = joints.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, _, code_idx2d, _ = self.vq_model.encode(motion, joints)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss1, pred_ids1, acc1 = self.res_transformer_aux(code_idx, conds, m_lens)
        ce_loss2, pred_ids2, acc2 = self.res_transformer_ts(code_idx2d, conds, m_lens)

        return ce_loss1, acc1, ce_loss2, acc2

    def update(self, batch_data):
        loss1, acc1, loss2, acc2 = self.forward(batch_data)

        self.opt_res_transformer_aux.zero_grad()
        loss1.backward()
        self.opt_res_transformer_aux.step()
        self.scheduler_aux.step()

        self.opt_res_transformer_ts.zero_grad()
        loss2.backward()
        self.opt_res_transformer_ts.step()
        self.scheduler_ts.step()

        return loss1.item(), acc1, loss2.item(), acc2

    def save(self, file_name, ep, total_it):
        res_trans_state_dict_aux = self.res_transformer_aux.state_dict() if not isinstance(self.res_transformer_aux, 
            torch.nn.parallel.DistributedDataParallel) else self.res_transformer_aux.module.state_dict()
        clip_weights1 = [e for e in res_trans_state_dict_aux.keys() if e.startswith('clip_model.')]
        for e in clip_weights1:
            del res_trans_state_dict_aux[e]
        res_trans_state_dict_ts = self.res_transformer_ts.state_dict() if not isinstance(self.res_transformer_ts, 
            torch.nn.parallel.DistributedDataParallel) else self.res_transformer_ts.module.state_dict()
        clip_weights2 = [e for e in res_trans_state_dict_ts.keys() if e.startswith('clip_model.')]
        for e in clip_weights2:
            del res_trans_state_dict_ts[e]
        state = {
            'res_transformer_aux': res_trans_state_dict_aux,
            'opt_res_transformer_aux': self.opt_res_transformer_aux.state_dict(),
            'scheduler_aux':self.scheduler_aux.state_dict(),
            'res_transformer_ts': res_trans_state_dict_ts,
            'opt_res_transformer_ts': self.opt_res_transformer_ts.state_dict(),
            'scheduler_ts':self.scheduler_ts.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer_aux.module.load_state_dict(checkpoint['res_transformer_aux'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        missing_keys, unexpected_keys = self.res_transformer_ts.module.load_state_dict(checkpoint['res_transformer_ts'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer_aux.load_state_dict(checkpoint['opt_res_transformer_aux']) # Optimizer
            self.scheduler_aux.load_state_dict(checkpoint['scheduler_aux']) # Scheduler
            self.opt_res_transformer_ts.load_state_dict(checkpoint['opt_res_transformer_ts']) # Optimizer
            self.scheduler_ts.load_state_dict(checkpoint['scheduler_ts']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataset, val_dataset, eval_dataset, batch_size, eval_wrapper, plot_eval):

        train_sampler = DistributedSampler(
            train_dataset, shuffle=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
        # eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
        eval_sampler = DistributedSampler(
                eval_dataset, shuffle=True)
        eval_val_loader = DataLoader(eval_dataset, batch_size=32, num_workers=4, drop_last=True, sampler=eval_sampler,
                                collate_fn=collate_fn)

        self.res_transformer_aux.to(self.device)
        self.res_transformer_ts.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer_aux = optim.AdamW(self.res_transformer_aux.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler_aux = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer_aux,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)
        self.opt_res_transformer_ts = optim.AdamW(self.res_transformer_ts.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler_ts = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer_ts,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self.res_transformer_aux, self.res_transformer_ts, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer_aux.train()
            self.res_transformer_ts.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                # train_sampler.set_epoch(epoch)
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                _, _, loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer_ts.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if self.rank == 0: self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

            epoch += 1

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, self.res_transformer_aux, self.res_transformer_ts, self.vq_model, self.logger, epoch, 
                best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )

            if self.rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

                print('==> Validation:')
                self.vq_model.eval()
                self.res_transformer_aux.eval()
                self.res_transformer_ts.eval()

                val_loss = []
                val_acc = []
                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        _, _, loss, acc = self.forward(batch_data)
                        val_loss.append(loss.item())
                        val_acc.append(acc)

                print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

                self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
                self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

                if np.mean(val_acc) > best_acc:
                    print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                    self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                    best_acc = np.mean(val_acc)
