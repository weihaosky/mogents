from torch.utils.data import DataLoader

import numpy as np
from os.path import join as pjoin

from data.t2m_dataset import Text2MotionDatasetEval, collate_fn # TODO
from utils.word_vectorizer import WordVectorizer
from utils.get_opt import get_opt


def get_dataset_motion_loader(opt_path, batch_size, fname, device):
    opt = get_opt(opt_path, device)

    if opt.dataset_name == 't2m' or opt.dataset_name == 'humanml3d' or opt.dataset_name == 'kit' or opt.dataset_name == 'motionx':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./checkpoints/glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset