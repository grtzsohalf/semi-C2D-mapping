import torch
import torch.utils.data
import numpy as np
import os
import _pickle as pk
import random

import sys


class myDataset(torch.utils.data.Dataset):

    def __init__(self, feat_dir, phn_boundary_path=None):
        super(myDataset).__init__()
        self.feat_dir = feat_dir
        if phn_boundary_path is not None:
            self.phn_boundary = pk.load(open(phn_boundary_path, 'rb'))
        else:
            self.phn_boundary = None
        print('finish loading data')

    def __len__(self):
        return len(os.listdir(self.feat_dir))

    def __getitem__(self, idx):
        feat = pk.load(open(os.path.join(self.feat_dir, str(idx+1)+'.mfcc.pkl'),'rb'))
        if self.phn_boundary is not None:
            return feat, len(feat), self.phn_boundary[idx+1]
        else:
            return feat, len(feat), None

    @staticmethod
    def get_collate_fn(prediction_num, neg_num, reduce_times, train=True):

        def _pad_sequence(np_list, length=None):
            tensor_list = [torch.from_numpy(np_array) for np_array in np_list]
            #print('tensor length: ', len(tensor_list))
            #for tensor in tensor_list:
            #    print('shape', tensor.size())
            pad_tensor = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
            if length is None:
                return pad_tensor
            else:
                pad_length = pad_tensor.size()[1]
                if pad_length >= length:
                    return pad_tensor[:, :length, :]
                else:
                    pad = torch.zeros([pad_tensor.size()[0], length-pad_length, pad_tensor.size()[2]])
                    return torch.cat([pad_tensor, pad], 1)

        def collate_fn(batch):
            # for dataloader
            if train:
                batch =sorted(batch, key=lambda x:x[1], reverse=True)
            all_feat, all_length, phn_boundary_list = zip(*batch)
            all_feat = _pad_sequence(all_feat)
            all_length = torch.tensor(all_length)

            tensor_length = all_feat.size()[1]
            interval = 2**reduce_times
            neg_shift = random.sample(range((prediction_num+1)*interval, tensor_length), neg_num)
            neg_shift = torch.tensor(neg_shift)

            return all_feat.float(), all_length, neg_shift, phn_boundary_list

        return collate_fn


if __name__ == '__main__':
    path = '/home/darong/darong/data/constrasive/processed_ls/feature'
    prediction_num=3
    neg_num=5
    reduce_times=2

    dataset = myDataset(path)
    collate_fn = myDataset.get_collate_fn(prediction_num, neg_num, reduce_times)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=collate_fn)
    for i, data in enumerate(data_loader):
        print('i', i)
        print('feat shape:', data[0].shape)
        print('len shape:', data[1])
        print('neg shape:', data[2])
        print('neg shape:', data[3])
        #if i == 10:
        #    break
