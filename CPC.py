import torch
import torch.nn as nn

from NCE_loss import *
from utils import *

class CPC(nn.Module):
    """ Implement a contrastive predictive coding model stated in:
        https://arxiv.org/abs/1807.03748.
    Args:
        input_dim:
        hidden_dim:
        prediction_num:
    Inputs: context, length, neg_shift, train
        - **context** :
        - **length** :
        - **neg_shift** :
        - **train** :
    Outputs: NCE_loss, context 
        - **NCE_loss** :
        - **context** :
    """

    def __init__(self, input_dim, hidden_dim, prediction_num=12):
        super(CPC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_num = prediction_num

        self.NCE_loss_layer = CountNCELoss(self.hidden_dim, self.hidden_dim, prediction_num=prediction_num)

    def forward(self, context, length, neg_shift, train=True):
        if train:
            return self.NCE_loss_layer(context, context, neg_shift, length), None
        else:
            return None, context

if __name__ == '__main__':
    batch = 3
    len_ = 100
    input_dim = 4
    feat_dim = 6

    test_feat = torch.rand([batch, len_, input_dim], dtype=torch.float32).cuda()
    shift = torch.tensor([1,2,3]).cuda()
    length = torch.tensor([90, 80, 60]).cuda()
    m = CPC(input_dim, feat_dim)
    m.cuda()
    loss, _ = m(test_feat, length, shift)
    print(loss)
    """
    test_feat = torch.rand([batch, len_, dim], dtype=torch.float32).cuda()
    print(test_feat)
    space = torch.tensor(1).cuda()
    m = shift(test_feat, space)
    m = m.cpu()
    print(m)
    """
