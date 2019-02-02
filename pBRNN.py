import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# The modules for a pBGRU encoder
# ==============================
#


######################################################################
# The pBGRU module
# ----------------
#
# Implement a pyramidal bi-directional GRU (pBGRU) (ref: LAS: 
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7472621)
#
# If the argument subsample <= 1, this class would be simply a stack of GRUs.
#

class pBGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate):
        super(pBGRU, self).__init__()

        layers, project_layers = [], []
        for i in range(n_layers):
            dim = input_dim if i == 0 else hidden_dim
            project_dim = hidden_dim * 4 if subsample[i] > 1 else hidden_dim * 2
            layers.append(nn.GRU(dim, hidden_dim, num_layers=1,
                bidirectional=True, batch_first=True))
            project_layers.append(nn.Linear(project_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)
        self.project_layers = nn.ModuleList(project_layers)
        self.subsample = subsample
        self.dropout_rate = dropout_rate

    def forward(self, input_pad, input_lens):
        for i, (layer, project_layer) in enumerate(zip(self.layers, self.project_layers)):
            # pack sequence 
            input_pack = pack_padded_sequence(input_pad, input_lens, batch_first=True)
            inputs, _ = layer(input_pack)
            output_pad, input_lens = pad_packed_sequence(inputs, batch_first=True)
            #input_pad = dropout_layer(input_pad)
            input_lens = input_lens.numpy()

            # subsampling
            sub = self.subsample[i]
            if sub > 1:
                # pad one frame
                if output_pad.size(1) % 2 == 1:
                    output_pad = F.pad(output_pad.transpose(1, 2), (0, 1), mode='replicate').transpose(1, 2)
                # concat two frames
                output_pad = output_pad.contiguous().view(output_pad.size(0), output_pad.size(1) // 2, output_pad.size(2) * 2)
                input_lens = [(length + 1) // sub for length in input_lens]

            projected = project_layer(output_pad)
            input_pad = F.relu(projected)
            input_pad = F.dropout(input_pad, self.dropout_rate, training=self.training)

        # type to list of int
        input_lens = np.array(input_lens, dtype=np.int64).tolist()
        return input_pad, input_lens


######################################################################
# The pBGRU encoder module
# ------------------------
#

class pBGRUEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate, in_channel=1):
        super(pBGRUEncoder, self).__init__()
        self.enc = pBGRU(input_dim=input_dim, hidden_dim=hidden_dim, 
                n_layers=n_layers, subsample=subsample, dropout_rate=dropout_rate)

    def forward(self, inputs, input_lens):
        outputs, output_lens = self.enc(inputs, input_lens)
        return outputs, output_lens
