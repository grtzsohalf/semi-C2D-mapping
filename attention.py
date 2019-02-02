import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import weight_init


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# The attention module
# ==================
#

class Attention(nn.Module):
    """
    Inputs: output, context
        - **output** (batch_size, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch_size, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch_size, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch_size, output_len, input_len): tensor containing attention weights.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.linear_out.apply(weight_init)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # (batch_size, out_len, dim) * (batch_size, in_len, dim) -> (batch_size, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch_size, out_len, in_len) * (batch_size, in_len, dim) -> (batch_size, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch_size, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch_size, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn



######################################################################
# The attention with location-aware module
# ----------------------------------------
#
